import os
import sys
import json
import torch
import random
import argparse
import datetime

import numpy as np
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin

from model import T5FineTuner

#from knockknock import slack_sender
#from slack import get_webhook_url, get_channel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#@slack_sender(webhook_url=get_webhook_url(), channel=get_channel())
def main(args, train_params):
    sys.setrecursionlimit(10000)
    set_seed(args.seed)
    model = T5FineTuner(args)
    trainer = pl.Trainer(**train_params)
    if args.do_train:
        if torch.cuda.current_device() == 0:
            now = datetime.datetime.now()
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Start Training...")
        if args.resume_from_checkpoint is None:
            trainer.fit(model)
        else:
            print(f"@@@ Resume Training from {args.resume_from_checkpoint}")
            trainer.fit(model, ckpt_path=args.resume_from_checkpoint)
        now = datetime.datetime.now()
        print(
            f"{torch.cuda.current_device()} // [{now.strftime('%Y-%m-%d %H:%M:%S')}] Done Training..."
        )
    if args.do_test:
        if torch.cuda.current_device() == 0:
            now = datetime.datetime.now()
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Start Testing...")
        trainer.test(model)
        now = datetime.datetime.now()
        print(
            f"{torch.cuda.current_device()} // [{now.strftime('%Y-%m-%d %H:%M:%S')}] Done Testing... "
        )
    return args.output_dir


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default=None, required=True, type=str)
    arg_ = parser.parse_args()

    with open(arg_.config) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)

    if hparam.wandb_log and hparam.do_train:
        wandb_logger = WandbLogger(
            project=hparam.wandb_project, name=hparam.wandb_run_name
        )
    else:
        wandb_logger = None

    args_dict = dict(
        output_dir=hparam.output_dir,
        dataset=hparam.dataset,
        model_name_or_path=hparam.model,
        tokenizer_name_or_path=hparam.model,
        max_input_length=hparam.max_input_length,
        max_output_length=hparam.max_output_length,
        learning_rate=hparam.learning_rate,
        lr_scheduler=hparam.lr_scheduler,  # exponential, constant
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        accelerator=hparam.accelerator,  # ddp or deepspeed
        num_train_epochs=hparam.num_train_epochs,
        train_batch_size=hparam.train_batch_size,
        eval_batch_size=hparam.eval_batch_size,
        gradient_accumulation_steps=hparam.gradient_accumulation_steps,
        n_gpu=hparam.n_gpu,
        num_workers=hparam.num_workers,
        resume_from_checkpoint=hparam.resume_from_checkpoint,
        val_check_interval=1.0,
        early_stop_callback=False,
        seed=42,
        check_val_every_n_epoch=hparam.check_val_every_n_epoch,
        train_file=hparam.train_file,
        dev_file=hparam.dev_file,
        test_file=hparam.test_file,
        prefix_tree_file=hparam.prefix_tree_file,
        constrained_decoding=True,
        do_train=hparam.do_train,
        do_test=hparam.do_test,
        test_model_path=hparam.test_model_path,
        val_beam_size=hparam.val_beam_size,
        freeze_encoder=hparam.freeze_encoder,
        freeze_vocab_emb=hparam.freeze_vocab_emb,
        contextualized_emb_num=hparam.contextualized_emb_num,  # new
        contextualized_file=hparam.contextualized_file,  # new - tokId_emb.pickle
        groupId2tokIdList=hparam.groupId2tokIdList,  # new - tokGroupId_tokIdList.pickle 
        tokId2groupId=hparam.tokId2groupId,  # new - tokId_tokGroupId.pickle 
        tokId2tokText=hparam.tokId2tokText,  # new - tokId_tokText.pickle 
        nodeId_tokIdList=hparam.nodeId_tokIdList,  # new - nodeId_tokIdList.pickle
        groupId_tree=hparam.groupId_tree, # new
        nodeId_tree=hparam.nodeId_tree, # new
        embedding_model=hparam.embedding_model # new - model used to extract embedding
    )
    args = argparse.Namespace(**args_dict)
    assert not (args.do_train and args.do_test), "Choose between train|test"
    assert not (args.groupId_tree and args.nodeId_tree), "Choose between groupId|nodeId: groupId for previous version and nodeId for new one"
    assert args.embedding_model in ["t5", "bert"]
    assert args.model_name_or_path in ['t5-base']

    if torch.cuda.current_device() == 0:
        print("#" * 80)
        print(args)
        print("#" * 80)

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        monitor="val em",
        mode="max",
        dirpath=args.output_dir,
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=5,
    )
    callbacks.append(checkpoint_callback)

    if args.lr_scheduler == "constant" and torch.cuda.current_device() == 0:
        print(f"@@@ Not Using Learning Rate Scheduler")
    else:
        lr_callback = LearningRateMonitor()
        callbacks.append(lr_callback)

    if args.accelerator == "ddp":
        plugins = DDPPlugin(find_unused_parameters=False)
        fp_16 = False
        if torch.cuda.current_device() == 0:
            print(f"@@@ Using DDP without FP16")
    elif args.accelerator == "deepspeed":
        plugins = DeepSpeedPlugin(stage=2, load_full_weights=True)
        fp_16 = True
        if torch.cuda.current_device() == 0:
            print(f"@@@ Using Deepspeed stage2 with FP16")
    else:
        raise NotImplementedError("** accelerator: Choose between (ddp|deepspeed)")

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        strategy=plugins,
        max_epochs=args.num_train_epochs,
        precision=16 if fp_16 else 32,
        default_root_dir=args.output_dir,
        checkpoint_callback=True,
        val_check_interval=args.val_check_interval,
        logger=wandb_logger,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
    )

    main(args, train_params)
