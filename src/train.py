import re
import os
import sys
import math
import json
import torch
import random
import pickle
import argparse
import datetime

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import periflow_sdk as pf

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback, ModelSummary
from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin

from model import T5BiEncoder, T5FineTuner, T5JointTuner
from pathlib import Path
from typing import Any, Optional, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT

from knockknock import slack_sender
from slack import get_webhook_url, get_channel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class PeriFlowCallback(Callback):
    def on_train_batch_start(self,
                             trainer: pl.Trainer,
                             pl_module: pl.LightningModule,
                             batch: Any,
                             batch_idx: int,
                             unused: int = 0) -> None:
        pf.start_step()

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: STEP_OUTPUT,
                           batch: Any,
                           batch_idx: int,
                           unused: int = 0) -> None:
        loss = float(outputs['loss'])
        pf.metric({
            "iteration": trainer.global_step,
            "loss": loss,
        })
        pf.end_step()

class PeriFlowTrainer(Trainer):
    def save_checkpoint(self,
                        filepath: Union[str, Path],
                        weights_only: bool = False,
                        storage_options: Optional[Any] = None) -> None:
        super().save_checkpoint(filepath, weights_only=weights_only, storage_options=storage_options)
        pf.upload_checkpoint()

@slack_sender(webhook_url=get_webhook_url(), channel=get_channel())
def main(args, train_params):
    sys.setrecursionlimit(10000)
    set_seed(args.seed)
    if args.model_type == "bi":
        model = T5BiEncoder(args)
    elif args.model_type == "joint":
        model = T5JointTuner(args)
    elif args.model_type == "gr":
        model = T5FineTuner(args)
    else:
        assert False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if torch.cuda.current_device() == 0:
        print('='*80)
        print(f"# of trainable parameters: {trainable_params}\n# of total parameters: {total_params}")
        print('='*80)

    if args.periflow:
        print(f'Using Periflow..')
        periflow_callback = PeriFlowCallback()
        train_params["callbacks"] = [periflow_callback, checkpoint_callback]
        train_params["enable_checkpointing"] = isinstance(checkpoint_callback, ModelCheckpoint)

        datalen = len(pd.DataFrame(pickle.load(open(os.path.join(args.dataset, args.train_file), "rb"))))
        num_steps_per_epoch = math.ceil(datalen / args.num_train_epochs)
        pf.init(total_train_steps=args.num_train_epochs * num_steps_per_epoch)

        trainer = PeriFlowTrainer(
            **train_params
        )

    else:
        trainer = pl.Trainer(**train_params)

    if args.do_train:
        if torch.cuda.current_device() == 0:
            now = datetime.datetime.now()
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Start Training...")
        if args.periflow:
            trainer.fit(model, ckpt_path=ckpt_path)

        else:
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
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--test_name", default=None, type=str)
    arg_ = parser.parse_args()

    with open(arg_.config) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)

    if hparam.wandb_log and hparam.do_train:
        wandb_logger = WandbLogger(
            project=hparam.wandb_project, name=hparam.wandb_run_name, save_dir="../wandb"
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
        max_context_length=hparam.max_context_length,
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
        test_file=arg_.test_file if arg_.test_file else hparam.test_file,
        constrained_decoding=True,
        do_train=hparam.do_train,
        do_test=hparam.do_test,
        test_model_path=hparam.test_model_path,
        test_name=arg_.test_name if arg_.test_name else hparam.test_name,
        val_beam_size=hparam.val_beam_size,
        freeze_encoder=hparam.freeze_encoder,
        freeze_vocab_emb=hparam.freeze_vocab_emb,
        contextualized_emb_num=hparam.contextualized_emb_num,  # new
        contextualized_file=hparam.contextualized_file,  # new - tokId_emb.pickle
        groupId2tokIdList=hparam.groupId2tokIdList,  # new - tokGroupId_tokIdList.pickle 
        tokId2groupId=hparam.tokId2groupId,  # new - tokId_tokGroupId.pickle 
        tokId2tokText=hparam.tokId2tokText,  # new - tokId_tokText.pickle 
        tokId2corpus=hparam.tokId2corpus,  # new - tokId_corpus.pickle 
        tree_type=hparam.tree_type,  # new - nodeId_tokIdList.pickle
        tree_path=hparam.tree_path, # new
        nodeId_sup=hparam.nodeId_sup, # new
        embedding_model=hparam.embedding_model, # new - model used to extract embedding
        max_beam_search=hparam.max_beam_search, # new - select a token which has maximum score in groupId
        model_type=hparam.model_type, # new - bi-encoder Training
        periflow=hparam.periflow, # new - periflow
        periflow_dir=hparam.periflow_dir, # new - directory of periflow
        limit_val_batches=hparam.limit_val_batches,
        train_c_emb=hparam.train_c_emb,
        bi_type=hparam.bi_type,
        gr_decoder_only=hparam.gr_decoder_only,
        gr_decoder_only_encoder_ckpt=hparam.gr_decoder_only_encoder_ckpt,
    )
    args = argparse.Namespace(**args_dict)
    assert not (args.do_train and args.do_test), "Choose between train|test"
    if args.model_type == "gr": assert args.tree_type in ["groupId", "nodeId", "clusterId"] 
    if args.model_type == "bi": assert args.accelerator == "ddp", "ddp is only supported for bi-encoder!"
    if args.model_type == "joint" and args.do_test:
        assert args.eval_batch_size == 1, "Batch Size larger than 1 is not implemented yet!"
    assert args.model_type in ["gr", "bi", "joint"]

    if torch.cuda.current_device() == 0:
        print("#" * 80)
        print(args)
        print("#" * 80)

    callbacks = []
    if args.periflow:
        if args.periflow_dir is not None:
            # When use PeriFlow with PyTorch Lightning, do not save the checkpoint twice (i.e., save_top_k > 0 && save_last = True)
            checkpoint_callback = ModelCheckpoint(
                dirpath=args.periflow_dir,
                filename="checkpoint-{step:07d}",
                save_last=False,
                every_n_epochs=1,
                save_top_k=1,
            )
            pattern = re.compile(r"step=(\d+)")
            checkpoint_iter = None
            for ckpt_path in Path(args.periflow_dir).glob("**/*"):
                step = int(pattern.findall(ckpt_path.name)[0])
                if checkpoint_iter is None:
                    checkpoint_iter = step
                else:
                    checkpoint_iter = max(checkpoint_iter, step)

            if checkpoint_iter is not None:
                ckpt_path = checkpoint_callback.format_checkpoint_name(dict(step=checkpoint_iter))
            else:
                ckpt_path = None
        else:
            checkpoint_callback = Callback()
            ckpt_path = None
    else:
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
        args.fp16 = False
        if torch.cuda.current_device() == 0:
            print(f"@@@ Using DDP without FP16")
    elif args.accelerator == "deepspeed":
        plugins = DeepSpeedPlugin(stage=2, load_full_weights=True)
        fp_16 = True
        args.fp16 = True
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
        limit_val_batches=args.limit_val_batches
    )
    main(args, train_params)
