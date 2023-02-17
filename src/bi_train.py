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

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback, ModelSummary
from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin

from biT5_model import BiT5Model
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


# @slack_sender(webhook_url=get_webhook_url(), channel=get_channel())
def main(args, train_params):
    sys.setrecursionlimit(10000)
    set_seed(args.seed)
    model = BiT5Model(args)
    print("using BiT5Model")

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

def _count_toknum(args):
    corpus = pd.read_csv(args.corpus_file)["corpus"] 
    for elem in corpus:
        tok = T 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default=None, required=True, type=str)
    parser.add_argument("--save_model_only", action="store_true")
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--test_name", default=None, type=str)
    parser.add_argument("--test_ret_num", default=None, type=int)
    parser.add_argument("--test_batch", default=None, type=int)
    parser.add_argument("--test_beam_size", default=None, type=int)
    arg_ = parser.parse_args()

    with open(arg_.config) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)

    if hparam.wandb_log and hparam.do_train:
        wandb_logger = WandbLogger(
            project=hparam.wandb_project, name=hparam.wandb_run_name, save_dir="/mnt/wandb"
        )
    else:
        wandb_logger = None

    args_dict = dict(
        output_dir=hparam.output_dir,
        dataset=hparam.dataset,
        model_name_or_path=hparam.model,
        tokenizer_name_or_path=hparam.model,
        doc_encoder_model=hparam.doc_encoder_model if "doc_encoder_model" in hparam else None,
        max_input_length=hparam.max_input_length,
        max_output_length=hparam.max_output_length,
        max_context_length=hparam.max_context_length if "max_context_length" in hparam else None,
        learning_rate=hparam.learning_rate,
        lr_scheduler=hparam.lr_scheduler,  # exponential, constant
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        accelerator=hparam.accelerator,  # ddp or deepspeed
        num_train_epochs=hparam.num_train_epochs,
        train_batch_size=hparam.train_batch_size,
        eval_batch_size=arg_.test_batch if arg_.test_batch else hparam.eval_batch_size,
        dump_batch_size=hparam.dump_batch_size if "dump_batch_size" in hparam else None,
        gradient_accumulation_steps=hparam.gradient_accumulation_steps,
        n_gpu=hparam.n_gpu,
        num_workers=hparam.num_workers,
        resume_from_checkpoint=hparam.resume_from_checkpoint,
        val_check_interval=1.0,
        early_stop_callback=False,
        seed=hparam.seed if "seed" in hparam else 42,
        check_val_every_n_epoch=hparam.check_val_every_n_epoch,
        train_file=hparam.train_file,
        dev_file=hparam.dev_file,
        test_file=arg_.test_file if arg_.test_file else hparam.test_file,
        dev_input2output=hparam.dev_input2output if "dev_input2output" in hparam else None,
        corpus_file=hparam.corpus_file if "corpus_file" in hparam else None,
        constrained_decoding=True,
        do_train=hparam.do_train,
        do_test=hparam.do_test,
        test_model_path=hparam.test_model_path,
        test_name=arg_.test_name if arg_.test_name else hparam.test_name,
        val_beam_size=arg_.test_beam_size if arg_.test_beam_size else hparam.val_beam_size,
        ret_num=arg_.test_ret_num if arg_.test_ret_num else hparam.val_beam_size,
        freeze_encoder=hparam.freeze_encoder,
        freeze_vocab_emb=hparam.freeze_vocab_emb,
        contextualized_file=hparam.contextualized_file,  # new - tokId_emb.pickle
        faiss_file = hparam.faiss_file,
        groupId2tokIdList=hparam.groupId2tokIdList,  # new - tokGroupId_tokIdList.pickle 
        tokId2groupId=hparam.tokId2groupId,  # new - tokId_tokGroupId.pickle 
        tokId2tokText=hparam.tokId2tokText,  # new - tokId_tokText.pickle 
        tokId2corpus=hparam.tokId2corpus,  # new - tokId_corpus.pickle 
        corpus2EmbMean=hparam.corpus2EmbMean if "corpus2EmbMean" in hparam else None,  # new - tokId_corpus.pickle 
        tree_type=hparam.tree_type,  # new - nodeId_tokIdList.pickle
        tree_path=hparam.tree_path, # new
        nodeId_sup=hparam.nodeId_sup if "nodeId_sup" in hparam else None, # new
        embedding_model=hparam.embedding_model, # new - model used to extract embedding
        max_beam_search=hparam.max_beam_search, # new - select a token which has maximum score in groupId
        model_type=hparam.model_type, # new - bi-encoder Training
        periflow=hparam.periflow, # new - periflow
        periflow_dir=hparam.periflow_dir if "periflow_dir" in hparam else None, # new - directory of periflow
        limit_val_batches=hparam.limit_val_batches if "limit_val_batches" in hparam else 1.0,
        train_c_emb=hparam.train_c_emb,
        bi_type=hparam.bi_type,
        bi_loss=hparam.bi_loss if "bi_loss" in hparam else None,
        gr_decoder_only=hparam.gr_decoder_only,
        gr_decoder_only_encoder_ckpt=hparam.gr_decoder_only_encoder_ckpt,
        reload_dataloader_every_n_epochs=hparam.reload_dataloader_every_n_epochs if "reload_dataloader_every_n_epochs" in hparam else False,
        do_save=hparam.do_save if "do_save" in hparam else None,
        tok_num=hparam.tok_num if "tok_num" in hparam else None,
        model_dim=hparam.model_dim if "model_dim" in hparam else None,
        dump_path=hparam.dump_path if "dump_path" in hparam else hparam.dataset, 
        save_model_only=arg_.save_model_only,
        clusterIdList2corpusList=hparam.clusterIdList2corpusList if "clusterIdList2corpusList" in hparam else None,
        cluster_num = hparam.cluster_num if "cluster_num" in hparam else None,
        dat_size = hparam.dat_size if "dat_size" in hparam else 37000000,
    ) 
    args = argparse.Namespace(**args_dict)
    if args.do_test: args.n_gpu = 1
    assert not (args.do_train and args.do_test), "Choose between train|test"
    torch.multiprocessing.set_start_method('spawn')

    if torch.cuda.current_device() == 0:
        print("#" * 80)
        print(args)
        print("#" * 80)

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        monitor="val_em",
        mode="max",
        dirpath=args.output_dir,
        filename="{epoch:02d}-{val_em:.2f}",
        save_top_k=15,
    )
    callbacks.append(checkpoint_callback)

    if args.lr_scheduler == "constant" and torch.cuda.current_device() == 0:
        print(f"@@@ Not Using Learning Rate Scheduler")
    elif hparam.wandb_log == False:
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
        limit_val_batches=args.limit_val_batches,
        reload_dataloaders_every_n_epochs=args.reload_dataloader_every_n_epochs
    )
    main(args, train_params)
