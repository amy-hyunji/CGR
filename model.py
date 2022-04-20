import os
import re
import sys
import json
import copy
import torch
import string
import pickle
import numpy as np
import pytorch_lightning as pl
import torch.distributed as dist

from data import GENREDataset

from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration, Adafactor
from torch.utils.data import DataLoader
from itertools import chain


class T5FineTuner(pl.LightningModule):
    def __init__(self, args):
        super(T5FineTuner, self).__init__()
        self.save_hyperparameters(args)

        if torch.cuda.current_device() == 0:
            self.print = True
        else:
            self.print = False

        # If in training mode, load ckpt for training
        if self.hparams.do_train:
            config = T5Config.from_pretrained(self.hparams.model_name_or_path)
            config.update(
                {"contextualized_emb_num": self.hparams.contextualized_emb_num}
            )
            config.update(
                {"contextualized_file": self.hparams.contextualized_file}
            )  # tokId_emb.pickle
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.hparams.model_name_or_path, config=config
            )
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.hparams.model_name_or_path
            )
            if self.print:
                print(f"@@@ Loading Model from {self.hparams.model_name_or_path}")

            self.em_score_list = []
            self.recall_score_list = []

        # If in testing mode, load ckpt for inference
        if self.hparams.do_test:
            raise NotImplementedError("Test Code is not implemented yet!")
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.hparams.test_model_path
            )
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.test_model_path)
            if self.print:
                print(f"@@@ Loading Model from {self.hparams.test_model_path}")
            self.test_input_list = []
            self.test_gt_list = []
            self.test_pred_list = []
            self.test_em_score_list = []
            self.test_recall_score_list = []

        self.trie_dict = pickle.load(open(self.hparams.prefix_tree_file, "rb"))
        self.contextualized_tokid2emb = pickle.load(
            open(self.hparams.contextualized_file, "rb")
        )
        assert (
            len(self.contextualized_tokid2emb.keys())
            == self.hparams.contextualized_emb_num
        )

    def _get_dataset(self, split):
        dataset = GENREDataset(
            tokenizer=self.tokenizer,
            split=split,
            hparams=self.hparams,
            tokid2emb=self.contextualized_tokid2emb,
        )
        return dataset

    def train_dataloader(self):
        train_dataset = self._get_dataset(split="train")
        dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.hparams.train_batch_size,
            drop_last=True,
            num_workers=self.hparams.num_workers,
        )
        return dataloader

    def val_dataloader(self):
        val_dataset = self._get_dataset(split="validation")
        dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=self.hparams.eval_batch_size,
            drop_last=True,
            num_workers=self.hparams.num_workers,
        )
        return dataloader

    def test_dataloader(self):
        test_dataset = self._get_dataset(split="test")
        dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=self.hparams.eval_batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
        )
        return dataloader

    def lmap(self, f, x):
        return list(map(f, x))

    def ids_to_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def forward(
        self,
        input_ids,
        attention_mask,
        lm_labels,
        decoder_attention_mask,
        decoder_inputs_embs,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=None,
            decoder_inputs_embs=decoder_inputs_embs,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _loss(self, batch):
        lm_labels = copy.deepcopy(batch["target_ids"])
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
            decoder_inputs_embs=batch["target_embs"],
        )
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log(
            "train loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return loss

    def get(self, batch_id, input_ids):
        return self._get_from_trie(input_ids, self.trie_dict)

    def _get_from_trie(self, input_ids, trie_dict):
        if len(input_ids) == 0:
            return list(trie_dict.keys())
        elif input_ids[0] in trie_dict:
            return self._get_from_trie(input_ids[1:], trie_dict[input_ids[0]])
        else:
            return []

    def _calculate_recall(self, pred, gt):
        assert len(pred) == self.hparams.val_beam_size
        _correct = 0
        for elem in pred:
            if self.normalize_answer(elem) == self.normalize_answer(gt):
                return 1
        return 0

    def _calculate_em(self, pred, gt):
        if self.normalize_answer(pred) == self.normalize_answer(gt):
            return 1
        else:
            return 0

    def calculate_scores(self, preds, gt_text, query):
        em_list = []
        recall_list = []
        for idx, (_query, _pred, _gt) in enumerate(zip(query, preds, gt_text)):
            _em = self._calculate_em(_pred[0], _gt)
            _recall = self._calculate_recall(_pred, _gt)
            if self.print and idx == 0:
                print(f"$" * 50)
                print(f"query: {_query}\npreds: {_pred}\ngt: {_gt}")
                print(f"em: {_em} // recall: {_recall}")
                print(f"$" * 50)
                print(" ")
            em_list.append(_em)
            recall_list.append(_recall)
        return em_list, recall_list

    def _val_step(self, batch, return_elem=False):
        # calculates recall and em -> returns the list of each score
        _generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            decoder_attention_mask=batch["target_mask"],
            max_length=self.hparams.max_output_length,
            num_beams=self.hparams.val_beam_size,
            num_return_sequences=self.hparams.val_beam_size,
            prefix_allowed_tokens_fn=lambda batch_id, sent: self.get(
                batch_id, sent.tolist()
            ),
            early_stopping=True,
        )
        _generated_text = self.ids_to_text(_generated_ids)

        inum = len(_generated_ids) // self.hparams.val_beam_size
        assert inum == len(batch["output"])
        generated_text = [
            _generated_text[
                i * self.hparams.val_beam_size : (i + 1) * self.hparams.val_beam_size
            ]
            for i in range(inum)
        ]

        em_list, recall_list = self.calculate_scores(
            generated_text, batch["output"], batch["input"]
        )

        if return_elem:
            assert (
                len(list(batch["input"]))
                == len(list(generated_text))
                == len(list(em_list))
            )
            return {
                "input": list(batch["input"]),
                "gt": list(batch["output"]),
                "pred": list(generated_text),
                "em": list(em_list),
                "recall": list(recall_list),
            }
        else:
            return em_list, recall_list

    def validation_step(self, batch, batch_idx):
        em_score, recall_score = self._val_step(batch)
        self.em_score_list.extend(list(em_score))
        self.recall_score_list.extend(list(recall_score))

    def validation_epoch_end(self, outputs):
        avg_em = np.mean(np.array(self.em_score_list))
        avg_recall = np.mean(np.array(self.recall_score_list))
        self.em_score_list = []
        self.recall_score_list = []
        self.log(
            "val em",
            avg_em,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val recall",
            avg_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return

    def test_step(self, batch, batch_idx):
        ret_dict = self._val_step(batch, return_elem=True)
        self.test_input_list.extend(ret_dict["input"])
        self.test_gt_list.extend(ret_dict["gt"])
        self.test_pred_list.extend(ret_dict["pred"])
        self.test_em_score_list.extend(ret_dict["em"])
        self.test_recall_score_list.extend(ret_dict["recall"])

    def _gather_object(self, obj):
        if self.print:
            print(f"## Gathering list from {self.hparams.n_gpu} process!")
        gathered = [None for _ in range(self.hparams.n_gpu)]
        dist.all_gather_object(gathered, obj)
        return gathered

    def gather_list(self, obj):
        gathered = self._gather_object(obj)
        output = []
        output = list(chain(*gathered))
        return output

    def test_epoch_end(self, outputs):
        os.makedirs(self.hparams.output_dir, exist_ok=True)
        _input = self.gather_list(self.test_input_list)
        _gt = self.gather_list(self.test_gt_list)
        _pred = self.gather_list(self.test_pred_list)
        _em = self.gather_list(self.test_em_score_list)
        _recall = self.gather_list(self.test_recall_score_list)
        assert len(_input) == len(_gt) == len(_pred) == len(_em) == len(_recall)
        if self.print:
            with open(
                os.path.join(
                    self.hparams.output_dir,
                    f"result_beam{self.hparams.val_beam_size}.json",
                ),
                "w",
            ) as f:
                json.dump(
                    {
                        "input": _input,
                        "gt": _gt,
                        "pred": _pred,
                        "em": _em,
                        "recall": _recall,
                    },
                    f,
                )
            print(
                f"Saving in {os.path.join(self.hparams.output_dir)}!\nnumber of elements: {len(_input)}"
            )
            print(f"EM: {np.array(_em).mean()}")
            print(f"Recall: {np.array(_recall).mean()}")

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            warmup_init=False,
            scale_parameter=False,
            relative_step=False,
        )
        self.opt = optimizer

        if self.hparams.lr_scheduler == "constant":
            return [optimizer]
        elif self.hparams.lr_scheduler == "exponential":
            len_data = len(self.train_dataloader())
            denominator = self.hparams.n_gpu
            steps_per_epoch = (
                (len_data // denominator) + 1
            ) // self.hparams.gradient_accumulation_steps
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                epochs=self.hparams.num_train_epochs,
                anneal_strategy="linear",
                cycle_momentum=False,
            )
            return [optimizer], [
                {"scheduler": scheduler, "interval": "step", "name": "learning_rate"}
            ]
        else:
            raise NotImplementedError("Choose lr_schduler from (constant|exponential)")

    def on_save_checkpoint(self, checkpoint):
        save_path = os.path.join(
            self.hparams.output_dir, f"best_tfmr_{self.current_epoch}"
        )
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
