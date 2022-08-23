import os
import re
import sys
import h5py
import json
import uuid
import copy
import torch
#import faiss
import string
import pickle
import numpy 
import numpy as np
import pandas as pd 
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributed as dist

from data import GENREDataset, JOINTDataset, MEANDataset
#from blob import get_blob_info, upload_directory_to_blob

from transformers import BartConfig, BartTokenizer, BartModel, BartForConditionalGeneration 
from transformers import BertTokenizer, Adafactor, AutoTokenizer
from torch.utils.data import DataLoader
from itertools import chain
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture 

from azure.storage.blob import (
    BlobServiceClient,
    BlobClient,
    ContainerClient,
    __version__,
)

class BartBaseClass(pl.LightningModule):
    def __init__(self):
        super(BartBaseClass, self).__init__()
        
        if torch.cuda.current_device() == 0:
            self.print = True 
        else:
            self.print = False 


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

    def _gather_object(self, obj):
        gathered = [None for _ in range(self.hparams.n_gpu)]
        dist.all_gather_object(gathered, obj)
        return gathered

    def gather_list(self, obj):
        gathered = self._gather_object(obj)
        output = []
        output = list(chain(*gathered))
        return output


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

    def _calculate_recall(self, pred, gt):
        assert len(pred) == self.hparams.val_beam_size, f'gt: {gt}\npred: {pred}' 
        _correct = 0
        for elem in pred:
            if elem is None: continue 
            if self.normalize_answer(elem) == self.normalize_answer(gt):
                return 100
        return 0

    def _calculate_em(self, pred, gt):
        if pred is None:
            return 0
        if self.normalize_answer(pred) == self.normalize_answer(gt):
            return 100
        else:
            return 0

    def lmap(self, f, x):
        return list(map(f, x))


    def _remove_zero(self, ids):
        if ids.count(0) == 1: 
            return ids
        for i, id in enumerate(ids):
            if i != 0 and id == 0:   
                return ids[:i] 
        assert False

    def _remove_prev_from_trie(self, trie, ids_list):
        temp_list = []
        for elem in ids_list:
            if elem in temp_list:
                continue
            else:
                temp_list.append(elem)
        ids_list = temp_list
        for i in range(len(ids_list)):
            ids_list[i][0] = 0
            ids_list[i] = self._remove_zero(ids_list[i])
        if self.hparams.tree_type == "groupId":
            for ids in ids_list:
                cur_dict = trie
                for i in range(len(ids)-1):
                    cur_dict = cur_dict[ids[i]]
                cur_dict.pop(ids[-1])
                """
                if len(cur_dict.keys()) == 1:
                    assert int(list(cur_dict.keys())[0]) == -2
                    cur_dict.pop(-2)
                """
                for i in range(len(ids)-1):
                    idx = -2-i 
                    _ids = int(ids[idx])
                    cur_dict = trie 
                    for j in range(len(ids)+idx):
                        cur_dict = cur_dict[int(ids[j])] 
                    if len(cur_dict[_ids]) == 0:
                        cur_dict.pop(_ids)
                        """
                        if len(cur_dict.keys()) == 1:
                            assert int(list(cur_dict.keys())[0]) == -2
                            cur_dict.pop(-2)
                        """
        elif self.hparams.tree_type == "nodeId":
            for ids in ids_list:
                c_nodeid = ids[-1]
                n_nodeid = list(trie[c_nodeid])[0]
                if len(trie[n_nodeid]) == 0:
                    trie[c_nodeid].remove(n_nodeid)
                for r_id in range(len(ids)-1):
                    c_nodeid = ids[-1-r_id]
                    p_nodeid = ids[-2-r_id]
                    if len(trie[c_nodeid]) == 0:
                        trie[p_nodeid].remove(c_nodeid)
        else:
            assert False

        return trie 

class BartBiEncoder(BartBaseClass):
    def __init__(self, args):
        super(BartBiEncoder, self).__init__()
        self.save_hyperparameters(args)

        if self.hparams.do_train:
            if self.hparams.bi_type in ['encoder_decoder']:
                self.model = BartModel.from_pretrained(self.hparams.model_name_or_path)
            else:
                assert False
            self.tokenizer = BartTokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)
            if self.print:
                print(f'@@@ In Training Mode...')
                print(f'@@@ Loading Model from {self.hparams.model_name_or_path}')
            self.em_score_list = []
            self.recall_score_list = []

            if self.hparams.periflow:
                self.connect_str, self.container_name = get_blob_info()
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)

        if self.hparams.do_test:
            if self.hparams.bi_type in ['encoder_decoder']:
                self.model = BartModel.from_pretrained(self.hparams.test_model_path)
            else:
                assert False
            self.tokenizer = BartTokenizer.from_pretrained(self.hparams.test_model_path)
            if self.print:
                print(f'@@@ In Test Mode ...')
                print(f'@@@ Loading Model from {self.hparams.test_model_path}')
            self.test_input_list = []
            self.test_gt_list = []
            self.test_gt_tok_list = []
            self.test_pred_list = []
            self.test_pred_tok_list = []
            self.test_em_score_list = []
            self.test_recall_score_list = []

        self.contextualized_tokid2emb = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), "rb"))
        self.contextualized_tensor = torch.tensor(list(self.contextualized_tokid2emb.values())).to(self.device)
        if self.hparams.fp16:
            self.contextualized_tensor = self.contextualized_tensor.half()
        self.contextualized_token = list(self.contextualized_tokid2emb.keys())
        self.tokId2corpus = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2corpus), "rb"))
        self.corpus2tokId = self._get_corpus2tokId()

        self.loss_fct = nn.CrossEntropyLoss()
        self.decoder_input_ids, self.decoder_attention_mask = self.get_decoder_input()

    def _get_corpus2tokId(self):
        corpus2tokId = defaultdict(list)
        for _tokId, _corpus in self.tokId2corpus.items():
            corpus2tokId[_corpus].append(_tokId)
        return corpus2tokId

    def _get_dataset(self, split):
        dataset = GENREDataset(
            tokenizer=self.tokenizer,
            split=split,
            hparams=self.hparams,
            tokid2emb=self.contextualized_tokid2emb 
        )
        return dataset

    def get_decoder_input(self):
        _tok = self.tokenizer("</s>", return_tensors='pt', add_special_tokens=False)
        _input_ids = _tok['input_ids'].to(self.device)
        _attention_mask = _tok['attention_mask'].to(self.device)
        return _input_ids, _attention_mask

    def forward(self, input_ids, attention_mask=None):
        bs = input_ids.shape[0]
        if self.hparams.bi_type in ['encoder_decoder']:
            d_input = torch.cat([self.decoder_input_ids]*bs, 0).to(self.device)
            d_att = torch.cat([self.decoder_attention_mask]*bs, 0).to(self.device)
            
            outputs = self.model(input_ids, 
                            attention_mask=attention_mask,
                            decoder_input_ids=d_input, 
                            decoder_attention_mask=d_att
                            )
        else:
            outputs = self.model(input_ids,
                            attention_mask=attention_mask,
                            )
        return outputs 

    def _get_embedding(self, batch):
        query_input_ids = batch['source_ids']
        query_attention_mask = batch['source_mask']
        key_input_ids = batch['target_ids'].detach().cpu().numpy()

        query_outputs = self(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
        )
        if self.hparams.bi_type in ['encoder_decoder', 'encoder_first']:
            query_output = query_outputs.last_hidden_state[:, 0]
        elif self.hparams.bi_type in ['encoder_mean']:
            query_output = torch.mean(query_outputs.last_hidden_state, dim=1) 
        else:
            assert False

        key_outputs = torch.cat([torch.tensor(self.contextualized_tokid2emb[key_input_id[0]]).unsqueeze(0) for key_input_id in key_input_ids], 0)
        if self.hparams.fp16:
            key_outputs = key_outputs.half()
        return query_output, key_outputs

    def _calculate_similarity(self, batch, total=False):
        z1, z2 = self._get_embedding(batch) # z1: query, z2: corpus 개수
        z1 = z1.to(self.device)
        z2 = z2.to(self.device)
        if total:
            sim = torch.inner(z1, self.contextualized_tensor.to(self.device)) # sim: [query 개수, corpus 개수]
        else:
            sim = torch.inner(z1, z2) 
        return sim

    def _common_step(self, batch):
        _sim = self._calculate_similarity(batch)
        labels = torch.arange(_sim.size(0)).long().to(self.device)
        loss = self.loss_fct(_sim, labels)
        return loss


    def _total_common_step(self, batch, all=False):
        _sim = self._calculate_similarity(batch, total=True)
        labels = torch.zeros_like(_sim)
        for i, (ids, corpus) in enumerate(zip(batch['target_ids'], batch['output'])):
            all_ids = self.corpus2tokId[corpus]
            assert ids[0] in all_ids, f'ids: {ids} // all_ids: {all_ids}'
            for ids in all_ids:
                labels[i][ids] = 1
            assert torch.count_nonzero(labels[i]).item() == len(all_ids)
        labels = torch.tensor(labels).float().to(self.device)
        loss = self.loss_fct(_sim, labels)
        return loss

    def training_step(self, batch, batch_idx):
        if self.hparams.bi_loss == "base":
            loss = self._common_step(batch)
        elif self.hparams.bi_loss == "total":
            loss = self._total_common_step(batch, all=False)
        elif self.hparams.bi_loss == "total-all":
            loss = self._total_common_step(batch, all=True)
        else:
            assert False, f"Check bi_loss type: {self.hparams.bi_loss}"
        
        self.log(
            "train_loss",
            loss, 
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return loss 

    def _calculate_score(self, indices, batch, batch_idx):
        em_list = []; recall_list = []; pred_list = []; pred_tok_list = []
        for idx, (_indice_list, _output, _input) in enumerate(zip(indices, batch['output'], batch['input'])):
            _predict = []; _pred_idx = []
            for _indice in _indice_list:
                tokId = self.contextualized_token[int(_indice)]
                _pred_idx.append(tokId)
                if tokId==0 or tokId==1 or tokId==2 or tokId==3:
                    _predict.append(None)
                else:
                    _predict.append(self.tokId2corpus[tokId])
            em_list.append(self._calculate_em(_predict[0], _output))
            recall_list.append(self._calculate_recall(_predict, _output))
            pred_list.append(_predict)
            pred_tok_list.append(_pred_idx)
            if self.print and idx == 0 and batch_idx%100 == 0:
                print(f"$" * 50)
                print(f"query: {_input}\ngt: {_output}\npredict: {_predict}")
                print(f"em: {em_list[-1]} // recall: {recall_list[-1]}")
                print(f"$" * 50)
                print(" ")
        return em_list, recall_list, pred_list, pred_tok_list

    def validation_step(self, batch, batch_idx):
        query_output, _ = self._get_embedding(batch)
        scores = torch.inner(query_output.to(self.device), self.contextualized_tensor.to(self.device)) # [# of query, # of corpus] 
        top_scores = torch.topk(scores, self.hparams.val_beam_size)
        indices = top_scores.indices # [# of query, self.hparams.val_beam_size]
        assert len(indices) == len(batch['output'])

        em_score, recall_score, _, _ = self._calculate_score(indices, batch, batch_idx)
        self.em_score_list.extend(list(em_score))
        self.recall_score_list.extend(list(recall_score))

    def validation_epoch_end(self, outputs):
        avg_em = np.mean(np.array(self.em_score_list))
        avg_recall = np.mean(np.array(self.recall_score_list))
        self.em_score_list = []
        self.recall_score_list = []
        self.log(
            "val_em",
            avg_em,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_recall",
            avg_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return

    def test_step(self, batch, batch_idx):
        query_output, _ = self._get_embedding(batch)
        scores = torch.inner(query_output.to(self.device), self.contextualized_tensor.to(self.device)) # [# of query, # of corpus] 
        top_scores = torch.topk(scores, self.hparams.val_beam_size)
        indices = top_scores.indices # [# of query, self.hparams.val_beam_size]
        assert len(indices) == len(batch['output'])

        em_score, recall_score, pred_list, pred_tok_list = self._calculate_score(indices, batch, batch_idx)
        self.test_input_list.extend(list(batch["input"]))
        self.test_gt_list.extend(list(batch["output"]))
        self.test_gt_tok_list.extend(list(batch["target_ids"].detach().cpu().numpy().tolist()))
        self.test_pred_list.extend(list(pred_list))
        self.test_pred_tok_list.extend(list(pred_tok_list))
        self.test_em_score_list.extend(list(em_score))
        self.test_recall_score_list.extend(list(recall_score))

    def _remove_dup(self, _dict):
        unique_num = len(set(_dict['input']))
        ret_dict = {'input': [], 'gt': [], 'gt_tok': [], 'pred': [], 'pred_tok': [], 'em': [], 'recall': []}
        for _input, _gt, _gt_tok, _pred, _pred_tok, _em, _recall in zip(_dict['input'], _dict['gt'], _dict['gt_tok'], _dict['pred'], _dict['pred_tok'], _dict['em'], _dict['recall']):
            if _input in ret_dict['input']: continue
            else:
                ret_dict['input'].append(_input)
                ret_dict['gt'].append(_gt)
                ret_dict['gt_tok'].append(_gt_tok)
                ret_dict['pred'].append(_pred)
                ret_dict['pred_tok'].append(_pred_tok)
                ret_dict['em'].append(_em)
                ret_dict['recall'].append(_recall)
        assert len(ret_dict['input']) == unique_num
        return ret_dict

    def test_epoch_end(self, outputs):
        os.makedirs(self.hparams.output_dir, exist_ok=True)
        _input = self.gather_list(self.test_input_list)
        _gt = self.gather_list(self.test_gt_list)
        _gt_tok = self.gather_list(self.test_gt_tok_list)
        _pred = self.gather_list(self.test_pred_list)
        _pred_tok = self.gather_list(self.test_pred_tok_list)
        _em = self.gather_list(self.test_em_score_list)
        _recall = self.gather_list(self.test_recall_score_list)
        save_dict = {'input': _input, 'gt': _gt, 'gt_tok': _gt_tok, 'pred': _pred, 'pred_tok': _pred_tok, 'em': _em, 'recall': _recall}
        save_dict = self._remove_dup(save_dict)
        assert len(_input) == len(_gt) == len(_pred) == len(_em) == len(_recall) == len(_gt_tok) == len(_pred_tok)
        if self.print:
            filename = f"{self.hparams.test_name}_result_beam{self.hparams.val_beam_size}.json"
            with open(
                os.path.join(
                    self.hparams.output_dir,
                    filename 
                ),
                "w",
            ) as f:
                json.dump(
                    save_dict,
                    f,
                )
            print(
                f"Saving in {os.path.join(self.hparams.output_dir, filename)}!\nnumber of elements: {len(_input)}"
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

        target_path = save_path
        if self.hparams.periflow:
            success = False
            i = 1
            while not success:
               try:
                  upload_directory_to_blob(save_path, target=target_path, container_name=self.container_name)
                  success = True
               except:
                  print(f'Failed on Uploading {target_path}')
                  _name = "best_tfmr_"*i+f"{self.current_epoch}"
                  target_path = os.path.join(self.hparams.output_dir, _name)
                  i += 1
