import os
import re
import sys
#import h5py
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

from data import ENTAILDataset, GENREDataset, JOINTDataset, MEANDataset, TESTDataset
#from blob import get_blob_info, upload_directory_to_blob

from grounded_T5 import T5ForConditionalGeneration as grounded_T5
from joint_T5 import T5Model as joint_T5
from split_T5 import T5ForConditionalGeneration as split_T5 
from contextualized_T5 import T5ForConditionalGeneration as contextualized_T5
from contextualized_T5_fix import T5ForConditionalGeneration as test_T5
from entail_T5 import T5ForConditionalGeneration as entail_T5
from entail_T5_w_enc import T5ForConditionalGeneration as entail_T5_w_enc
from entail_T5_split import T5ForConditionalGeneration as entail_T5_split

from transformers import T5Config, T5Tokenizer, T5Model, T5EncoderModel, T5ForConditionalGeneration 
from transformers import BertTokenizer, Adafactor, AutoTokenizer
from torch.utils.data import DataLoader
from itertools import chain
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture 
from collections import Counter
from all_gather import my_all_gather, my_all_gather2
from utils import *

# from azure.storage.blob import (
#     BlobServiceClient,
#     BlobClient,
#     ContainerClient,
#     __version__,
# )

class FaissKMeans:
    def __init__(self, n_clusters=10, n_init=3, max_iter=50):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X, y):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]
        return self.cluster_centers_

    def predict(self, X):
        ret = self.kmeans.index.search(X.astype(np.float32), 1)[1]
        return [elem[0] for elem in ret]

class T5BaseClass(pl.LightningModule):
    def __init__(self):
        super(T5BaseClass, self).__init__()
        
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

    def _get_dtype(self):
        if self.hparams.fp16:
            return "float16"
        else:
            return "float32"

    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            text = text.replace("<title>","<")
            text = text.replace("<context>","<")
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _calculate_recall(self, pred, gt):
        assert len(pred) == self.hparams.ret_num, f'gt: {gt}\npred: {pred}' 
        if type(gt) == list:
            gt = [self.normalize_answer(el) for el in gt]
            for elem in pred:
                if elem is None: continue 
                if self.normalize_answer(elem) in gt:
                    return 100
        else:
            for elem in pred:
                if elem is None: continue 
                if self.normalize_answer(elem) == self.normalize_answer(gt):
                    return 100
        return 0

    def _split_by_sp(self, sen_list, sp):
        if type(sen_list) != list:
            sen_list = [sen_list]
        ret_list = []
        for sen in sen_list:
            ret_list.extend(sen.split(sp))
        return ret_list

    def _multi_split_by_sp(self, sen_list):
        sen_list = self._split_by_sp(sen_list, '[BECAUSE]')
        sen_list = self._split_by_sp(sen_list, '[AND]')
        sen_list = self._split_by_sp(sen_list, '[INFER]')
        return sen_list

    def _split_dec(self, sen_list):
        es_list = []; vs_list = []
        for elem in sen_list:
            if self.normalize_answer(elem) == "":
               continue 
            elif '[Es]' in elem:
               es_list.append(self.normalize_answer(elem))
            elif '[Vs]' in elem:
               vs_list.append(self.normalize_answer(elem))
            else:
               assert False, f"{elem} does not contain neither [Es] and [Vs]!"
        return set(es_list), set(vs_list)

    def _calculate_iter_entail_f1(self, pred, gt):
        pred_es = []; pred_vs = []; gt_es = []; gt_vs = []

        for _pred, _gt in zip(pred, gt):
           pred_sen = self._multi_split_by_sp(_pred)
           gt_sen = self._multi_split_by_sp(_gt)

           _pred_es, _pred_vs = self._split_dec(pred_sen)
           _gt_es, _gt_vs = self._split_dec(gt_sen)
         
           pred_es.extend(_pred_es); pred_vs.extend(_pred_vs) 
           gt_es.extend(_gt_es); gt_vs.extend(_gt_vs) 

        common = Counter(pred_es) & Counter(gt_es)
        #num_same = sum(common.values())
        num_same = len(common.keys())

        if num_same == 0:
            es_f1 = 0
        else:
            es_prec = 1.0 * num_same / len(pred_es)
            es_rec = 1.0 * num_same / len(gt_es)
            es_f1 = (2*es_prec*es_rec)/(es_prec+es_rec)

        common = Counter(pred_vs) & Counter(gt_vs)
        #num_same = sum(common.values())
        num_same = len(common.keys())

        if num_same == 0:
            vs_f1 = 0
        else:
            vs_prec = 1.0 * num_same / len(pred_vs)
            vs_rec = 1.0 * num_same / len(gt_vs)
            vs_f1 = (2*vs_prec*vs_rec) / (vs_prec+vs_rec)
         
        return es_f1*100, vs_f1*100
         

    def _calculate_entail_f1(self, pred, gt):
        pred_sen = self._multi_split_by_sp(pred)
        gt_sen = self._multi_split_by_sp(gt)

        pred_es, pred_vs = self._split_dec(pred_sen)
        gt_es, gt_vs = self._split_dec(gt_sen)

        common = Counter(pred_es) & Counter(gt_es)
        #num_same = sum(common.values())
        num_same = len(common.keys())

        if num_same == 0:
            es_f1 = 0
        else:
            es_prec = 1.0 * num_same / len(pred_es)
            es_rec = 1.0 * num_same / len(gt_es)
            es_f1 = (2*es_prec*es_rec)/(es_prec+es_rec)

        common = Counter(pred_vs) & Counter(gt_vs)
        #num_same = sum(common.values())
        num_same = len(common.keys())

        if num_same == 0:
            vs_f1 = 0
        else:
            vs_prec = 1.0 * num_same / len(pred_vs)
            vs_rec = 1.0 * num_same / len(gt_vs)
            vs_f1 = (2*vs_prec*vs_rec) / (vs_prec+vs_rec)
         
        return es_f1*100, vs_f1*100
         

    def _calculate_em(self, pred, gt):
        if pred is None: 
            return 0
        if type(gt) == list and type(pred) != list:
            gt = [self.normalize_answer(el) for el in gt]
            if self.normalize_answer(pred) in gt:
                return 100
            else:
                return 0
        elif type(gt) == list and type(pred) == list:
            gt = [self.normalize_answer(el) for el in gt]
            pred = [self.normalize_answer(el) for el in pred]
            if set(gt) == set(pred):
                return 100  
            else:
                return 0
        else:
            if self.normalize_answer(pred) == self.normalize_answer(gt):
                return 100
            else:
                return 0


    def _calculate_f1(self, pred_list, gt_list):
       assert type(pred_list) == list, type(gt_list) == list 
       pred_list = list(set([self.normalize_answer(elem) for elem in pred_list]))
       gt_list = list(set([self.normalize_answer(elem) for elem in gt_list]))
       common = Counter(pred_list) & Counter(gt_list)
       # num_same = sum(common.values())
       num_same = len(common.keys())

       if num_same == 0: return 0

       precision = 1.0 * num_same / len(pred_list)
       recall = 1.0 * num_same / len(gt_list)
       f1 = (2*precision*recall) / (precision+recall)*100
       return f1

    def _calculate_tok_f1(self, pred, gt):
        assert isinstance(pred, str) and isinstance(gt, str)
        pred_toks = self.normalize_answer(pred).split()
        gt_toks = self.normalize_answer(gt).split()
        common = Counter(pred_toks) & Counter(gt_toks)
        # num_same = sum(common.values())
        num_same = len(common.keys())

        if num_same == 0: return 0

        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gt_toks)
        f1 = (2*precision*recall) / (precision+recall)*100

        return f1


    def _calculate_iter_tok_f1(self, pred, gt):
        assert isinstance(pred, list) and isinstance(gt, list)
        f1_score = []

        for _pred, _gt in zip(pred, gt):
           pred_toks = self.normalize_answer(_pred).split()
           gt_toks = self.normalize_answer(_gt).split()
           common = Counter(pred_toks) & Counter(gt_toks)
           #num_same = sum(common.values())
           num_same = len(common.keys())

           if num_same == 0: return 0

           precision = 1.0 * num_same / len(pred_toks)
           recall = 1.0 * num_same / len(gt_toks)
           f1 = (2*precision*recall) / (precision+recall)*100
          
           f1_score.append(f1)

        return np.array(f1_score).mean()

    def lmap(self, f, x):
        return list(map(f, x))

    def _remove_zero(self, ids):
        if ids.count(0) == 1: 
            return ids
        for i, id in enumerate(ids):
            if i != 0 and id == 0:   
                return ids[:i] 
        assert False

    def _remove_prev_from_trie(self, trie, ids_list, pad_ids=0):
        ids_list = [list(elem) for elem in ids_list]
        print(f'Remove.. {ids_list}')
        temp_list = []
        for elem in ids_list:
            if elem in temp_list:
                continue
            else:
                temp_list.append(elem)
        ids_list = temp_list
        for i in range(len(ids_list)):
            ids_list[i][0] = pad_ids
            if pad_ids == 0: 
                ids_list[i] = self._remove_zero(ids_list[i])
        if self.hparams.tree_type in ["groupId", "clusterId"]:
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

class T5grTuner(T5BaseClass): 
    def __init__(self, args):
        super(T5grTuner, self).__init__()
        self.save_hyperparameters(args)
        self.trie = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tree_path), "rb"))
        self.contextualized_tokid2emb = pickle.load(
            open(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), "rb")
        )
        self.contextualized_emb_num = len(self.contextualized_tokid2emb.keys())
        print(f"$$$$$$ # of embedding: {self.contextualized_emb_num}")
        self.config = T5Config.from_pretrained(self.hparams.model_name_or_path)
        self.config.update({"fp16": self.hparams.fp16})
        self.config.update({"train_c_emb": self.hparams.train_c_emb}) 
        self.config.update({"do_test": self.hparams.do_test}) 
        self.config.update({"tie_enc_dec_vocab": self.hparams.tie_vocab_emb}) 
        self.config.update(
            {"contextualized_emb_num": self.contextualized_emb_num}
        )
        self.config.update(
            {"contextualized_file": os.path.join(self.hparams.dataset, self.hparams.contextualized_file)}
        )  # tokId_emb.pickle
        self.config.update({"freeze_vocab_emb": self.hparams.freeze_vocab_emb})
        self.config.update({"change_lm_head": self.hparams.change_lm_head})
        self.config.update({"change_dec": self.hparams.change_dec})
        self.config.update({"change_enc": self.hparams.change_enc})
        self.config.update({"original_T5": self.hparams.original_T5})
        self.groupId2tokId= pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.groupId2tokIdList), "rb"))
        self.tokId2groupId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2groupId), 'rb'))
        self.tokId2tokText = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2tokText), 'rb'))
        self.save_epoch = []

    def ids_to_text(self, _generated_ids, skip_special_tokens=True):
        generated_ids = []
        for _ids in _generated_ids:
            _ids = copy.deepcopy(_ids)
            _ids = _ids.detach().cpu().numpy()
            _text = [self.tokId2tokText[_id] for _id in _ids]
            generated_ids.append(self.dec_tok.convert_tokens_to_ids(_text))
        gen_text = self.dec_tok.batch_decode(
            generated_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True
        )
        if not skip_special_tokens: 
            while "<pad>" in gen_text:
                gen_text = gen_text.replace('<pad>', '')
        text = self.lmap(str.strip, gen_text)
        return text

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_attention_mask=None,
        decoder_input_ids=None,
        lm_labels=None,
        return_dict=True
    ):
        if lm_labels is None:
            assert decoder_input_ids is not None
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                return_dict=return_dict
            )
        if decoder_input_ids is None:
            assert lm_labels is not None
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=lm_labels,
                return_dict=return_dict
            ) 
 
    def validation_step(self, batch, batch_idx):
        if self.hparams.save_model_only:
           self.convert_checkpoint("new_ckpt")
           print(f"Done saving checkpoint.. {self.current_epoch}")

        em_score, recall_score = self._val_step(batch, batch_idx)
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
        ret_dict = self._test_step(batch, batch_idx, return_elem=True)
        if ret_dict is None: return None
        
        if "gt" not in ret_dict:
            self.test_input_list.extend(ret_dict["input"])
            self.test_pred_list.extend(ret_dict["pred"])
            _input = self.gather_list(self.test_input_list)
            _pred = self.gather_list(self.test_pred_list)
            with open(self.test_save_name, "w") as f:
               json.dump({
                        "input": _input,
                        "pred": _pred
                     }, f)    
            print(f"Saving in {self.test_save_name}!")
            return


        if self.hparams.model_type == "total":
            self.test_input_list.extend(ret_dict["input"])
            self.test_gt_list.extend(ret_dict["gt"])
            self.test_gt_tok_list.extend(ret_dict["gt_tok"])
            self.test_pred_tok_list.extend(ret_dict["pred_tok"])
        else:
            self.test_input_list.extend(ret_dict["input"])
            self.test_gt_list.extend(ret_dict["gt"])
            self.test_pred_list.extend(ret_dict["pred"])
            if self.hparams.do_title:
               self.em_score_list.extend(ret_dict["em"]) 
               self.f1_score_list.extend(ret_dict["f1"]) 
               self.title_em_score_list.extend(ret_dict["title_em"]) 
               self.title_f1_score_list.extend(ret_dict["title_f1"]) 
            else:
               self.test_gt_tok_list.extend(ret_dict["gt_tok"])
               self.test_pred_tok_list.extend(ret_dict["pred_tok"])
               self.test_em_score_list.extend(ret_dict["em"])
               self.test_recall_score_list.extend(ret_dict["recall"])
        self._save_test() 

    def _save_test(self, epoch_end=False):
        os.makedirs(self.hparams.output_dir, exist_ok=True)
        if self.hparams.model_type == "total":
            _input = self.gather_list(self.test_input_list)
            _gt = self.gather_list(self.test_gt_list)
            _gt_tok = self.gather_list(self.test_gt_tok_list)
            _pred_tok = self.gather_list(self.test_pred_tok_list)
            assert len(_input) == len(_gt) == len(_pred_tok)
            if self.print:
                with open(self.test_save_name, "w") as f:
                    json.dump(
                        {
                            "input": _input,
                            "gt": _gt,
                            "gt_tok": _gt_tok,
                            "pred_tok": _pred_tok,
                        },
                        f,
                    )
                if epoch_end:
                    print(
                        f"Saving in {self.test_save_name}!"
                    )
        elif self.hparams.model_type == "multihop":
            _input = self.gather_list(self.test_input_list)
            _pred = self.gather_list(self.test_pred_list)
            with open(self.test_save_name, "w") as f:
                 json.dump(
                     {
                         "input": _input,
                         "pred": _pred,
                     },
                     f,
                 )
        else:
            _input = self.gather_list(self.test_input_list) 
            _gt = self.gather_list(self.test_gt_list)
            _pred = self.gather_list(self.test_pred_list)
            if self.hparams.do_title:
               _em = self.gather_list(self.em_score_list)
               _title_em = self.gather_list(self.title_em_score_list)
               _f1 = self.gather_list(self.f1_score_list)
               _title_f1 = self.gather_list(self.title_f1_score_list)
               assert len(_input) == len(_gt) == len(_pred) == len(_em) == len(_title_em) == len(_f1) == len(_title_f1)
               if self.print:
                  with open(self.test_save_name, "w") as f:
                     json.dump(
                           {
                              "input": _input,
                              "gt": _gt,
                              "pred": _pred,
                              "em": _em,
                              "title_em": _title_em,
                              "f1": _f1,
                              "title_f1": _title_f1,
                           },
                           f,
                        )
                  if epoch_end:
                     print(f"Saving in {self.test_save_name}!\nNumber of elements: {len(_input)}")
                     print(f"EM: {np.array(_em).mean()}")
                     print(f"Title EM: {np.array(_title_em).mean()}")
                     print(f"F1: {np.array(_f1).mean()}")
                     print(f"Title F1: {np.array(_title_f1).mean()}")
            else:
               _gt_tok = self.gather_list(self.test_gt_tok_list)
               _pred_tok = self.gather_list(self.test_pred_tok_list)
               _em = self.gather_list(self.test_em_score_list)
               _recall = self.gather_list(self.test_recall_score_list)
               assert len(_input) == len(_gt) == len(_pred) == len(_em) == len(_recall) == len(_gt_tok) == len(_pred_tok)
               if self.print:
                   with open(self.test_save_name, "w") as f:
                       json.dump(
                           {
                               "input": _input,
                               "gt": _gt,
                               "gt_tok": _gt_tok,
                               "pred": _pred,
                               "pred_tok": _pred_tok,
                               "em": _em,
                               "recall": _recall,
                           },
                           f,
                       )
                   if epoch_end:
                       print(
                           f"Saving in {self.test_save_name}!\nnumber of elements: {len(_input)}"
                       )
                       print(f"EM: {np.array(_em).mean()}")
                       print(f"Recall: {np.array(_recall).mean()}")
    
    def test_epoch_end(self, outputs):
        self._save_test(epoch_end=True)

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
        try:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            self.save_epoch.append(self.current_epoch)
        except:
            print(f"Fail to save ... {self.current_epoch}")
        if len(self.save_epoch) > 5:
            rm_file = f"best_tfmr_{self.save_epoch[0]}"
            os.system(f"rm -rf {os.path.join(self.hparams.output_dir, rm_file)}")
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

    def convert_checkpoint(self, sub_path=None):
        if sub_path is None:
           save_path = os.path.join(
               self.hparams.output_dir, f"best_tfmr_{self.current_epoch}"
           )
        else:
           save_path = os.path.join(
               self.hparams.output_dir, sub_path     
           )
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.save_epoch.append(self.current_epoch)
        if len(self.save_epoch) > 5:
            rm_file = f"best_tfmr_{self.save_epoch[0]}"
            os.system(f"rm -rf {os.path.join(self.hparams.output_dir, rm_file)}")
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

class T5AsyncBaseTuner(T5BaseClass):
    def __init__(self, args):
        super(T5AsyncBaseTuner, self).__init__()
        self.save_hyperparameters(args)

        os.makedirs(self.hparams.output_dir, exist_ok=True)
        os.makedirs(self.hparams.dataset, exist_ok=True)
        self.config = T5Config.from_pretrained(self.hparams.model_name_or_path)
        self.load_epoch=-5
        if self.hparams.contextualized_file is None:
            if self.hparams.do_train:
                if self.hparams.resume_from_checkpoint is None:
                    if (self.hparams.cluster_num > 0 and f"k-means_corpus_tokenList_{self.hparams.cluster_num}.pickle" in os.listdir(self.hparams.dataset)) or (self.hparams.cluster_num == -1 and "corpus_tokenList.pickle" in os.listdir(self.hparams.dataset)):
                        print(f'***** Loading Dataset from Local!!')
                        self.tokId2tokText, self.tokId2groupId, self.groupId2tokId, self.trie, self.contextualized_tokid2emb, self.corpus_tokenList_dict = self._load_dataset(epoch=0)
                    else:
                        print(f'***** Constructing New One :) !!')
                        self.tokId2tokText, self.tokId2groupId, self.groupId2tokId, self.trie, self.contextualized_tokid2emb, self.corpus_tokenList_dict = self._dump_new_dataset(path="base")
                        self._dump_all(path="base")

                else:
                    self.load_epoch = int(self.hparams.resume_from_checkpoint.split('/')[-1].split('=')[1].split('-')[0])
                    base_path = "/".join(self.hparams.resume_from_checkpoint.split('/')[:-1])
                    base_path = os.path.join(base_path, f"best_tfmr_{self.load_epoch}")
                    if (self.hparams.cluster_num > 0 and f"k-means_corpus_tokenList_{self.hparams.cluster_num}.pickle" in os.listdir(base_path)) or (self.hparams.cluster_num == -1 and "corpus_tokenList.pickle" in os.listdir(base_path)):
                        print(f'***** Loading Dataset from Epoch: {self.load_epoch}!!')
                        self.tokId2tokText, self.tokId2groupId, self.groupId2tokId, self.trie, self.contextualized_tokid2emb, self.corpus_tokenList_dict = self._load_dataset(epoch=self.load_epoch)
                    else:
                        assert False
                        print(f'***** Constructing New One :) !!')
                        self.tokId2tokText, self.tokId2groupId, self.groupId2tokId, self.trie, self.contextualized_tokid2emb, self.corpus_tokenList_dict = self._dump_new_dataset(path="resume")
                        self._dump_all(path="base")
            if self.hparams.do_test:
                test_epoch = int(self.hparams.test_model_path.split('_')[-1])
                if (self.hparams.cluster_num > 0 and f"k-means_corpus_tokenList_{self.hparams.cluster_num}.pickle" in os.listdir(self.hparams.test_model_path)) or (self.hparams.cluster_num == -1 and "corpus_tokenList.pickle" in os.listdir(self.hparams.test_model_path)):
                    print(f'***** Loading Dataset from Local!!')
                    self.tokId2tokText, self.tokId2groupId, self.groupId2tokId, self.trie, self.contextualized_tokid2emb, self.corpus_tokenList_dict = self._load_dataset(epoch=test_epoch)
                else:
                    print(f'***** Constructing New One :) !!')
                    self.tokId2tokText, self.tokId2groupId, self.groupId2tokId, self.trie, self.contextualized_tokid2emb, self.corpus_tokenList_dict = self._dump_new_dataset(path="test")
                    self._dump_all(path="test")
            print(f'============== DONE! ================') 
        elif self.hparams.contextualized_file.endswith('.pickle'):
            if self.hparams.do_test or self.hparams.resume_from_checkpoint: 
                assert False
            self.contextualized_tokid2emb = pickle.load(
                open(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), "rb")
            )
            self.config.update(
                {"contextualized_file": os.path.join(self.hparams.dataset, self.hparams.contextualized_file)}#self.contextualized_tokid2emb}
            )  # tokId_emb.pickle
        elif self.hparams.contextualized_file.endswith('.hdf5'):
            if self.hparams.do_test or self.hparams.resume_from_checkpoint:
                assert False
            f = h5py.File(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), "r")
            self.contextualized_tokid2emb = {}
            for id in f.keys():
                self.contextualized_tokid2emb[int(id)] = self._get_emb_from_file(hf=f, _id=id, path=None, file_type="hdf5") 
            self.config.update(
                {"contextualized_file": os.path.join(self.hparams.dataset, self.hparams.contextualized_file)}#self.contextualized_tokid2emb}
            )  # tokId_emb.pickle
        else:
            assert False

        self.contextualized_emb_num = len(self.contextualized_tokid2emb.keys())
        self.config.update({"fp16": self.hparams.fp16})
        self.config.update({"train_c_emb": self.hparams.train_c_emb}) 
        self.config.update({"do_test": self.hparams.do_test}) 
        self.config.update(
            {"contextualized_emb_num": self.contextualized_emb_num}
        )
        self.config.update({"freeze_vocab_emb": self.hparams.freeze_vocab_emb})

        if self.hparams.contextualized_file is not None:
            self.groupId2tokId= pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.groupId2tokIdList), "rb"))
            self.tokId2groupId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2groupId), 'rb'))
            self.tokId2tokText = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2tokText), 'rb'))
            self.trie = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tree_path), "rb"))
            self.corpus_tokenList_dict = None

        self.save_epoch = []

    def _load_dataset(self, epoch):
        if epoch == 0:
            load_path = self.hparams.dataset
        else:
            load_path = os.path.join(self.hparams.output_dir, f"best_tfmr_{epoch}")

        if self.hparams.cluster_num > 0:
            tokId2tokText = pickle.load(open(os.path.join(load_path, f'k-means_clusterId_tokText_{self.hparams.cluster_num}.pickle'), 'rb'))
            tokId2groupId = pickle.load(open(os.path.join(load_path, f'k-means_clusterId_tokGroupId_{self.hparams.cluster_num}.pickle'), 'rb'))
            groupId2tokId = pickle.load(open(os.path.join(load_path, f'k-means_tokGroupId_clusterIdList_{self.hparams.cluster_num}.pickle'), 'rb'))
            trie = pickle.load(open(os.path.join(load_path, 'groupId_tree.pickle'), 'rb'))
            contextualized_tokid2emb = pickle.load(open(os.path.join(load_path, f'k-means_clusterId_clusterEmb_{self.hparams.cluster_num}.pickle'), 'rb'))
            corpus_tokenList_dict = pickle.load(open(os.path.join(load_path, f'k-means_corpus_tokenList_{self.hparams.cluster_num}.pickle'), 'rb'))
            self.config.update({'contextualized_file': os.path.join(load_path, f'k-means_clusterId_clusterEmb_{self.hparams.cluster_num}.pickle')})
        else:
            tokId2tokText = pickle.load(open(os.path.join(load_path, 'tokId_tokText.pickle'), 'rb'))
            tokId2groupId = pickle.load(open(os.path.join(load_path, 'tokId_tokGroupId.pickle'), 'rb'))
            groupId2tokId = pickle.load(open(os.path.join(load_path, 'tokGroupId_tokIdList.pickle'), 'rb'))
            trie = pickle.load(open(os.path.join(load_path, 'groupId_tree.pickle'), 'rb'))
            contextualized_tokid2emb = pickle.load(open(os.path.join(load_path, 'tokId_emb.pickle'), 'rb'))
            corpus_tokenList_dict = pickle.load(open(os.path.join(load_path, 'corpus_tokenList.pickle'), 'rb'))
            self.config.update({'contextualized_file': os.path.join(load_path, f'tokId_emb.pickle')})
        return tokId2tokText, tokId2groupId, groupId2tokId, trie, contextualized_tokid2emb, corpus_tokenList_dict

    def _dump_all(self, path):
        if self.hparams.cluster_num > 0:
            self._dump(f'k-means_clusterId_tokText_{self.hparams.cluster_num}', self.tokId2tokText, path)
            self._dump(f'k-means_clusterId_tokGroupId_{self.hparams.cluster_num}', self.tokId2groupId, path)
            self._dump(f'k-means_tokGroupId_clusterIdList_{self.hparams.cluster_num}', self.groupId2tokId, path)
            self._dump('groupId_tree', self.trie, path)
            self._dump(f'k-means_clusterId_clusterEmb_{self.hparams.cluster_num}', self.contextualized_tokid2emb, path)
            self._dump(f'k-means_corpus_tokenList_{self.hparams.cluster_num}', self.corpus_tokenList_dict, path)
            if path == "base":
               self.config.update({'contextualized_file': os.path.join(self.hparams.dump_path, f'k-means_clusterId_clusterEmb_{self.hparams.cluster_num}.pickle')})
            elif path == "test":
               self.config.update({'contextualized_file': os.path.join(self.hparams.test_model_path, f'k-means_clusterId_clusterEmb_{self.hparams.cluster_num}.pickle')})
            else:
               self.config.update({'contextualized_file': os.path.join(path, f'k-means_clusterId_clusterEmb_{self.hparams.cluster_num}.pickle')})
        else:
            self._dump('tokId_tokText', self.tokId2tokText, path)
            self._dump('tokId_tokGroupId', self.tokId2groupId, path)
            self._dump('tokGroupId_tokIdList', self.groupId2tokId, path)
            self._dump('groupId_tree', self.trie, path)
            self._dump('tokId_emb', self.contextualized_tokid2emb, path)
            self._dump('corpus_tokenList', self.corpus_tokenList_dict, path) 
            if path == "base":
               self.config.update({'contextualized_file': os.path.join(self.hparams.dump_path, f'tokId_emb.pickle')})
            elif path == "test":
               self.config.update({'contextualized_file': os.path.join(self.hparams.test_model_path, f'tokId_emb.pickle')})
            else:
               self.config.update({'contextualized_file': os.path.join(path, f'tokId_emb.pickle')})

    def _dump(self, f_name, value, path):
        if path == "base":
            save_path = self.hparams.dump_path
        elif path == "test":
            save_path = self.hparams.test_model_path
        else:
            save_path = path
        
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, f"{f_name}.pickle"), 'wb') as f:
            pickle.dump(value, f)

    def ids_to_text(self, _generated_ids, skip_special_tokens=True):
        generated_ids = []
        for _ids in _generated_ids:
            _ids = copy.deepcopy(_ids)
            _ids = _ids.detach().cpu().numpy()
            _text = [self.tokId2tokText[_id] for _id in _ids]
            generated_ids.append(self.dec_tok.convert_tokens_to_ids(_text))
        gen_text = self.dec_tok.batch_decode(
            generated_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True
        )
        text = self.lmap(str.strip, gen_text)
        if not skip_special_tokens:
            while "<pad>" in text:
               text = text.replace('<pad>', '')
        return text


    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_attention_mask=None,
        decoder_input_ids=None,
        lm_labels=None,
        return_dict=True
    ):
        if lm_labels is None:
            assert decoder_input_ids is not None
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                return_dict=return_dict
            )
        if decoder_input_ids is None:
            assert lm_labels is not None
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=lm_labels,
                return_dict=return_dict
            ) 

 
    def validation_step(self, batch, batch_idx):
        em_score, recall_score = self._val_step(batch, batch_idx)
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
    
        ret_dict = self._test_step(batch, batch_idx, return_elem=True)
        if ret_dict is None: return None 
        self.test_input_list.extend(ret_dict["input"])
        self.test_gt_list.extend(ret_dict["gt"])
        self.test_gt_tok_list.extend(ret_dict["gt_tok"])
        self.test_pred_list.extend(ret_dict["pred"])
        self.test_pred_tok_list.extend(ret_dict["pred_tok"])
        self.test_em_score_list.extend(ret_dict["em"])
        self.test_recall_score_list.extend(ret_dict["recall"])
        self._save_test() 

    def _save_test(self, epoch_end=False):
        os.makedirs(self.hparams.output_dir, exist_ok=True)
        _input = self.gather_list(self.test_input_list)
        _gt = self.gather_list(self.test_gt_list)
        _gt_tok = self.gather_list(self.test_gt_tok_list)
        _pred = self.gather_list(self.test_pred_list)
        _pred_tok = self.gather_list(self.test_pred_tok_list)
        _em = self.gather_list(self.test_em_score_list)
        _recall = self.gather_list(self.test_recall_score_list)
        assert len(_input) == len(_gt) == len(_pred) == len(_em) == len(_recall) == len(_gt_tok) == len(_pred_tok)
        if self.print:
            with open(self.test_save_name, "w") as f:
                json.dump(
                    {
                        "input": _input,
                        "gt": _gt,
                        "gt_tok": _gt_tok,
                        "pred": _pred,
                        "pred_tok": _pred_tok,
                        "em": _em,
                        "recall": _recall,
                    },
                    f,
                )
            if epoch_end:
                print(
                    f"Saving in {self.test_save_name}!\nnumber of elements: {len(_input)}"
                )
                print(f"EM: {np.array(_em).mean()}")
                print(f"Recall: {np.array(_recall).mean()}")

    def test_epoch_end(self, outputs):
        self._save_test(epoch_end=True)

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

    # TODO: do dumping in best_tfmr
    def on_save_checkpoint(self, checkpoint):
        save_path = os.path.join(
            self.hparams.output_dir, f"best_tfmr_{self.current_epoch}"
        )
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self._dump_all(save_path)

        self.save_epoch.append(self.current_epoch)
        if len(self.save_epoch) > 5:
            rm_file = f"best_tfmr_{self.save_epoch[0]}"
            os.system(f"rm -rf {os.path.join(self.hparams.output_dir, rm_file)}")

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

class SentenceT5(pl.LightningModule):
    def __init__(self, hparams):
        super(SentenceT5, self).__init__()
        #self.hparams = hparams 
        self.save_hyperparameters(hparams)

        if self.hparams.do_test:
            print(f"### Loading {hparams.test_model}")
            if self.hparams.bi_type=="encoder_decoder":
                self.model = T5Model.from_pretrained(hparams.test_model)
            elif self.hparams.bi_type=="encoder_mean":
                self.model = T5Model.from_pretrained(hparams.test_model).get_encoder()
            else:
                raise NotImplementedError("Check the embedding type!")
                    
            self.tokenizer = T5Tokenizer.from_pretrained(hparams.test_model)
        else:
            if self.hparams.bi_type=="encoder_decoder":
                self.model = T5Model.from_pretrained(hparams.model_name_or_path)
            elif self.hparams.bi_type=="encoder_mean":
                self.model = T5Model.from_pretrained(hparams.model_name_or_path).get_encoder()
            else:
                raise NotImplementedError("Check the embedding type!")
    
            self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)

        self.sim = Similarity(temp=0.01)
        self.gpu_num = torch.cuda.current_device()
        self.loss_fct = nn.CrossEntropyLoss()

        self.val_loss = []

        self.corpus = None
        self.corpus_embedding = []
        self.test = {'query': [], 'similarity_score': [], 'predict': []} 

    def get_dataset(self, split):
        dataset = SentenceDataset(self.tokenizer, split=split, args=self.hparams)
        return dataset

    def train_dataloader(self):
        train_dataset = self.get_dataset("train")
        print(f"## Number of Train dataset: {len(train_dataset)}")
        dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
        return dataloader

    def val_dataloader(self):
        val_dataset = self.get_dataset("validation")
        print(f"## Number of Val dataset: {len(val_dataset)}")
        dataloader = DataLoader(val_dataset, shuffle=False, batch_size=self.hparams.train_batch_size, drop_last=False, num_workers=self.hparams.num_workers)
        return dataloader

    def test_dataloader(self):
        test_dataset = self.get_dataset("test")
        print(f"## Number of Test dataset: {len(test_dataset)}")
        dataloader = DataLoader(test_dataset, shuffle=False, batch_size=self.hparams.eval_batch_size, drop_last=False, num_workers=self.hparams.num_workers)
        return dataloader

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None):
        if self.hparams.bi_type=="encoder-decoder":
            outputs = self.model(input_ids, 
                            attention_mask=attention_mask, 
                            decoder_input_ids=decoder_input_ids, 
                            decoder_attention_mask=decoder_attention_mask, 
                            )
        elif self.hparams.bi_type=="encoder-mean":
            outputs = self.model(input_ids, 
                            attention_mask=attention_mask, 
                            )

        return outputs
    
    def _get_embedding(self, batch):
        query_input_ids = batch['query_ids']
        query_attention_mask = batch['query_mask']
        key_input_ids = batch['key_ids']
        key_attention_mask = batch['key_mask']
        decoder_input_ids = batch['decoder_ids']
        decoder_attention_mask = batch['decoder_mask']

        query_outputs = self(
            input_ids=query_input_ids, 
            attention_mask=query_attention_mask, 
            decoder_input_ids=decoder_input_ids, 
            decoder_attention_mask=decoder_attention_mask
            ) 
        
        key_outputs = self(
            input_ids=key_input_ids, 
            attention_mask=key_attention_mask, 
            decoder_input_ids=decoder_input_ids, 
            decoder_attention_mask=decoder_attention_mask
            )

        if self.hparams.bi_type == "encoder-decoder":
            query_output = query_outputs.last_hidden_state[:, 0] # [bs, hidden_size=1024]
            key_output = key_outputs.last_hidden_state[:, 0]
            return query_output,  key_output 
        elif self.hparams.bi_type == "encoder-mean":
            assert False
            query_output = torch.mean(query_outputs.last_hidden_state, 1)
            key_output = torch.mean(key_outputs.last_hidden_state, 1)
            return query_output,  key_output
        else:
            raise NotImplementedError("Check the Embedding Type!")

    def _calculate_similarity(self, batch, test=False):
        z1, z2 = self._get_embedding(batch) # z1 -> [5, 1024]
        sim = torch.inner(z1, z2)
        return sim

    def _common_step(self, batch, test=False):
        _sim = self._calculate_similarity(batch, test)
        labels = torch.arange(_sim.size(0)).long().to(self.device)
        loss = self.loss_fct(_sim, labels)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def _gather_batch(self, _batch):
        gather = [None for _ in range(self.trainer.num_processes)]
        dist.all_gather_object(gather, _batch)
        return gather 

    def gather_batch(self, _batch):
        gather = self._gather_batch(_batch)
        output = []
        for z1, z2 in gather:
            if len(output) == 0:
                output.append(z1.to(self.device))
                output.append(z2.to(self.device))
            else:
                output[0] = torch.cat((output[0].to(self.device), z1.to(self.device)))
                output[1] = torch.cat((output[1].to(self.device), z2.to(self.device)))
        return output

    def validation_step(self, batch, batch_idx):
        # batch -> query_ids, query_mask, key_ids, key_mask, decoder_ids, decoder_mask, query
        loss = self._common_step(batch, test=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.val_loss.append(np.array(loss.cpu()))
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = np.mean(np.array(self.val_loss))
        self.val_loss = []
        self.log('val_avg_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return

    def configure_optimizers(self):
        model = self.model
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = Adafactor(optimizer_grouped_parameters, lr=self.hparams.learning_rate, warmup_init=False, scale_parameter=False, relative_step=False,)
        self.opt = optimizer 
       
        if self.hparams.lr_scheduler == 'constant':
            print("~!~! learning rate scheduler: CONSTANT")
            return [optimizer]
        elif self.hparams.lr_scheduler == "exponential":
            print("~!~! learning rate scheduler: EXPONENTIAL")
            len_data = len(self.train_dataloader())
            denominator=self.hparams.n_gpu
            steps_per_epoch=((len_data//denominator)+1)//self.hparams.gradient_accumulation_steps
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate, steps_per_epoch=steps_per_epoch, pct_start=0.1, epochs=self.hparams.num_train_epochs, anneal_strategy= 'linear', cycle_momentum=False)
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'learning_rate'}] 
 

    def on_save_checkpoint(self, checkpoint):
        save_path = os.path.join(self.hparams.output_dir, f"best_tfmr_epoch_{self.current_epoch}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

class T5BiEncoder(T5BaseClass):
    def __init__(self, args):
        super(T5BiEncoder, self).__init__()
        self.save_hyperparameters(args)

        if self.hparams.do_train:
            if self.hparams.bi_type in ['encoder_mean', 'encoder_first']:
                self.model = T5EncoderModel.from_pretrained(self.hparams.model_name_or_path)
            elif self.hparams.bi_type in ['encoder_decoder']:
                self.model = T5Model.from_pretrained(self.hparams.model_name_or_path)
            else:
                assert False
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)
            if self.print:
                print(f'@@@ In Training Mode...')
                print(f'@@@ Loading Model from {self.hparams.model_name_or_path}')
            self.em_score_list = []
            self.recall_score_list = []

            if self.hparams.periflow:
                self.connect_str, self.container_name = get_blob_info()
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)

        if self.hparams.do_test:
            if self.hparams.bi_type in ['encoder_mean', 'encoder_first']:
                self.model = T5EncoderModel.from_pretrained(self.hparams.test_model_path)
            elif self.hparams.bi_type in ['encoder_decoder']:
                self.model = T5Model.from_pretrained(self.hparams.test_model_path)
            else:
                assert False
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.test_model_path)
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

        print('!!! Loading Embedding !!!')
        if self.hparams.contextualized_file.endswith('dat'):
            self.faiss = True
            self.contextualized_tokid2emb = np.memmap(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), dtype=self._get_dtype(), mode="readonly", shape=(37000000, 1024))
            if "faiss.index" in os.listdir(self.hparams.dataset):
                self.contextualized_tensor = faiss.read_index(os.path.join(self.hparams.dataset, "faiss.index"))
                self.contextualized_token = json.load(open(os.path.join(self.hparams.dataset, "faiss.token.json")))
            else:
                print(f'BUILD Faiss Index!! ')
                self.contextualized_tensor = faiss.IndexFlatIP(1024)
                self.contextualized_token = []
                for i in tqdm(range(37000000)):
                    emb = self.contextualized_tokid2emb[i][:]
                    emb = np.expand_dims(emb, axis=0)
                    self.contextualized_tensor.add(emb)
                    self.contextualized_token.append(i)
                print(f'Saving!')
                faiss.write_index(self.contextualized_tensor, os.path.join(self.hparams.dataset, "faiss.index"))
                with open(os.path.join(self.hparams.dataset, "faiss.token.json"), 'w') as f:
                    json.dump(self.contextualized_token, f)


            #self.contextualized_tensor = torch.tensor(self.contextualized_tokid2emb)#.to(self.device)
            #self.contextualized_token = np.arange(37000000) 
        elif self.hparams.contextualized_file.endswith('pickle'):
            self.faiss = False 
            self.contextualized_tokid2emb = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), "rb"))
            self.contextualized_tensor = torch.tensor(list(self.contextualized_tokid2emb.values()))#.to(self.device)
            if self.hparams.fp16:
                assert False
                self.contextualized_tensor = self.contextualized_tensor.half()
            self.contextualized_token = list(self.contextualized_tokid2emb.keys())
        else:
            # assert False
            self.faiss = True
            self.contextualized_tokid2emb = np.memmap(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), dtype=self._get_dtype(), mode="readonly", shape=(27000000, 1024))
            if "faiss.index" in os.listdir(self.hparams.dataset):
                self.contextualized_tensor = faiss.read_index(os.path.join(self.hparams.dataset, "faiss.index"))
                self.contextualized_token = json.load(open(os.path.join(self.hparams.dataset, "faiss.token.json")))
            else:
                print(f'BUILD Faiss Index!! ')
                self.contextualized_tensor = faiss.IndexFlatIP(1024)
                self.contextualized_token = []
                for i in tqdm(range(27000000)):
                    emb = self.contextualized_tokid2emb[i][:]
                    emb = np.expand_dims(emb, axis=0)
                    self.contextualized_tensor.add(emb)
                    self.contextualized_token.append(i)
                print(f'Saving!')
                faiss.write_index(self.contextualized_tensor, os.path.join(self.hparams.dataset, "faiss.index"))
                with open(os.path.join(self.hparams.dataset, "faiss.token.json"), 'w') as f:
                    json.dump(self.contextualized_token, f)
            
        self.tokId2corpus = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2corpus), "rb"))
        if self.hparams.cluster_num < 0:
            self.corpus2tokId = self._get_corpus2tokId()
        else:
            self.corpus2tokId = self._get_corpus2clusterId()

        self.loss_fct = nn.CrossEntropyLoss()
        self.decoder_input_ids, self.decoder_attention_mask = self.get_decoder_input()

    def _get_corpus2tokId(self):
        corpus2tokId = defaultdict(list)
        for _tokId, _corpus in self.tokId2corpus.items():
            corpus2tokId[_corpus].append(_tokId)
        return corpus2tokId

    def _get_corpus2clusterId(self):
        corpus2clusterId = defaultdict(list)
        for _clusterId, _corpus_list in self.tokId2corpus.items():
            for _corpus in _corpus_list:
                corpus2clusterId[_corpus].append(_clusterId) 
        return corpus2clusterId

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
            #assert torch.count_nonzero(labels[i]).item() == len(all_ids)
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

    def _calculate_cluster_em(self, pred_list, output):
        if pred_list is None: 
            return 0
        if output in pred_list:
            return 100
        else:
            return 0 

    def _calculate_cluster_recall(self, pred_list, output):
        for pred in pred_list:
            if pred is None: continue
            if output in pred:
                return 100
        return 0

    def _calculate_score(self, indices, batch, batch_idx):
        em_list = []; recall_list = []; pred_list = []; pred_tok_list = []
        for idx, (_indice_list, _output, _input) in enumerate(zip(indices, batch['output'], batch['input'])):
            _predict = []; _pred_idx = []
            for _indice in _indice_list:
                tokId = self.contextualized_token[int(_indice)]
                _pred_idx.append(tokId)
                if tokId == 0 or tokId == 1:
                    _predict.append(None)
                else:
                    _predict.append(self.tokId2corpus[tokId])
            if self.hparams.cluster_num < 0:
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
            else:
                em_list.append(self._calculate_cluster_em(_predict[0], _output))
                recall_list.append(self._calculate_cluster_recall(_predict, _output))
                pred_list.append(_predict)
                pred_tok_list.append(_pred_idx)
        return em_list, recall_list, pred_list, pred_tok_list

    def validation_step(self, batch, batch_idx):
        query_output, _ = self._get_embedding(batch)
        if self.faiss:
            _, indices = self.contextualized_tensor.search(query_output.detach().cpu().numpy(), self.hparams.val_beam_size)
        else:
            scores = torch.inner(query_output.to('cpu'), self.contextualized_tensor)
            top_scores = torch.topk(scores, self.hparams.val_beam_size)
            indices = top_scores.indices # [# of query, self.hparams.val_beam_size]
        #scores = torch.inner(query_output.to(self.device), self.contextualized_tensor.to(self.device)) # [# of query, # of corpus] 
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
        scores = torch.inner(query_output.to('cpu'), self.contextualized_tensor) # [# of query, # of corpus] 
        #scores = torch.inner(query_output.to(self.device), self.contextualized_tensor.to(self.device)) # [# of query, # of corpus] 
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
        try:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
        except:
            print(f"## Error during saving {save_path}")

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

class T5FineTuner(T5grTuner):
    def __init__(self, args):
        super(T5FineTuner, self).__init__(args)
        # If in training mode, load ckpt for training
        if self.hparams.do_train:
            if self.print: print(self.config)
            if self.hparams.train_c_emb:
                self.model = contextualized_T5.from_pretrained(
                    self.hparams.model_name_or_path, config=self.config, ignore_mismatched_sizes=True
                )
            else:
                self.model = contextualized_T5.from_pretrained(
                    self.hparams.model_name_or_path, config=self.config
                )

            if self.hparams.gr_decoder_only_encoder_ckpt is not None:
                print(f'===== Loading encoder ckpt from.. {self.hparams.gr_decoder_only_encoder_ckpt}')
                m = torch.load(os.path.join(self.hparams.gr_decoder_only_encoder_ckpt, "pytorch_model.bin"))
                model_dict = self.model.state_dict()
                for k in m.keys():
                    if 'decoder.embed_tokens' in k:
                        continue
                    if k in model_dict:
                        pname = k
                        pval = m[k]
                        model_dict[pname] = pval.clone().to(model_dict[pname].device)
                self.model.load_state_dict(model_dict, strict=False)
            else:
                print(f'===== Encoder ckpt is same as Decoder ckpt')
            if self.hparams.gr_decoder_only:
                for n, p in self.model.get_encoder().named_parameters():
                    p.requires_grad = False
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.hparams.tokenizer_name_or_path
            )
            if self.hparams.periflow:
                self.connect_str, self.container_name = get_blob_info()
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)
            if self.print:
                print(f"@@@ Loading Model from {self.hparams.model_name_or_path}")
                print(f'@@@ Loading decoder embedding: {self.hparams.embedding_model}')
            self.em_score_list = []
            self.recall_score_list = []
            self.dev_input2output = json.load(open(os.path.join(self.hparams.dataset, self.hparams.dev_input2output), "r"))

        # If in testing mode, load ckpt for inference
        if self.hparams.do_test:
            self.model = contextualized_T5.from_pretrained(
                self.hparams.test_model_path, config=self.config, ignore_mismatched_sizes=True
            )
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.test_model_path)
            if self.print:
                print(f"@@@ Loading Model from {self.hparams.test_model_path}")
            
            self.test_save_name = os.path.join(self.hparams.output_dir, f"{self.hparams.test_name}_{self.hparams.tree_type}_mbs_{self.hparams.max_beam_search}_result_beam{self.hparams.val_beam_size}.json")               
            if os.path.exists(self.test_save_name):
                 prev_f = json.load(open(self.test_save_name))
                 print(f"@@@ Loading Previous file!! => #: {len(prev_f['input'])}")
                 self.test_input_list = prev_f['input']
                 self.test_gt_list = prev_f['gt']
                 self.test_gt_tok_list = prev_f['gt_tok']
                 self.test_pred_list = prev_f['pred']
                 self.test_pred_tok_list = prev_f['pred_tok']
                 self.test_em_score_list = prev_f['em']
                 self.test_recall_score_list = prev_f['recall']
            else:
                 print(f'@@@ Initialize Test!!')
                 self.test_input_list = []
                 self.test_gt_list = []
                 self.test_gt_tok_list = []
                 self.test_pred_list = []
                 self.test_pred_tok_list = []
                 self.test_em_score_list = []
                 self.test_recall_score_list = []

        if self.hparams.freeze_encoder:
            if self.print:
                print(f"@@@ Freeze Encoder!")
                encoder = self.model.get_encoder()
                for n, p in encoder.named_parameters():
                    p.requires_grad=False
        #### Tokenizer for generation step!
        self.dec_tok = AutoTokenizer.from_pretrained(
            self.hparams.embedding_model
        )
        
        if self.hparams.tree_type == "nodeId":
            nodeId_sup = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.nodeId_sup), 'rb'))
            self.nodeId2groupdId = nodeId_sup['group_set']
            self.nodeId2tokId = nodeId_sup['token_set']
            self.groupId2nodeId = nodeId_sup['inv_group_set']
            self.tokId2nodeId = nodeId_sup['inv_token_set']
        self.cnt_over = 0
        self.len_test_dataset = len(self.test_dataloader())

    def _get_dataset(self, split):
        dataset = GENREDataset(
            tokenizer=self.tokenizer,
            split=split,
            hparams=self.hparams,
            tokid2emb=self.contextualized_tokid2emb 
        )
        return dataset
    def _get_max_tokId_from_tokIdList(self, tokIdList, score):
        tokIdList = sorted(tokIdList)
        idx = score[tokIdList].detach().cpu().numpy().argmax()
        max_tokId = tokIdList[idx]
        return max_tokId
    def _get_tokIdList_from_nodeIdList(self, nodeIdList, score):                        
        assert self.hparams.tree_type == "nodeId"            
        tokIdList = []
        if self.hparams.max_beam_search:
            for nodeId in nodeIdList:
                max_tokId = self._get_max_tokId_from_tokIdList(list(self.nodeId2tokId[nodeId]), score)
                tokIdList.append(max_tokId)
            return list(set(tokIdList))
        else:                
            for nodeId in nodeIdList:
                tokIdList.extend(list(self.nodeId2tokId[nodeId]))                
            return list(set(tokIdList))

    def _get_tokIdList_from_groupIdList(self, groupIdList, score):
        assert self.hparams.tree_type == "groupId"
        tokIdList = []
        
        # for groupId Tree with max beam search
        if self.hparams.max_beam_search:
            for groupId in groupIdList:
                max_tokId = self._get_max_tokId_from_tokIdList(self.groupId2tokId[groupId], score)
                tokIdList.append(max_tokId)
            return list(set(tokIdList))
        # for normal groupId Tree
        else:
            for groupId in groupIdList:
                tokIdList.extend(self.groupId2tokId[groupId])
            return list(set(tokIdList))
   
    def _get_nodeId_from_tokId(self, tokId):
        nodeId = list(self.tokId2node/_teId[tokId])
        assert len(nodeId) == 1
        return nodeId[0] 
    
    def _get_groupId_from_tokId(self, tokId):
        return self.tokId2groupId[tokId]


    def _loss(self, batch):
        lm_labels = copy.deepcopy(batch["target_ids"])
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
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
    def get_list(self, batch_id, input_ids, score, trie_list):
        # starts with pad token & groupId for pad token is -1
        assert input_ids[0] == 0
        assert trie_list is not None and isinstance(trie_list, list)
        return self._get_from_trie(input_ids, trie_list[batch_id], score)
    
    def get(self, batch_id, input_ids, score, trie_dict=None):
        # starts with pad token & groupId for pad token is -1
        assert input_ids[0] == 0
        if trie_dict is None:
            trie_dict = self.trie
        return self._get_from_trie(input_ids, trie_dict, score)

    """
    input_ids가 들어오면, 해당 tokId가 속한 groupId 찾고, 그걸 가지고 trie_dict 넘어간 다음
    해당 subtree의 key들(groupId) 를 모은 tokId return
    """
    def _get_from_trie(self, input_ids, trie_dict, score):
        #print(f"input_ids: {input_ids}")
        if self.hparams.tree_type == "groupId":
            if len(input_ids) == 0:
                possible_GroupList = list(trie_dict.keys())
                tokIdList = self._get_tokIdList_from_groupIdList(possible_GroupList, score)
                return tokIdList
            else:
                curGroupId = self._get_groupId_from_tokId(input_ids[0])
                if curGroupId in list(trie_dict.keys()):
                    return self._get_from_trie(input_ids[1:], trie_dict[curGroupId], score) 
                else:
                    return []
        elif self.hparams.tree_type == "nodeId":
            if input_ids[-1] == 1:
                return []
            else:
                NodeId = self._get_nodeId_from_tokId(input_ids[-1])
                next_nId_List = list(trie_dict[NodeId])
                tokIdList = self._get_tokIdList_from_nodeIdList(next_nId_List, score)
                return tokIdList
        elif self.hparams.tree_type == "clusterId":
            if len(input_ids) == 0:
                return list(trie_dict.keys()) 
            else:
                if input_ids[0] in list(trie_dict.keys()):
                    return self._get_from_trie(input_ids[1:], trie_dict[clusterId], score)
                else:
                    return []
        else:
            raise NotImplementedError('tree type should be either groupId_tree or nodeId_tree!')

    def calculate_scores(self, preds, gt_text, query, batch_idx):
        em_list = []
        recall_list = []
        for idx, (_query, _pred, _gt) in enumerate(zip(query, preds, gt_text)):
            _em = self._calculate_em(_pred[0], _gt)
            _recall = self._calculate_recall(_pred, _gt)
            """
            if self.print and idx == 0:
                print(f"$" * 50)
                print(f"query: {_query}\npreds: {_pred}\ngt: {_gt}")
                print(f"em: {_em} // recall: {_recall}")
                print(f"$" * 50)
                print(" ")
            """
            em_list.append(_em)
            recall_list.append(_recall)
        return em_list, recall_list

    def _val_step(self, batch, batch_idx, return_elem=False):
        # calculates recall and em -> returns the list of each score
        _generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            #decoder_attention_mask=batch["target_mask"],
            max_length=self.hparams.max_output_length,
            num_beams=self.hparams.val_beam_size,
            num_return_sequences=self.hparams.val_beam_size,
            prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get(
                batch_id, sent.tolist(), scores
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
        generated_ids = [
            _generated_ids[
               i * self.hparams.val_beam_size : (i+1) * self.hparams.val_beam_size
            ].detach().cpu().numpy().tolist()
            for i in range(inum)
        ]
        output_list = [self.dev_input2output[_input] for _input in batch["input"]]
        em_list, recall_list = self.calculate_scores(
            generated_text, output_list, batch["input"], batch_idx
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
                "gt_tok": list(batch["target_ids"].detach().cpu().numpy().tolist()),
                "pred": list(generated_text),
                "pred_tok": list(generated_ids),
                "em": list(em_list),
                "recall": list(recall_list),
            }
        else:
            return em_list, recall_list

    def _find_unique_path(self, seq, trie):
        _ids = seq.tolist()
        _ids = [int(elem) for elem in _ids]
        
        generated_ids = []
        if _ids[-1] == 0:
            for el in range(len(_ids)):
                if el != 0 and _ids[el] == 0:
                    continue
                else:
                    generated_ids.append(_ids[el])
        else:
            generated_ids = _ids 
        assert generated_ids[-1] != 0
        generated_ids = generated_ids[:-1]
        
        cur_dict = trie 
        for i in range(len(generated_ids) - 1):
            cur_dict = cur_dict[int(generated_ids[i])]
        if len(cur_dict[generated_ids[-1]]) == 0:
            add_list.append(generated_ids)
        else:
            cur_dict = cur_dict[int(generated_ids[-1])]
            # check if it is an unique path
            keys = list(cur_dict.keys())
            if len(keys) > 1:
                print(f"More than 1: {keys}")
                keys = [keys[0]]
            while len(cur_dict[keys[0]]) != 0:
                generated_ids.append(keys[0])
                cur_dict = cur_dict[keys[0]]
                keys = list(cur_dict.keys())
                if len(keys) > 1:
                    print(f"More than 1: {keys}")
                    keys = [keys[0]]
            generated_ids.append(1)
        return generated_ids

    def _test_step(self, batch, batch_idx, return_elem=False):
       
        # for case where it resume test 
        test_input = []
        for _input in batch['input']:
            if _input in self.test_input_list: continue
            else: test_input.append(_input) 
        test_num = len(test_input)
        start_num = len(batch['input'])-test_num
        if test_num == 0: return None
        test_output = batch['output'][start_num:]
        test_source_ids = batch['source_ids'][start_num:]
        test_source_masks = batch['source_mask'][start_num:]
        test_target_ids = batch['target_ids'][start_num:]
        test_target_masks = batch['target_mask'][start_num:]
        assert len(test_output) == test_num, f'test_output: {len(test_output)}\ttest_num: {test_num}'
        _trie_list = [copy.deepcopy(self.trie) for _ in range(test_num)]
        
        #unique_pred = []; unique_ids = []
        unique_pred_list = [[] for _ in range(test_num)] 
        unique_ids_list = [[] for _ in range(test_num)] 
        over = [0]*test_num
        for _iter in range(self.hparams.ret_num*2):
            print("="*80)
            print(f"iter: {_iter} // DONE: {over.count(1)} / {test_num}")
            print(unique_pred_list)
            print("="*80)
            _generated_ids = self.model.generate(
                test_source_ids, 
                attention_mask=test_source_masks,
                use_cache=True,
                decoder_attention_mask=test_target_masks,
                max_length=self.hparams.max_output_length,
                num_beams=self.hparams.val_beam_size,
                num_return_sequences=self.hparams.val_beam_size,
                prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get_list(
                    batch_id, sent.tolist(), scores, trie_list=_trie_list
                ),
                early_stopping=True,
            )
            _generated_text = self.ids_to_text(_generated_ids)
            inum = len(_generated_ids) // self.hparams.val_beam_size
            assert inum == len(test_output) 
            generated_text = [
                _generated_text[
                    i * self.hparams.val_beam_size : (i + 1) * self.hparams.val_beam_size
                ]
                for i in range(inum)
            ]
            generated_ids = [
                _generated_ids[
                    i * self.hparams.val_beam_size : (i+1) * self.hparams.val_beam_size
                ].detach().cpu().numpy().tolist()
                for i in range(inum)
            ]
            """ 
            for b_texts, b_ids in zip(generated_text, generated_ids):
                for _text, _ids in zip(b_texts, b_ids):
                    g_ids = [self.tokId2groupId[el] for el in _ids]
            """
            print(f"prediction: {generated_text}")
            print("*"*80) 
            for bid, (batch_text, batch_ids) in enumerate(zip(generated_text, generated_ids)):
                if over[bid] == 1: continue
                ### iterate over batch (val_beam_size)
                _upper_ids = []
                _trie_dict = _trie_list[bid]
                for _text, _ids in zip(batch_text, batch_ids):
                    if _ids in unique_ids_list[bid]:
                        continue
                    else:
                        if _text in unique_pred_list[bid]:
                            assert not self.hparams.tree_type == "nodeId"
                            _upper_ids.append([self.tokId2groupId[el] for el in _ids]) 
                        elif len(unique_pred_list[bid]) < self.hparams.ret_num:
                            unique_pred_list[bid].append(_text)
                            unique_ids_list[bid].append(_ids)
                            if self.hparams.tree_type == "groupId":
                                _upper_ids.append([self.tokId2groupId[el] for el in _ids]) 
                            elif self.hparams.tree_type == "nodeId":
                                temp = []
                                eos_pos = (np.array(_ids) == 1).nonzero()[0][0]
                                for el in _ids[:eos_pos]:
                                    assert len(self.tokId2nodeId[el]) == 1, self.tokId2nodeId[el]
                                    temp.append(list(self.tokId2nodeId[el])[0])
                                # find the end token
                                cur_nId = temp[-1]
                                next_nId = _trie_dict[cur_nId]
                                end_nId = list(next_nId.intersection(self.tokId2nodeId[1]))
                                assert len(end_nId) == 1
                                temp.append(end_nId[0])
                                _upper_ids.append(temp)
                            else:
                                assert False
                        else:
                            pass
                # remove from _trie_dict
                _trie_dict = self._remove_prev_from_trie(_trie_dict, _upper_ids)
                _trie_list[bid] = _trie_dict
                if len(unique_pred_list[bid]) >= self.hparams.ret_num:
                    over[bid] = 1
            if over.count(1) == test_num:
                break
            """
            if (over.count(1) == test_num) or (self.cnt_over == self.len_test_dataset):
                self.cnt_over += over.count(1)
                break
            """
            
        for i in range(len(unique_pred_list)):
            unique_pred_list[i] = unique_pred_list[i][:self.hparams.ret_num]
        #self._flush_first_beam_dict()
        #if self.print: print(f"## UNIQUE PRED: {unique_pred[:self.hparams.val_beam_size]}")
        em_list, recall_list = self.calculate_scores(
            unique_pred_list, batch["output"], batch["input"], batch_idx
        )
        if return_elem:
            assert (
                len(list(test_input))
                == len(list(generated_text))
                == len(list(em_list))
                == len(list(unique_pred_list))
                == len(list(unique_ids_list))
            )
            return {
                "input": list(test_input),
                "gt": list(test_output),
                "gt_tok": [el.detach().cpu().numpy().tolist() for el in test_target_ids],
                "pred": list(unique_pred_list),
                "pred_tok": list(unique_ids_list),
                "em": list(em_list),
                "recall": list(recall_list),
            }
        else:
            return em_list, recall_list


class T5TestTuner(T5grTuner):
    def __init__(self, args):
        super(T5TestTuner, self).__init__(args)
        if self.print: print(f" &&&&&& Loading.. T5TestTuner!!")
        # If in training mode, load ckpt for training
        if self.hparams.do_train:
            if self.print: print(self.config)
            if self.hparams.train_c_emb:
                self.model = test_T5.from_pretrained(
                    self.hparams.model_name_or_path, config=self.config, ignore_mismatched_sizes=True
                )
            else:
                self.model = test_T5.from_pretrained(
                    self.hparams.model_name_or_path, config=self.config
                )

            if self.hparams.gr_decoder_only_encoder_ckpt is not None:
                print(f'===== Loading encoder ckpt from.. {self.hparams.gr_decoder_only_encoder_ckpt}')
                m = torch.load(os.path.join(self.hparams.gr_decoder_only_encoder_ckpt, "pytorch_model.bin"))
                model_dict = self.model.state_dict()
                for k in m.keys():
                    if 'decoder.embed_tokens' in k:
                        continue
                    if k in model_dict:
                        pname = k
                        pval = m[k]
                        model_dict[pname] = pval.clone().to(model_dict[pname].device)
                self.model.load_state_dict(model_dict, strict=False)
            else:
                print(f'===== Encoder ckpt is same as Decoder ckpt')
            if self.hparams.gr_decoder_only:
                for n, p in self.model.get_encoder().named_parameters():
                    p.requires_grad = False
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.hparams.tokenizer_name_or_path
            )
            if self.hparams.periflow:
                self.connect_str, self.container_name = get_blob_info()
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)
            if self.print:
                print(f"@@@ Loading Model from {self.hparams.model_name_or_path}")
                print(f'@@@ Loading decoder embedding: {self.hparams.embedding_model}')
            self.em_score_list = []
            self.recall_score_list = []
            self.dev_input2output = json.load(open(os.path.join(self.hparams.dataset, self.hparams.dev_input2output), "r"))

        # If in testing mode, load ckpt for inference
        if self.hparams.do_test:
            self.model = test_T5.from_pretrained(
                self.hparams.test_model_path, config=self.config, ignore_mismatched_sizes=True
            )
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.test_model_path)
            if self.print:
                print(f"@@@ Loading Model from {self.hparams.test_model_path}")
            
            self.test_save_name = os.path.join(self.hparams.output_dir, f"{self.hparams.test_name}_{self.hparams.tree_type}_mbs_{self.hparams.max_beam_search}_result_beam{self.hparams.val_beam_size}.json")               
            if os.path.exists(self.test_save_name):
                 prev_f = json.load(open(self.test_save_name))
                 print(f"@@@ Loading Previous file!! => #: {len(prev_f['input'])}")
                 self.test_input_list = prev_f['input']
                 self.test_gt_list = prev_f['gt']
                 self.test_gt_tok_list = prev_f['gt_tok']
                 self.test_pred_list = prev_f['pred']
                 self.test_pred_tok_list = prev_f['pred_tok']
                 self.test_em_score_list = prev_f['em']
                 self.test_recall_score_list = prev_f['recall']
            else:
                 print(f'@@@ Initialize Test!!')
                 self.test_input_list = []
                 self.test_gt_list = []
                 self.test_gt_tok_list = []
                 self.test_pred_list = []
                 self.test_pred_tok_list = []
                 self.test_em_score_list = []
                 self.test_recall_score_list = []

        if self.hparams.freeze_encoder:
            if self.print:
                print(f"@@@ Freeze Encoder!")
                encoder = self.model.get_encoder()
                for n, p in encoder.named_parameters():
                    p.requires_grad=False
        #### Tokenizer for generation step!
        self.dec_tok = AutoTokenizer.from_pretrained(
            self.hparams.embedding_model
        )
        
        if self.hparams.tree_type == "nodeId":
            nodeId_sup = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.nodeId_sup), 'rb'))
            self.nodeId2groupdId = nodeId_sup['group_set']
            self.nodeId2tokId = nodeId_sup['token_set']
            self.groupId2nodeId = nodeId_sup['inv_group_set']
            self.tokId2nodeId = nodeId_sup['inv_token_set']
        self.cnt_over = 0
        self.len_test_dataset = len(self.test_dataloader())

    def _get_dataset(self, split):
        dataset = TESTDataset(
            tokenizer=self.tokenizer,
            split=split,
            hparams=self.hparams,
            tokid2emb=self.contextualized_tokid2emb 
        )
        return dataset
    def _get_max_tokId_from_tokIdList(self, tokIdList, score):
        tokIdList = sorted(tokIdList)
        idx = score[tokIdList].detach().cpu().numpy().argmax()
        max_tokId = tokIdList[idx]
        return max_tokId
    def _get_tokIdList_from_nodeIdList(self, nodeIdList, score):                        
        assert self.hparams.tree_type == "nodeId"            
        tokIdList = []
        if self.hparams.max_beam_search:
            for nodeId in nodeIdList:
                max_tokId = self._get_max_tokId_from_tokIdList(list(self.nodeId2tokId[nodeId]), score)
                tokIdList.append(max_tokId)
            return list(set(tokIdList))
        else:                
            for nodeId in nodeIdList:
                tokIdList.extend(list(self.nodeId2tokId[nodeId]))                
            return list(set(tokIdList))

    def _get_tokIdList_from_groupIdList(self, groupIdList, score):
        assert self.hparams.tree_type == "groupId"
        tokIdList = []
        
        # for groupId Tree with max beam search
        if self.hparams.max_beam_search:
            for groupId in groupIdList:
                max_tokId = self._get_max_tokId_from_tokIdList(self.groupId2tokId[groupId], score)
                tokIdList.append(max_tokId)
            return list(set(tokIdList))
        # for normal groupId Tree
        else:
            for groupId in groupIdList:
                tokIdList.extend(self.groupId2tokId[groupId])
            return list(set(tokIdList))
   
    def _get_nodeId_from_tokId(self, tokId):
        nodeId = list(self.tokId2node/_teId[tokId])
        assert len(nodeId) == 1
        return nodeId[0] 
    
    def _get_groupId_from_tokId(self, tokId):
        return self.tokId2groupId[tokId]


    def _loss(self, batch):
        lm_labels = copy.deepcopy(batch["target_ids"])
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
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
    def get_list(self, batch_id, input_ids, score, trie_list):
        # starts with pad token & groupId for pad token is -1
        assert input_ids[0] == 0
        assert trie_list is not None and isinstance(trie_list, list)
        return self._get_from_trie(input_ids, trie_list[batch_id], score)
    
    def get(self, batch_id, input_ids, score, trie_dict=None):
        # starts with pad token & groupId for pad token is -1
        assert input_ids[0] == 0
        if trie_dict is None:
            trie_dict = self.trie
        return self._get_from_trie(input_ids, trie_dict, score)

    """
    input_ids가 들어오면, 해당 tokId가 속한 groupId 찾고, 그걸 가지고 trie_dict 넘어간 다음
    해당 subtree의 key들(groupId) 를 모은 tokId return
    """
    def _get_from_trie(self, input_ids, trie_dict, score):
        #print(f"input_ids: {input_ids}")
        if self.hparams.tree_type == "groupId":
            if len(input_ids) == 0:
                possible_GroupList = list(trie_dict.keys())
                tokIdList = self._get_tokIdList_from_groupIdList(possible_GroupList, score)
                return tokIdList
            else:
                curGroupId = self._get_groupId_from_tokId(input_ids[0])
                if curGroupId in list(trie_dict.keys()):
                    return self._get_from_trie(input_ids[1:], trie_dict[curGroupId], score) 
                else:
                    return []
        elif self.hparams.tree_type == "nodeId":
            if input_ids[-1] == 1:
                return []
            else:
                NodeId = self._get_nodeId_from_tokId(input_ids[-1])
                next_nId_List = list(trie_dict[NodeId])
                tokIdList = self._get_tokIdList_from_nodeIdList(next_nId_List, score)
                return tokIdList
        elif self.hparams.tree_type == "clusterId":
            if len(input_ids) == 0:
                return list(trie_dict.keys()) 
            else:
                if input_ids[0] in list(trie_dict.keys()):
                    return self._get_from_trie(input_ids[1:], trie_dict[clusterId], score)
                else:
                    return []
        else:
            raise NotImplementedError('tree type should be either groupId_tree or nodeId_tree!')

    def calculate_scores(self, preds, gt_text, query, batch_idx):
        em_list = []
        recall_list = []
        for idx, (_query, _pred, _gt) in enumerate(zip(query, preds, gt_text)):
            _em = self._calculate_em(_pred[0], _gt)
            _recall = self._calculate_recall(_pred, _gt)
            """
            if self.print and idx == 0:
                print(f"$" * 50)
                print(f"query: {_query}\npreds: {_pred}\ngt: {_gt}")
                print(f"em: {_em} // recall: {_recall}")
                print(f"$" * 50)
                print(" ")
            """
            em_list.append(_em)
            recall_list.append(_recall)
        return em_list, recall_list

    def _val_step(self, batch, batch_idx, return_elem=False):
        # calculates recall and em -> returns the list of each score
        _generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            #decoder_attention_mask=batch["target_mask"],
            max_length=self.hparams.max_output_length,
            num_beams=self.hparams.val_beam_size,
            num_return_sequences=self.hparams.val_beam_size,
            prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get(
                batch_id, sent.tolist(), scores
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
        generated_ids = [
            _generated_ids[
               i * self.hparams.val_beam_size : (i+1) * self.hparams.val_beam_size
            ].detach().cpu().numpy().tolist()
            for i in range(inum)
        ]
        output_list = [self.dev_input2output[_input] for _input in batch["input"]]
        em_list, recall_list = self.calculate_scores(
            generated_text, output_list, batch["input"], batch_idx
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
                "gt_tok": list(batch["target_ids"].detach().cpu().numpy().tolist()),
                "pred": list(generated_text),
                "pred_tok": list(generated_ids),
                "em": list(em_list),
                "recall": list(recall_list),
            }
        else:
            return em_list, recall_list

    def _find_unique_path(self, seq, trie):
        _ids = seq.tolist()
        _ids = [int(elem) for elem in _ids]
        
        generated_ids = []
        if _ids[-1] == 0:
            for el in range(len(_ids)):
                if el != 0 and _ids[el] == 0:
                    continue
                else:
                    generated_ids.append(_ids[el])
        else:
            generated_ids = _ids 
        assert generated_ids[-1] != 0
        generated_ids = generated_ids[:-1]
        
        cur_dict = trie 
        for i in range(len(generated_ids) - 1):
            cur_dict = cur_dict[int(generated_ids[i])]
        if len(cur_dict[generated_ids[-1]]) == 0:
            add_list.append(generated_ids)
        else:
            cur_dict = cur_dict[int(generated_ids[-1])]
            # check if it is an unique path
            keys = list(cur_dict.keys())
            if len(keys) > 1:
                print(f"More than 1: {keys}")
                keys = [keys[0]]
            while len(cur_dict[keys[0]]) != 0:
                generated_ids.append(keys[0])
                cur_dict = cur_dict[keys[0]]
                keys = list(cur_dict.keys())
                if len(keys) > 1:
                    print(f"More than 1: {keys}")
                    keys = [keys[0]]
            generated_ids.append(1)
        return generated_ids

    def _test_step(self, batch, batch_idx, return_elem=False):
       
        # for case where it resume test 
        test_input = batch['input'] 
        test_output = batch['output']
        test_source_ids = batch['source_ids']
        test_source_masks = batch['source_mask']
        test_target_ids = batch['target_ids']
        test_target_masks = batch['target_mask']

        _generated_ids = self.model.generate(
            test_source_ids, 
            attention_mask=test_source_masks,
            use_cache=True,
            decoder_attention_mask=test_target_masks,
            max_length=self.hparams.max_output_length,
            num_beams=self.hparams.val_beam_size,
            num_return_sequences=self.hparams.ret_num,
            prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get(
                batch_id, sent.tolist(), scores
            ),
            early_stopping=True,
        )
        _generated_text = self.ids_to_text(_generated_ids)
        inum = len(_generated_ids) // self.hparams.ret_num
        assert inum == len(test_output) 
        generated_text = [
            _generated_text[
                i * self.hparams.ret_num : (i + 1) * self.hparams.ret_num
            ]
            for i in range(inum)
        ]
        generated_ids = [
            _generated_ids[
                i * self.hparams.ret_num : (i+1) * self.hparams.ret_num
            ].detach().cpu().numpy().tolist()
            for i in range(inum)
        ]

        em_list, recall_list = self.calculate_scores(
            generated_text, batch["output"], batch["input"], batch_idx
        )

        if return_elem:
            assert (
                len(list(test_input))
                == len(list(generated_text))
                == len(list(em_list))
            )
            return {
                "input": list(test_input),
                "gt": list(test_output),
                "gt_tok": [el.detach().cpu().numpy().tolist() for el in test_target_ids],
                "pred": list(generated_text),
                "pred_tok": list(generated_ids),
                "em": list(em_list),
                "recall": list(recall_list),
            }
        else:
            return em_list, recall_list


class T5_title_context(T5grTuner):
    def __init__(self, args):
        super(T5_title_context, self).__init__(args)
        # If in training mode, load ckpt for training
        print("using title_context model")
        if self.hparams.do_train:
            if self.hparams.train_c_emb:
                self.model = contextualized_T5.from_pretrained(
                    self.hparams.model_name_or_path, config=self.config, ignore_mismatched_sizes=True
                )
            else:
                self.model = contextualized_T5.from_pretrained(
                    self.hparams.model_name_or_path, config=self.config
                )
            if self.hparams.gr_decoder_only_encoder_ckpt is not None:
                print(f'===== Loading encoder ckpt from.. {self.hparams.gr_decoder_only_encoder_ckpt}')
                m = torch.load(os.path.join(self.hparams.gr_decoder_only_encoder_ckpt, "pytorch_model.bin"))
                model_dict = self.model.state_dict()
                for k in m.keys():
                    if 'decoder.embed_tokens' in k:
                        continue
                    if k in model_dict:
                        pname = k
                        pval = m[k]
                        model_dict[pname] = pval.clone().to(model_dict[pname].device)
                self.model.load_state_dict(model_dict, strict=False)
            else:
                print(f'===== Encoder ckpt is same as Decoder ckpt')
            if self.hparams.gr_decoder_only:
                for n, p in self.model.get_encoder().named_parameters():
                    p.requires_grad = False
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.hparams.tokenizer_name_or_path
            )


            if self.hparams.periflow:
                self.connect_str, self.container_name = get_blob_info()
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)
            if self.print:
                print(f"@@@ Loading Model from {self.hparams.model_name_or_path}")
                print(f'@@@ Loading decoder embedding: {self.hparams.embedding_model}')
            self.em_score_list = []
            self.recall_score_list= []
            self.f1_score_list = []
            self.title_score_list= []
            self.context_score_list= []
            self.dev_input2output = json.load(open(os.path.join(self.hparams.dataset, self.hparams.dev_input2output), "r"))

        # If in testing mode, load ckpt for inference
        if self.hparams.do_test:
            self.model = contextualized_T5.from_pretrained(
                self.hparams.test_model_path, config=self.config, ignore_mismatched_sizes=True
            )
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.test_model_path)
            if self.print:
                print(f"@@@ Loading Model from {self.hparams.test_model_path}")
            
            self.test_save_name = os.path.join(self.hparams.output_dir, f"{self.hparams.test_name}_{self.hparams.tree_type}_mbs_{self.hparams.max_beam_search}_result_beam{self.hparams.val_beam_size}.json")               
            if os.path.exists(self.test_save_name):
                 prev_f = json.load(open(self.test_save_name))
                 print(f"@@@ Loading Previous file!! => #: {len(prev_f['input'])}")
                 self.test_input_list = prev_f['input']
                 self.test_gt_list = prev_f['gt']
                 self.test_pred_list = prev_f['pred']
                 self.em_score_list = prev_f['em']
                 self.title_em_score_list = prev_f['title_em']
                 self.f1_score_list = prev_f['f1']
                 self.title_f1_score_list = prev_f['title_f1']
            else:
                 print(f'@@@ Initialize Test!!')
                 self.test_input_list = []
                 self.test_gt_list = []
                 self.test_pred_list = []
                 self.em_score_list= []
                 self.title_em_score_list= []
                 self.f1_score_list= []
                 self.title_f1_score_list= []
           
            self.dev_input2output = json.load(open(os.path.join(self.hparams.dataset, self.hparams.dev_input2output), "r"))

        if self.hparams.freeze_encoder:
            if self.print:
                print(f"@@@ Freeze Encoder!")
                encoder = self.model.get_encoder()
                for n, p in encoder.named_parameters():
                    p.requires_grad=False
        #### Tokenizer for generation step!
        self.dec_tok = AutoTokenizer.from_pretrained(
            self.hparams.embedding_model
        )

        
        if self.hparams.tree_type == "nodeId":
            nodeId_sup = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.nodeId_sup), 'rb'))
            self.nodeId2groupdId = nodeId_sup['group_set']
            self.nodeId2tokId = nodeId_sup['token_set']
            self.groupId2nodeId = nodeId_sup['inv_group_set']
            self.tokId2nodeId = nodeId_sup['inv_token_set']
        self.cnt_over = 0
        self.len_test_dataset = len(self.test_dataloader())


        print("adding special tokens")
        self.tokenizer.add_tokens(["<title>","<context>"])
        self.dec_tok.add_tokens(["<title>","<context>"])

    def _get_dataset(self, split):
        dataset = GENREDataset(
            tokenizer=self.tokenizer,
            split=split,
            hparams=self.hparams,
            tokid2emb=self.contextualized_tokid2emb 
        )
        return dataset

    def _get_max_tokId_from_tokIdList(self, tokIdList, score):
        tokIdList = sorted(tokIdList)
        idx = score[tokIdList].detach().cpu().numpy().argmax()
        max_tokId = tokIdList[idx]
        return max_tokId
    def _get_tokIdList_from_nodeIdList(self, nodeIdList, score):                        
        assert self.hparams.tree_type == "nodeId"            
        tokIdList = []
        if self.hparams.max_beam_search:
            for nodeId in nodeIdList:
                max_tokId = self._get_max_tokId_from_tokIdList(list(self.nodeId2tokId[nodeId]), score)
                tokIdList.append(max_tokId)
            return list(set(tokIdList))
        else:                
            for nodeId in nodeIdList:
                tokIdList.extend(list(self.nodeId2tokId[nodeId]))                
            return list(set(tokIdList))
    def _get_tokIdList_from_groupIdList(self, groupIdList, score):
        assert self.hparams.tree_type == "groupId"
        tokIdList = []
        
        # for groupId Tree with max beam search
        if self.hparams.max_beam_search:
            for groupId in groupIdList:
                max_tokId = self._get_max_tokId_from_tokIdList(self.groupId2tokId[groupId], score)
                tokIdList.append(max_tokId)
            return list(set(tokIdList))
        # for normal groupId Tree
        else:
            for groupId in groupIdList:
                tokIdList.extend(self.groupId2tokId[groupId])
            return list(set(tokIdList))
   
    def _get_nodeId_from_tokId(self, tokId):
        nodeId = list(self.tokId2node/_teId[tokId])
        assert len(nodeId) == 1
        return nodeId[0] 
    
    def _get_groupId_from_tokId(self, tokId):
        return self.tokId2groupId[tokId]

    def _loss(self, batch):
        lm_labels = copy.deepcopy(batch["target_ids"])
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
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

    def get_list(self, batch_id, input_ids, score, trie_list):
        # starts with pad token & groupId for pad token is -1
        assert input_ids[0] == 0
        assert trie_list is not None and isinstance(trie_list, list)
        return self._get_from_trie(input_ids, trie_list[batch_id], score)
    

        """
    input_ids가 들어오면, 해당 tokId가 속한 groupId 찾고, 그걸 가지고 trie_dict 넘어간 다음
    해당 subtree의 key들(groupId) 를 모은 tokId return
    """
    def _get_from_trie(self, input_ids, trie_dict, score):
        #print(f"input_ids: {input_ids}")
        if self.hparams.tree_type == "groupId":
            if len(input_ids) == 0:
                possible_GroupList = list(trie_dict.keys())
                tokIdList = self._get_tokIdList_from_groupIdList(possible_GroupList, score)
                return tokIdList
            else:
                curGroupId = self._get_groupId_from_tokId(input_ids[0])
                if curGroupId in list(trie_dict.keys()):
                    return self._get_from_trie(input_ids[1:], trie_dict[curGroupId], score) 
                else:
                    return []
        elif self.hparams.tree_type == "nodeId":
            if input_ids[-1] == 1:
                return []
            else:
                NodeId = self._get_nodeId_from_tokId(input_ids[-1])
                next_nId_List = list(trie_dict[NodeId])
                tokIdList = self._get_tokIdList_from_nodeIdList(next_nId_List, score)
                return tokIdList
        else:
            raise NotImplementedError('tree type should be either groupId_tree or nodeId_tree!')

    def get(self, batch_id, input_ids, score, trie_dict=None):
        # starts with pad token & groupId for pad token is -1
        assert input_ids[0] == 0
        if trie_dict is None:
            trie_dict = self.trie
        return self._get_from_trie(input_ids, trie_dict, score)


    
    def get_title_context(self,input):
        if type(input) == list:
            title = []
            context = []
            for _input in input:
                _title,_context = self.get_title_context(_input)
                title.append(_title)
                context.append(_context)
            return title,context
        else:
            assert type(input)==str, f"{type(input)} not supported"
            _input = re.split("</s>",input)[0] ## pred has </s> but gt dosen't 
            _input+= "</s>" ## "<context>NONE</s>" ## 
            m = re.match("<title>(.*?)<context>(.*?)</s>",_input)
            if m:
                title = m.group(1).strip()
                context = m.group(2).strip()
            else:
                title = None
                context = None
            return title,context

    def calculate_scores(self, preds, gt_text, query, batch_idx):
        em_list = []
        title_em_list = []
        context_em_list = []
        recall_list = []
        f1_list = []
        for idx, (_query, _pred, _gt) in enumerate(zip(query, preds, gt_text)):
            pred_title, pred_context = self.get_title_context(_pred[0])
            gt_title,gt_context = self.get_title_context(_gt)
            _em = self._calculate_em(_pred[0], _gt)
            # print(pred_title,gt_title)
            _title_em = self._calculate_em(pred_title,gt_title)
            # print(pred_context,gt_context)
            _context_em = self._calculate_em(pred_context,gt_context)
            # _recall = self._calculate_recall(_pred, _gt)
            # _f1 = self._calculate_f1(_pred, _gt)
            if self.print and idx == 0:
                print(f"$" * 50)
                print(f"query: {_query}\npreds: {_pred}\ngt: {_gt}")
                print(f"pred_title: {pred_title} gt_title: {gt_title}")
                print(f"pred_context: {pred_context} gt_context: {gt_context}")
                print(f"em: {_em} // title_em: {_title_em} // context_em: {_context_em}")
                print(f"$" * 50)
                print(" ")
            em_list.append(_em)
            title_em_list.append(_title_em)
            context_em_list.append(_context_em)
            # recall_list.append(_recall)
            # f1_list.append(_f1)
        return em_list, recall_list, f1_list, title_em_list, context_em_list

    def _val_step(self, batch, batch_idx, return_elem=False):
        # calculates recall and em -> returns the list of each score
        _generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            #decoder_attention_mask=batch["target_mask"],
            max_length=self.hparams.max_output_length,
            num_beams=self.hparams.val_beam_size,
            num_return_sequences=self.hparams.val_beam_size,
            prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get(
                batch_id, sent.tolist(), scores
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
        generated_ids = [
            _generated_ids[
               i * self.hparams.val_beam_size : (i+1) * self.hparams.val_beam_size
            ].detach().cpu().numpy().tolist()
            for i in range(inum)
        ]
        output_list = [self.dev_input2output[_input] for _input in batch["input"]]
        em_list, recall_list, f1_list, title_em_list, context_em_list  = self.calculate_scores(
            generated_text, output_list, batch["input"], batch_idx
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
                "gt_tok": list(batch["target_ids"].detach().cpu().numpy().tolist()),
                "pred": list(generated_text),
                "pred_tok": list(generated_ids),
                "em": list(em_list),
                "recall": list(recall_list),
                "title_em":list(title_em_list),
                "context_em":list(context_em_list)
            }
        else:
            return em_list, recall_list, f1_list, title_em_list, context_em_list

    def _find_unique_path(self, seq, trie):
        _ids = seq.tolist()
        _ids = [int(elem) for elem in _ids]
        
        generated_ids = []
        if _ids[-1] == 0:
            for el in range(len(_ids)):
                if el != 0 and _ids[el] == 0:
                    continue
                else:
                    generated_ids.append(_ids[el])
        else:
            generated_ids = _ids 
        assert generated_ids[-1] != 0
        generated_ids = generated_ids[:-1]
        
        cur_dict = trie 
        for i in range(len(generated_ids) - 1):
            cur_dict = cur_dict[int(generated_ids[i])]
        if len(cur_dict[generated_ids[-1]]) == 0:
            add_list.append(generated_ids)
        else:
            cur_dict = cur_dict[int(generated_ids[-1])]
            # check if it is an unique path
            keys = list(cur_dict.keys())
            if len(keys) > 1:
                print(f"More than 1: {keys}")
                keys = [keys[0]]
            while len(cur_dict[keys[0]]) != 0:
                generated_ids.append(keys[0])
                cur_dict = cur_dict[keys[0]]
                keys = list(cur_dict.keys())
                if len(keys) > 1:
                    print(f"More than 1: {keys}")
                    keys = [keys[0]]
            generated_ids.append(1)
        return generated_ids

    def validation_step(self, batch, batch_idx):
        if self.hparams.save_model_only:
           self.convert_checkpoint("new_ckpt")
           print(f"Done saving checkpoint.. {self.current_epoch}")
           sys.exit()

        em_score, recall_score, f1_score, title_score, context_score = self._val_step(batch,batch_idx)
        self.em_score_list.extend(list(em_score))
        self.recall_score_list.extend(list(recall_score))
        self.f1_score_list.extend(list(f1_score))
        self.title_score_list.extend(list(title_score))
        self.context_score_list.extend(list(context_score))

    def validation_epoch_end(self, outputs):
        avg_em = np.mean(np.array(self.em_score_list))
        avg_recall = np.mean(np.array(self.recall_score_list))
        avg_f1 = np.mean(np.array(self.f1_score_list))
        avg_title_em = np.mean(np.array(self.title_score_list))
        avg_context_em = np.mean(np.array(self.context_score_list))

        self.em_score_list = []
        self.recall_score_list = []
        self.f1_score_list = []
        self.title_score_list = []
        self.context_score_list = []
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
            "val_title_em",
            avg_title_em,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_context_em",
            avg_context_em,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return

    def tokenize4next_hop(self, input_list, pred_list):
        sen_list = [f"{_input} {_pred[0]}" for _input, _pred in zip(input_list, pred_list)]
        tok_ret = self.tokenizer.batch_encode_plus(
                  sen_list,
                  max_length=self.hparams.max_input_length,
                  padding="max_length",
                  truncation=True,
                  return_tensors="pt"
              )      
        return tok_ret["input_ids"].to(self.device), tok_ret["attention_mask"].to(self.device), sen_list

    # ret_num => maximum iteration step
    def _test_step(self, batch, batch_idx, return_elem=False):
      
        test_source_ids = batch["source_ids"]
        test_source_masks = batch["source_mask"]
        test_target_masks = batch["target_mask"]
        test_input = batch["input"]
        _test_input = test_input
        test_output = [self.dev_input2output[el] for el in test_input]
        #test_output = batch["output"]
        test_pred = [[] for _ in range (len(batch["input"]))]
        iternum = np.array([len(el) for el in test_output]).max()
         
        retnum = 1
        for iter_idx in range(iternum):
            _generated_ids = self.model.generate(
                test_source_ids, 
                attention_mask=test_source_masks,
                use_cache=True,
                max_length=self.hparams.max_output_length,
                num_beams=self.hparams.val_beam_size,
                num_return_sequences=retnum,
                prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get(
                    batch_id, sent.tolist(), scores
                ),
                early_stopping=True,
            )
            _generated_text = self.ids_to_text(_generated_ids)
            inum = len(_generated_ids) // retnum 
            assert inum == len(test_output) 
            generated_text = [
                _generated_text[
                    i * retnum : (i + 1) * retnum 
                ]
                for i in range(inum)
            ]
            for idx, elem in enumerate(generated_text):
                test_pred[idx].append(elem[0])
            test_source_ids, test_source_masks, _test_input = self.tokenize4next_hop(_test_input, generated_text)

            
        title_em_list, em_list, title_f1_list, f1_list = self.calculate_test_scores(
            test_pred, test_output
        )

        print(f"="*80)
        print(test_pred[0])
        print(f"="*80)
        print(test_output[0])
        print(f"="*80)
        print(f"Title EM: {title_em_list[0]}\tEM: {em_list[0]}\tTitle F1: {title_f1_list[0]}\tF1: {f1_list[0]}")
        print(" ")

        if return_elem:
            assert (
                len(list(test_input))
                == len(list(generated_text))
                == len(list(em_list))
                == len(list(f1_list))
            )
            return {
                "input": list(test_input),
                "gt": list(test_output),
                "pred": list(test_pred),
                "em": list(em_list),
                "title_em": list(title_em_list),
                "f1": list(f1_list),
                "title_f1": list(title_f1_list),
            }
        else:
            return em_list, recall_list


    def calculate_test_scores(self, pred_list, gt_list):
        em_list = []; title_em_list = []; f1_list = []; title_f1_list = []
        for idx, (_pred_list, _gt_list) in enumerate(zip(pred_list, gt_list)):

            _pred_list = _pred_list[:len(_gt_list)]

            pred_title, pred_context = self.get_title_context(_pred_list)
            gt_title, gt_context = self.get_title_context(_gt_list)

            title_em_list.append(self._calculate_em(pred_title, gt_title))
            em_list.append(self._calculate_em(_pred_list, _gt_list))
            title_f1_list.append(self._calculate_f1(pred_title, gt_title))
            f1_list.append(self._calculate_f1(_pred_list, _gt_list))

        return title_em_list, em_list, title_f1_list, f1_list 



class T5_COT(T5_title_context):
    def __init__(self, args):
        super(T5_COT,self).__init__(args)
        print("but doing cot format")

    def get(self, batch_id, input_ids, score, trie_dict=None):
        # starts with pad token & groupId for pad token is -1
        assert input_ids[0] == 0
        if trie_dict is None:
            trie_dict = self.trie

        split_idx = 0
        for idx, id in enumerate(input_ids):
            if id == 2:
                split_idx = idx
        if split_idx!=0:
            # print('next hop!') 
            input_ids = [0] + input_ids[split_idx:]
        return self._get_from_trie(input_ids, trie_dict, score)

    """
    input_ids가 들어오면, 해당 tokId가 속한 groupId 찾고, 그걸 가지고 trie_dict 넘어간 다음
    해당 subtree의 key들(groupId) 를 모은 tokId return
    """
    def _get_from_trie(self, input_ids, trie_dict, score):
        #print(f"input_ids: {input_ids}")
        if self.hparams.tree_type == "groupId":
            if len(input_ids) == 0:
                possible_GroupList = list(trie_dict.keys())
                tokIdList = self._get_tokIdList_from_groupIdList(possible_GroupList, score)
                if 1 in tokIdList:
                    tokIdList.append(2)
                return tokIdList
            else:
                curGroupId = self._get_groupId_from_tokId(input_ids[0])
                if curGroupId in list(trie_dict.keys()):
                    return self._get_from_trie(input_ids[1:], trie_dict[curGroupId], score) 
                else:
                    return []
        elif self.hparams.tree_type == "nodeId":
            if input_ids[-1] == 1:
                return []
            else:
                NodeId = self._get_nodeId_from_tokId(input_ids[-1])
                next_nId_List = list(trie_dict[NodeId])
                tokIdList = self._get_tokIdList_from_nodeIdList(next_nId_List, score)
                return tokIdList
        elif self.hparams.tree_type == "clusterId":
            if len(input_ids) == 0:
                return list(trie_dict.keys()) 
            else:
                if input_ids[0] in list(trie_dict.keys()):
                    return self._get_from_trie(input_ids[1:], trie_dict[clusterId], score)
                else:
                    return []
        else:
            raise NotImplementedError('tree type should be either groupId_tree or nodeId_tree!')
        
    def _tok_f1_score(self, pred, gt):
        pred_toks = self.normalize_answer(pred).split()
        gt_toks = self.normalize_answer(gt).split()
        common = Counter(pred_toks) & Counter(gt_toks)
        #num_same = sum(common.values())
        num_same = len(common.keys())

        if num_same == 0: return 0

        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gt_toks)
        f1 = (2*precision*recall) / (precision+recall) * 100 
        return f1
    
    def _calculate_tok_f1(self, pred, gt):
        # assert type(pred) == list, type(gt) == list 
        if isinstance(gt,str):
            gt = [gt]
        _pred = pred[0]
        _f1_list = np.array([self._tok_f1_score(_pred, _gt) for _gt in gt])
        return _f1_list.mean()*100 

    def _calculate_em(self, pred, gt):
        if pred is None: 
            return 0
        if type(gt) == list:
            gt = [self.normalize_answer(el) for el in gt]
            preds = [self.normalize_answer(el) for el in pred]
            if len(gt)!=len(preds):
                return 0
            else:
                for g,p in zip(gt,preds):
                    if g!=p:
                        return 0
                return 100
        else:
            if self.normalize_answer(pred) == self.normalize_answer(gt):
                return 100
            else:
                return 0
            
    def get_title_context(self,input):
        if type(input) == list:
            title = []
            context = []
            for _input in input:
                _title,_context = self.get_title_context(_input)
                if _title!=None:
                    title.extend(_title)
                if _context!=None:
                    context.extend(_context)
            return title,context
        else:
            assert type(input)==str, f"{type(input)} not supported"
            _input = re.split("</s>",input)[0] ## pred has </s> but gt dosen't 
            _input+= "<title>" ## "<context>NONE</s>" ## 
            m1 = re.findall("<title>(.*?)<context>",_input)
            m2 = re.findall("<context>(.*?)<title>",_input)
            if m1:
                title = [m.strip() for m in m1]
                context = [m.strip() for m in m2]
            else:
                title = None
                context = None
            return title,context

    def calculate_scores(self, preds, gt_text, query, batch_idx):
        em_list = []; title_em_list = []; context_em_list = []
        recall_list = []
        f1_list = []; title_f1_list = []
        for idx, (_query, _pred, _gt) in enumerate(zip(query, preds, gt_text)):
            pred_title, pred_context = self.get_title_context(_pred[0]) ## list , list
            gt_title, gt_context = self.get_title_context(_gt) ## list , list 
            # _em = self._calculate_em(_pred[0], _gt)
            # print(pred_title,gt_title)
            _title_em = self._calculate_em(pred_title, gt_title)
            # print(pred_context,gt_context)
            _context_em = self._calculate_em(pred_context, gt_context)
            _em = _title_em*_context_em/100
            _f1 = self._calculate_f1(_pred, _gt)
            _title_f1 = self._calculate_f1(pred_title, gt_title)

            if self.print and idx == 0:
                print(f"$" * 50)
                print(f"query: {_query}\npreds: {_pred}\ngt: {_gt}")
                print(f"pred_title: {pred_title} gt_title: {gt_title}")
                print(f"pred_context: {pred_context} gt_context: {gt_context}")
                print(f"em: {_em} // title_em: {_title_em} // context_em: {_context_em} // _f1: {_f1}")
                print(f"$" * 50)
                print(" ")
            em_list.append(_em)
            title_em_list.append(_title_em)
            context_em_list.append(_context_em)
            # recall_list.append(_recall)
            f1_list.append(_f1)
            title_f1_list.append(_title_f1)
        return em_list, recall_list, f1_list, title_f1_list, title_em_list, context_em_list

    def _val_step(self, batch, batch_idx, return_elem=False):
        # calculates recall and em -> returns the list of each score
        _generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            #decoder_attention_mask=batch["target_mask"],
            max_length=self.hparams.max_output_length,
            num_beams=self.hparams.val_beam_size,
            num_return_sequences=self.hparams.val_beam_size,
            prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get(
                batch_id, sent.tolist(), scores
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
        generated_ids = [
            _generated_ids[
               i * self.hparams.val_beam_size : (i+1) * self.hparams.val_beam_size
            ].detach().cpu().numpy().tolist()
            for i in range(inum)
        ]
        output_list = [self.dev_input2output[_input] for _input in batch["input"]]
        em_list, recall_list, f1_list, title_f1_list, title_em_list, context_em_list  = self.calculate_scores(
            generated_text, output_list, batch["input"], batch_idx
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
                "gt_tok": list(batch["target_ids"].detach().cpu().numpy().tolist()),
                "pred": list(generated_text),
                "pred_tok": list(generated_ids),
                "em": list(em_list),
                "recall": list(recall_list),
                "title_em":list(title_em_list),
                "context_em":list(context_em_list),
                "f1":list(f1_list),
                "title_f1": list(title_f1_list)
            }
        else:
            return em_list, recall_list, f1_list, title_em_list, context_em_list
        
    def validation_step(self, batch, batch_idx):
        if self.hparams.save_model_only:
           self.convert_checkpoint("new_ckpt")
           print(f"Done saving checkpoint.. {self.current_epoch}")
           sys.exit()

        em_score, recall_score, f1_score, title_score, context_score = self._val_step(batch,batch_idx)
        self.em_score_list.extend(list(em_score))
        self.recall_score_list.extend(list(recall_score))
        self.f1_score_list.extend(list(f1_score))
        self.title_score_list.extend(list(title_score))
        self.context_score_list.extend(list(context_score))

    def validation_epoch_end(self, outputs):
        avg_em = np.mean(np.array(self.em_score_list))
        avg_title_em = np.mean(np.array(self.title_score_list))
        avg_context_em = np.mean(np.array(self.context_score_list))
        avg_recall = np.mean(np.array(self.recall_score_list))
        avg_f1 = np.mean(np.array(self.f1_score_list))

        self.em_score_list = []
        self.title_score_list = []
        self.context_score_list = []
        self.recall_score_list = []
        self.f1_score_list = []

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
            "val_title_em",
            avg_title_em,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_context_em",
            avg_context_em,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_f1",
            avg_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return
       
    def test_step(self, batch, batch_idx):
        ret_dict = self._val_step(batch, batch_idx, return_elem=True)
        self.test_input_list.extend(ret_dict["input"])
        self.test_gt_list.extend(ret_dict["gt"])
        self.test_pred_list.extend(ret_dict["pred"])
        self.em_score_list.extend(ret_dict["em"])
        self.title_em_score_list.extend(ret_dict["title_em"])
        self.f1_score_list.extend(ret_dict["f1"])
        self.title_f1_score_list.extend(ret_dict["title_f1"])

   

class T5MultiHop(T5grTuner):
    def __init__(self, args):
        super(T5MultiHop, self).__init__(args)
        # If in training mode, load ckpt for training
        if self.hparams.do_train:
            raise NotImplementedError("Training step is not implemented Yet")
            if self.hparams.train_c_emb:
                self.model = contextualized_T5.from_pretrained(
                    self.hparams.model_name_or_path, config=self.config, ignore_mismatched_sizes=True
                )
            else:
                self.model = contextualized_T5.from_pretrained(
                    self.hparams.model_name_or_path, config=self.config
                )
            if self.hparams.gr_decoder_only_encoder_ckpt is not None:
                print(f'===== Loading encoder ckpt from.. {self.hparams.gr_decoder_only_encoder_ckpt}')
                m = torch.load(os.path.join(self.hparams.gr_decoder_only_encoder_ckpt, "pytorch_model.bin"))
                model_dict = self.model.state_dict()
                for k in m.keys():
                    if 'decoder.embed_tokens' in k:
                        continue
                    if k in model_dict:
                        pname = k
                        pval = m[k]
                        model_dict[pname] = pval.clone().to(model_dict[pname].device)
                self.model.load_state_dict(model_dict, strict=False)
            else:
                print(f'===== Encoder ckpt is same as Decoder ckpt')
            if self.hparams.gr_decoder_only:
                for n, p in self.model.get_encoder().named_parameters():
                    p.requires_grad = False
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.hparams.tokenizer_name_or_path
            )
            if self.hparams.periflow:
                self.connect_str, self.container_name = get_blob_info()
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)
            if self.print:
                print(f"@@@ Loading Model from {self.hparams.model_name_or_path}")
                print(f'@@@ Loading decoder embedding: {self.hparams.embedding_model}')
            self.em_score_list = []
            self.recall_score_list = []
            self.dev_input2output = json.load(open(os.path.join(self.hparams.dataset, self.hparams.dev_input2output), "r"))

        # If in testing mode, load ckpt for inference
        if self.hparams.do_test:
            self.model = contextualized_T5.from_pretrained(
                self.hparams.test_model_path, config=self.config, ignore_mismatched_sizes=True
            )
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.test_model_path)
            if self.print:
                print(f"@@@ Loading Model from {self.hparams.test_model_path}")
            
            self.test_save_name = os.path.join(self.hparams.output_dir, f"{self.hparams.test_name}_{self.hparams.tree_type}_mbs_{self.hparams.max_beam_search}_result_beam{self.hparams.val_beam_size}.json")              
            title2context = json.load(open("/data/project/rw/contextualized_GENRE/dataset/hotpot/title2context.json"))
            tok_title2context = json.load(open("/data/project/rw/contextualized_GENRE/dataset/hotpot/tok_title2context.json"))
            self.title2context = {**title2context, **tok_title2context}
            

            if os.path.exists(self.test_save_name):
                 prev_f = json.load(open(self.test_save_name))
                 print(f"@@@ Loading Previous file!! => #: {len(prev_f['input'])}")
                 self.test_input_list = prev_f['input']
                 self.test_pred_list = prev_f['pred']
            else:
                 print(f'@@@ Initialize Test!!')
                 self.test_input_list = []
                 self.test_gt_list = []
                 self.test_gt_tok_list = []
                 self.test_pred_list = []
                 self.test_pred_tok_list = []
                 self.test_em_score_list = []
                 self.test_recall_score_list = []

        if self.hparams.freeze_encoder:
            if self.print:
                print(f"@@@ Freeze Encoder!")
                encoder = self.model.get_encoder()
                for n, p in encoder.named_parameters():
                    p.requires_grad=False
        #### Tokenizer for generation step!
        self.dec_tok = AutoTokenizer.from_pretrained(
            self.hparams.embedding_model
        )
        
        if self.hparams.tree_type == "nodeId":
            nodeId_sup = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.nodeId_sup), 'rb'))
            self.nodeId2groupdId = nodeId_sup['group_set']
            self.nodeId2tokId = nodeId_sup['token_set']
            self.groupId2nodeId = nodeId_sup['inv_group_set']
            self.tokId2nodeId = nodeId_sup['inv_token_set']
        self.cnt_over = 0
        self.len_test_dataset = len(self.test_dataloader())

    def _get_dataset(self, split):
        dataset = GENREDataset(
            tokenizer=self.tokenizer,
            split=split,
            hparams=self.hparams,
            tokid2emb=self.contextualized_tokid2emb 
        )
        return dataset
    def _get_max_tokId_from_tokIdList(self, tokIdList, score):
        tokIdList = sorted(tokIdList)
        idx = score[tokIdList].detach().cpu().numpy().argmax()
        max_tokId = tokIdList[idx]
        return max_tokId
    def _get_tokIdList_from_nodeIdList(self, nodeIdList, score):                        
        assert self.hparams.tree_type == "nodeId"            
        tokIdList = []
        if self.hparams.max_beam_search:
            for nodeId in nodeIdList:
                max_tokId = self._get_max_tokId_from_tokIdList(list(self.nodeId2tokId[nodeId]), score)
                tokIdList.append(max_tokId)
            return list(set(tokIdList))
        else:                
            for nodeId in nodeIdList:
                tokIdList.extend(list(self.nodeId2tokId[nodeId]))                
            return list(set(tokIdList))

    def _get_tokIdList_from_groupIdList(self, groupIdList, score):
        assert self.hparams.tree_type == "groupId"
        tokIdList = []
        
        # for groupId Tree with max beam search
        if self.hparams.max_beam_search:
            for groupId in groupIdList:
                max_tokId = self._get_max_tokId_from_tokIdList(self.groupId2tokId[groupId], score)
                tokIdList.append(max_tokId)
            return list(set(tokIdList))
        # for normal groupId Tree
        else:
            for groupId in groupIdList:
                tokIdList.extend(self.groupId2tokId[groupId])
            return list(set(tokIdList))
   
    def _get_nodeId_from_tokId(self, tokId):
        nodeId = list(self.tokId2node/_teId[tokId])
        assert len(nodeId) == 1
        return nodeId[0] 
    
    def _get_groupId_from_tokId(self, tokId):
        return self.tokId2groupId[tokId]


    def _loss(self, batch):
        lm_labels = copy.deepcopy(batch["target_ids"])
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
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
    def get_list(self, batch_id, input_ids, score, trie_list):
        # starts with pad token & groupId for pad token is -1
        assert input_ids[0] == 0
        assert trie_list is not None and isinstance(trie_list, list)
        return self._get_from_trie(input_ids, trie_list[batch_id], score)
    
    def get(self, batch_id, input_ids, score, trie_dict=None):
        # starts with pad token & groupId for pad token is -1
        assert input_ids[0] == 0
        if trie_dict is None:
            trie_dict = self.trie
        return self._get_from_trie(input_ids, trie_dict, score)

    """
    input_ids가 들어오면, 해당 tokId가 속한 groupId 찾고, 그걸 가지고 trie_dict 넘어간 다음
    해당 subtree의 key들(groupId) 를 모은 tokId return
    """
    def _get_from_trie(self, input_ids, trie_dict, score):
        #print(f"input_ids: {input_ids}")
        if self.hparams.tree_type == "groupId":
            if len(input_ids) == 0:
                possible_GroupList = list(trie_dict.keys())
                tokIdList = self._get_tokIdList_from_groupIdList(possible_GroupList, score)
                return tokIdList
            else:
                curGroupId = self._get_groupId_from_tokId(input_ids[0])
                if curGroupId in list(trie_dict.keys()):
                    return self._get_from_trie(input_ids[1:], trie_dict[curGroupId], score) 
                else:
                    return []
        elif self.hparams.tree_type == "nodeId":
            if input_ids[-1] == 1:
                return []
            else:
                NodeId = self._get_nodeId_from_tokId(input_ids[-1])
                next_nId_List = list(trie_dict[NodeId])
                tokIdList = self._get_tokIdList_from_nodeIdList(next_nId_List, score)
                return tokIdList
        elif self.hparams.tree_type == "clusterId":
            if len(input_ids) == 0:
                return list(trie_dict.keys()) 
            else:
                if input_ids[0] in list(trie_dict.keys()):
                    return self._get_from_trie(input_ids[1:], trie_dict[clusterId], score)
                else:
                    return []
        else:
            raise NotImplementedError('tree type should be either groupId_tree or nodeId_tree!')

    def calculate_scores(self, preds, gt_text, query, batch_idx):
        em_list = []
        recall_list = []
        for idx, (_query, _pred, _gt) in enumerate(zip(query, preds, gt_text)):
            _em = self._calculate_em(_pred[0], _gt)
            _recall = self._calculate_recall(_pred, _gt)
            """
            if self.print and idx == 0:
                print(f"$" * 50)
                print(f"query: {_query}\npreds: {_pred}\ngt: {_gt}")
                print(f"em: {_em} // recall: {_recall}")
                print(f"$" * 50)
                print(" ")
            """
            em_list.append(_em)
            recall_list.append(_recall)
        return em_list, recall_list

    def _val_step(self, batch, batch_idx, return_elem=False):
        # calculates recall and em -> returns the list of each score
        _generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            #decoder_attention_mask=batch["target_mask"],
            max_length=self.hparams.max_output_length,
            num_beams=self.hparams.val_beam_size,
            num_return_sequences=self.hparams.val_beam_size,
            prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get(
                batch_id, sent.tolist(), scores
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
        generated_ids = [
            _generated_ids[
               i * self.hparams.val_beam_size : (i+1) * self.hparams.val_beam_size
            ].detach().cpu().numpy().tolist()
            for i in range(inum)
        ]
        output_list = [self.dev_input2output[_input] for _input in batch["input"]]
        em_list, recall_list = self.calculate_scores(
            generated_text, output_list, batch["input"], batch_idx
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
                "gt_tok": list(batch["target_ids"].detach().cpu().numpy().tolist()),
                "pred": list(generated_text),
                "pred_tok": list(generated_ids),
                "em": list(em_list),
                "recall": list(recall_list),
            }
        else:
            return em_list, recall_list

    def _find_unique_path(self, seq, trie):
        _ids = seq.tolist()
        _ids = [int(elem) for elem in _ids]
        
        generated_ids = []
        if _ids[-1] == 0:
            for el in range(len(_ids)):
                if el != 0 and _ids[el] == 0:
                    continue
                else:
                    generated_ids.append(_ids[el])
        else:
            generated_ids = _ids 
        assert generated_ids[-1] != 0
        generated_ids = generated_ids[:-1]
        
        cur_dict = trie 
        for i in range(len(generated_ids) - 1):
            cur_dict = cur_dict[int(generated_ids[i])]
        if len(cur_dict[generated_ids[-1]]) == 0:
            add_list.append(generated_ids)
        else:
            cur_dict = cur_dict[int(generated_ids[-1])]
            # check if it is an unique path
            keys = list(cur_dict.keys())
            if len(keys) > 1:
                print(f"More than 1: {keys}")
                keys = [keys[0]]
            while len(cur_dict[keys[0]]) != 0:
                generated_ids.append(keys[0])
                cur_dict = cur_dict[keys[0]]
                keys = list(cur_dict.keys())
                if len(keys) > 1:
                    print(f"More than 1: {keys}")
                    keys = [keys[0]]
            generated_ids.append(1)
        return generated_ids

    def tokenize4sec_hop(self, input, first_hop):
         
        sen_list = []
        for elem in first_hop:
            try:
               context = self.title2context[elem]
            except:
               print(f'Missing {elem}')
            sen_list.append(f"{input} <P1> {elem}: {context}")

        #sen_list = [f"{input} <P1> {elem}" for elem in first_hop]
        return self.tokenizer.batch_encode_plus(
            sen_list,
            max_length=self.hparams.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )


    def _test_step(self, batch, batch_idx, return_elem=False):
       
        # for case where it resume test 
        test_input = []
        for _input in batch['input']:
            if _input in self.test_input_list: continue
            else: test_input.append(_input) 
        test_num = len(test_input)
        start_num = len(batch['input'])-test_num
        if test_num == 0: return None
        test_input = batch['input'][start_num:]
        test_output = batch['output'][start_num:]
        test_source_ids = batch['source_ids'][start_num:]
        test_source_masks = batch['source_mask'][start_num:]
        test_target_ids = batch['target_ids'][start_num:]
        test_target_masks = batch['target_mask'][start_num:]
        assert len(test_output) == test_num, f'test_output: {len(test_output)}\ttest_num: {test_num}'
        _trie_list = [copy.deepcopy(self.trie) for _ in range(test_num)]
     
        
        for hop in range(2):
           
           if hop == 0:
              unique_pred_list = [[] for _ in range(test_num)] 
              unique_ids_list = [[] for _ in range(test_num)] 
              over = [0]*test_num
              
              for _iter in range(self.hparams.ret_num*2):
                  print("="*80)
                  print(f"iter: {_iter} // DONE: {over.count(1)} / {test_num}")
                  print(unique_pred_list)
                  print("="*80)
                  _generated_ids = self.model.generate(
                      test_source_ids, 
                      attention_mask=test_source_masks,
                      use_cache=True,
                      decoder_attention_mask=test_target_masks,
                      max_length=self.hparams.max_output_length,
                      num_beams=self.hparams.val_beam_size,
                      num_return_sequences=self.hparams.val_beam_size,
                      prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get_list(
                          batch_id, sent.tolist(), scores, trie_list=_trie_list
                      ),
                      early_stopping=True,
                  )
                  _generated_text = self.ids_to_text(_generated_ids)
                  inum = len(_generated_ids) // self.hparams.val_beam_size
                  assert inum == len(test_output) 
                  generated_text = [
                      _generated_text[
                          i * self.hparams.val_beam_size : (i + 1) * self.hparams.val_beam_size
                      ]
                      for i in range(inum)
                  ]
                  generated_ids = [
                      _generated_ids[
                          i * self.hparams.val_beam_size : (i+1) * self.hparams.val_beam_size
                      ].detach().cpu().numpy().tolist()
                      for i in range(inum)
                  ]
                  """ 
                  for b_texts, b_ids in zip(generated_text, generated_ids):
                      for _text, _ids in zip(b_texts, b_ids):
                          g_ids = [self.tokId2groupId[el] for el in _ids]
                  """
                  print(f"prediction: {generated_text}")
                  print("*"*80) 
                  for bid, (batch_text, batch_ids) in enumerate(zip(generated_text, generated_ids)):
                      if over[bid] == 1: continue
                      ### iterate over batch (val_beam_size)
                      _upper_ids = []
                      _trie_dict = _trie_list[bid]
                      for _text, _ids in zip(batch_text, batch_ids):
                          if _ids in unique_ids_list[bid]:
                              continue
                          else:
                              if _text in unique_pred_list[bid]:
                                  assert not self.hparams.tree_type == "nodeId"
                                  _upper_ids.append([self.tokId2groupId[el] for el in _ids]) 
                              elif len(unique_pred_list[bid]) < self.hparams.ret_num:
                                  unique_pred_list[bid].append(_text)
                                  unique_ids_list[bid].append(_ids)
                                  if self.hparams.tree_type == "groupId":
                                      _upper_ids.append([self.tokId2groupId[el] for el in _ids]) 
                                  elif self.hparams.tree_type == "nodeId":
                                      temp = []
                                      eos_pos = (np.array(_ids) == 1).nonzero()[0][0]
                                      for el in _ids[:eos_pos]:
                                          assert len(self.tokId2nodeId[el]) == 1, self.tokId2nodeId[el]
                                          temp.append(list(self.tokId2nodeId[el])[0])
                                      # find the end token
                                      cur_nId = temp[-1]
                                      next_nId = _trie_dict[cur_nId]
                                      end_nId = list(next_nId.intersection(self.tokId2nodeId[1]))
                                      assert len(end_nId) == 1
                                      temp.append(end_nId[0])
                                      _upper_ids.append(temp)
                                  else:
                                      assert False
                              else:
                                  pass
                      # remove from _trie_dict
                      _trie_dict = self._remove_prev_from_trie(_trie_dict, _upper_ids)
                      _trie_list[bid] = _trie_dict
                      if len(unique_pred_list[bid]) >= self.hparams.ret_num:
                          over[bid] = 1
                  if over.count(1) == test_num:
                      break
                  """
                  if (over.count(1) == test_num) or (self.cnt_over == self.len_test_dataset):
                      self.cnt_over += over.count(1)
                      break
                  """
                 
              for i in range(len(unique_pred_list)):
                  unique_pred_list[i] = unique_pred_list[i][:self.hparams.ret_num]
                  unique_ids_list[i] = unique_ids_list[i][:self.hparams.ret_num]
       
              print(unique_pred_list)
              print(unique_ids_list)
 
           if hop == 1:
              #_trie_list = [copy.deepcopy(self.trie) for _ in range(test_num)]
              # tokenize 
              output_ret = []
              for _input, elem, tok_ids in zip(test_input, unique_pred_list, unique_ids_list[0]):
                  group_ids = [self.tokId2groupId[el] for el in tok_ids]
                  tok_ret_list = self.tokenize4sec_hop(_input, elem)
                  #_trie_dict = self._remove_prev_from_trie(self.trie, [group_ids])
                  
                  _generated_ids = self.model.generate(
                           tok_ret_list["input_ids"].to(self.device),
                           attention_mask=tok_ret_list["attention_mask"].to(self.device),
                           use_cache=True,
                           max_length=self.hparams.max_output_length,
                           num_beams=10,
                           num_return_sequences=1,
                           prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get(
                                 batch_id, sent.tolist(), scores#, _trie_dict
                           ),
                           early_stopping=True
                        )
                  _generated_text = self.ids_to_text(_generated_ids)
                  output_ret.append([[f, s] for (f, s) in zip(elem, _generated_text)])

              if self.print: print(output_ret)

        if return_elem:
            assert (
                len(list(test_input))
                == len(list(generated_text))
                == len(list(output_ret))
            )
            return {
                "input": list(test_input),
                "pred": list(output_ret),
            }
        else:
            return em_list, recall_list



class T5TotalTuner(T5grTuner):
    def __init__(self, args):
        super(T5TotalTuner, self).__init__(args)

        # For clusterId
        self.pad_token_id = 71929
        self.eos_token_id = 90223
        self.config.update({"decoder_start_token_id": self.pad_token_id})
        self.config.update({"pad_token_id": self.pad_token_id})

        # If in training mode, load ckpt for training
        if self.hparams.do_train:
            if self.hparams.train_c_emb:
                self.model = contextualized_T5.from_pretrained(
                    self.hparams.model_name_or_path, config=self.config, ignore_mismatched_sizes=True
                )
            else:
                self.model = contextualized_T5.from_pretrained(
                    self.hparams.model_name_or_path, config=self.config
                )
            if self.hparams.gr_decoder_only_encoder_ckpt is not None:
                print(f'===== Loading encoder ckpt from.. {self.hparams.gr_decoder_only_encoder_ckpt}')
                m = torch.load(os.path.join(self.hparams.gr_decoder_only_encoder_ckpt, "pytorch_model.bin"))
                model_dict = self.model.state_dict()
                for k in m.keys():
                    if 'decoder.embed_tokens' in k:
                        continue
                    if k in model_dict:
                        pname = k
                        pval = m[k]
                        model_dict[pname] = pval.clone().to(model_dict[pname].device)
                self.model.load_state_dict(model_dict, strict=False)
            else:
                print(f'===== Encoder ckpt is same as Decoder ckpt')
            if self.hparams.gr_decoder_only:
                for n, p in self.model.get_encoder().named_parameters():
                    p.requires_grad = False
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.hparams.tokenizer_name_or_path
            )
            if self.hparams.periflow:
                self.connect_str, self.container_name = get_blob_info()
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)
            if self.print:
                print(f"@@@ Loading Model from {self.hparams.model_name_or_path}")
                print(f'@@@ Loading decoder embedding: {self.hparams.embedding_model}')
            self.em_score_list = []
            self.recall_score_list = []

        # If in testing mode, load ckpt for inference
        if self.hparams.do_test:
            self.model = contextualized_T5.from_pretrained(
                self.hparams.test_model_path, config=self.config, ignore_mismatched_sizes=True
            )
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.test_model_path)
            if self.print:
                print(f"@@@ Loading Model from {self.hparams.test_model_path}")
            
            self.test_save_name = os.path.join(self.hparams.output_dir, f"{self.hparams.test_name}_{self.hparams.tree_type}_mbs_{self.hparams.max_beam_search}_result_beam{self.hparams.val_beam_size}.json")               
            if os.path.exists(self.test_save_name):
                 prev_f = json.load(open(self.test_save_name))
                 print(f"@@@ Loading Previous file!! => #: {len(prev_f['input'])}")
                 self.test_input_list = prev_f['input']
                 self.test_gt_list = prev_f['gt']
                 self.test_gt_tok_list = prev_f['gt_tok']
                 self.test_pred_tok_list = prev_f['pred_tok']
            else:
                 print(f'@@@ Initialize Test!!')
                 self.test_input_list = []
                 self.test_gt_list = []
                 self.test_gt_tok_list = []
                 self.test_pred_list = []
                 self.test_pred_tok_list = []
                 self.test_em_score_list = []
                 self.test_recall_score_list = []
            self.em_score_list = []
            self.recall_score_list = []
            self.clusterIdList2corpusList = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.clusterIdList2corpusList), "rb"))
            self.clusterIdList = list(self.clusterIdList2corpusList.keys()) 

        if self.hparams.freeze_encoder:
            if self.print:
                print(f"@@@ Freeze Encoder!")
                encoder = self.model.get_encoder()
                for n, p in encoder.named_parameters():
                    p.requires_grad=False
        #### Tokenizer for generation step!
        self.dec_tok = AutoTokenizer.from_pretrained(
            self.hparams.embedding_model
        )
        
        if self.hparams.tree_type == "nodeId":
            nodeId_sup = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.nodeId_sup), 'rb'))
            self.nodeId2groupdId = nodeId_sup['group_set']
            self.nodeId2tokId = nodeId_sup['token_set']
            self.groupId2nodeId = nodeId_sup['inv_group_set']
            self.tokId2nodeId = nodeId_sup['inv_token_set']
        self.cnt_over = 0
        self.len_test_dataset = len(self.test_dataloader())
        self.dev_input2output = json.load(open(os.path.join(self.hparams.dataset, self.hparams.dev_input2output), "r"))


    def _get_dataset(self, split):
        dataset = GENREDataset(
            tokenizer=self.tokenizer,
            split=split,
            hparams=self.hparams,
            tokid2emb=self.contextualized_tokid2emb 
        )
        return dataset
    def _get_max_tokId_from_tokIdList(self, tokIdList, score):
        tokIdList = sorted(tokIdList)
        idx = score[tokIdList].detach().cpu().numpy().argmax()
        max_tokId = tokIdList[idx]
        return max_tokId
    def _get_tokIdList_from_nodeIdList(self, nodeIdList, score):                        
        assert self.hparams.tree_type == "nodeId"            
        tokIdList = []
        if self.hparams.max_beam_search:
            for nodeId in nodeIdList:
                max_tokId = self._get_max_tokId_from_tokIdList(list(self.nodeId2tokId[nodeId]), score)
                tokIdList.append(max_tokId)
            return list(set(tokIdList))
        else:                
            for nodeId in nodeIdList:
                tokIdList.extend(list(self.nodeId2tokId[nodeId]))                
            return list(set(tokIdList))

    def _get_tokIdList_from_groupIdList(self, groupIdList, score):
        assert self.hparams.tree_type == "groupId"
        tokIdList = []
        
        # for groupId Tree with max beam search
        if self.hparams.max_beam_search:
            for groupId in groupIdList:
                max_tokId = self._get_max_tokId_from_tokIdList(self.groupId2tokId[groupId], score)
                tokIdList.append(max_tokId)
            return list(set(tokIdList))
        # for normal groupId Tree
        else:
            for groupId in groupIdList:
                tokIdList.extend(self.groupId2tokId[groupId])
            return list(set(tokIdList))
   
    def _get_nodeId_from_tokId(self, tokId):
        nodeId = list(self.tokId2node/_teId[tokId])
        assert len(nodeId) == 1
        return nodeId[0] 
    
    def _get_groupId_from_tokId(self, tokId):
        return self.tokId2groupId[tokId]


    def _loss(self, batch):
        lm_labels = copy.deepcopy(batch["target_ids"])
        #lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        lm_labels[lm_labels[:, :] == self.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
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
    def get_list(self, batch_id, input_ids, score, trie_list):
        # starts with pad token & groupId for pad token is -1
        #assert input_ids[0] == 0
        assert input_ids[0] == self.pad_token_id
        assert trie_list is not None and isinstance(trie_list, list)
        return self._get_from_trie(input_ids, trie_list[batch_id], score)
    
    def get(self, batch_id, input_ids, score, trie_dict=None):
        # starts with pad token & groupId for pad token is -1
        #assert input_ids[0] == 0 
        assert input_ids[0] == self.pad_token_id
        if trie_dict is None:
            trie_dict = self.trie
        return self._get_from_trie(input_ids, trie_dict, score)

    """
    input_ids가 들어오면, 해당 tokId가 속한 groupId 찾고, 그걸 가지고 trie_dict 넘어간 다음
    해당 subtree의 key들(groupId) 를 모은 tokId return
    """
    def _get_from_trie(self, input_ids, trie_dict, score):
        #print(f"input_ids: {input_ids}")
        if self.hparams.tree_type == "groupId":
            if len(input_ids) == 0:
                possible_GroupList = list(trie_dict.keys())
                tokIdList = self._get_tokIdList_from_groupIdList(possible_GroupList, score)
                return tokIdList
            else:
                curGroupId = self._get_groupId_from_tokId(input_ids[0])
                if curGroupId in list(trie_dict.keys()):
                    return self._get_from_trie(input_ids[1:], trie_dict[curGroupId], score) 
                else:
                    return []
        elif self.hparams.tree_type == "nodeId":
            if input_ids[-1] == 1:
                return []
            else:
                NodeId = self._get_nodeId_from_tokId(input_ids[-1])
                next_nId_List = list(trie_dict[NodeId])
                tokIdList = self._get_tokIdList_from_nodeIdList(next_nId_List, score)
                return tokIdList
        elif self.hparams.tree_type == "clusterId":
            if len(input_ids) == 0:
                return list(trie_dict.keys()) 
            else:
                if input_ids[0] in list(trie_dict.keys()):
                    return self._get_from_trie(input_ids[1:], trie_dict[input_ids[0]], score)
                else:
                    return []
        else:
            raise NotImplementedError('tree type should be either groupId_tree or nodeId_tree!')

    def calculate_scores(self, preds, gt_text, query, batch_idx):
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

    def remove_padding(self, ids, pad_token_id, eos_token_id, is_gt=True):
        ids = torch.tensor(ids)
        if is_gt == True:
            pad_idx = (ids == pad_token_id).nonzero()[0,0]
            ids = ids[:pad_idx].tolist()
            ids = [pad_token_id] + ids
        else:
            pad_idx = (ids == eos_token_id).nonzero()
            #assert len(pad_idx) > 0, ids
            
            if len(pad_idx) == 0:
                print("!!! len(pred) > max_output_lenth !!!", ids)
                ids = ids[1:].tolist()
                #ids = ids + [eos_token_id]
            else:
                pad_idx = (ids == eos_token_id).nonzero()[0,0]
                ids = ids[1:pad_idx+1].tolist()
                #ids = ids + [eos_token_id]
        return ids

    def _ids_calculate_recall(self, pred, gt, pad_token_id, eos_token_id):
        assert len(pred) == self.hparams.val_beam_size, f'gt: {gt}\npred: {pred}'
        #gt = self.remove_padding(gt, pad_token_id, eos_token_id, is_gt=True)
        if type(gt) == list:
            for elem in pred:
                if elem is None: continue 
                elem = self.remove_padding(elem, pad_token_id, eos_token_id, is_gt=False)
                if elem in gt:
                    return 100
        else:
            for elem in pred:
                if elem is None: continue 
                elem = self.remove_padding(elem, pad_token_id, eos_token_id, is_gt=False)
                if elem == gt:
                    return 100
        return 0
    
    def _ids_calculate_em(self, pred, gt, pad_token_id, eos_token_id):
        #gt = self.remove_padding(gt, pad_token_id, eos_token_id, is_gt=True)
        pred = self.remove_padding(pred, pad_token_id, eos_token_id, is_gt=False)
        if type(gt) == list:
            if pred in gt:
                return 100
        else:
            if pred == gt:
                return 100
        return 0

    def ids_calculate_scores(self, preds_ids, gt_ids, query, batch_idx):
        em_list = []
        recall_list = []
        for idx, (_query, _pred, _gt) in enumerate(zip(query, preds_ids, gt_ids)):
            _em = self._ids_calculate_em(_pred[0], _gt, self.pad_token_id, self.eos_token_id)
            _recall = self._ids_calculate_recall(_pred, _gt, self.pad_token_id, self.eos_token_id)
            if self.print and idx == 0:
                print(f"$" * 50)
                print(f"query: {_query}\npreds: {_pred}\ngt: {_gt}")
                print(f"em: {_em} // recall: {_recall}")
                print(f"$" * 50)
                print(" ")
            em_list.append(_em)
            recall_list.append(_recall)
        return em_list, recall_list

    def _val_step(self, batch, batch_idx, return_elem=False):
        # calculates recall and em -> returns the list of each score
        _generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            #decoder_attention_mask=batch["target_mask"],
            max_length=self.hparams.max_output_length,
            num_beams=self.hparams.val_beam_size,
            num_return_sequences=self.hparams.val_beam_size,
            prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get(
                batch_id, sent.tolist(), scores
            ),
            early_stopping=True,
            decoder_start_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        #_generated_text = self.ids_to_text(_generated_ids)
        inum = len(_generated_ids) // self.hparams.val_beam_size
        assert inum == len(batch["output"])
        
        #generated_text = [
        #    _generated_text[
        #        i * self.hparams.val_beam_size : (i + 1) * self.hparams.val_beam_size
        #    ]
        #    for i in range(inum)
        #]
        generated_ids = [
            _generated_ids[
               i * self.hparams.val_beam_size : (i+1) * self.hparams.val_beam_size
            ].detach().cpu().numpy().tolist()
            for i in range(inum)
        ]
        output_list = [self.dev_input2output[_input] for _input in batch["input"]]

        #em_list, recall_list = self.calculate_scores(
        #    generated_text, output_list, batch["input"], batch_idx
        #)
        em_list, recall_list = self.ids_calculate_scores(
            generated_ids, output_list, batch["input"], batch_idx
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
                "gt_tok": list(batch["target_ids"].detach().cpu().numpy().tolist()),
                "pred": list(generated_text),
                "pred_tok": list(generated_ids),
                "em": list(em_list),
                "recall": list(recall_list),
            }
        else:
            return em_list, recall_list

    def _find_unique_path(self, seq, trie):
        _ids = seq.tolist()
        _ids = [int(elem) for elem in _ids]
        
        generated_ids = []
        if _ids[-1] == 0:
            for el in range(len(_ids)):
                if el != 0 and _ids[el] == 0:
                    continue
                else:
                    generated_ids.append(_ids[el])
        else:
            generated_ids = _ids 
        assert generated_ids[-1] != 0
        generated_ids = generated_ids[:-1]
        
        cur_dict = trie 
        for i in range(len(generated_ids) - 1):
            cur_dict = cur_dict[int(generated_ids[i])]
        if len(cur_dict[generated_ids[-1]]) == 0:
            add_list.append(generated_ids)
        else:
            cur_dict = cur_dict[int(generated_ids[-1])]
            # check if it is an unique path
            keys = list(cur_dict.keys())
            if len(keys) > 1:
                print(f"More than 1: {keys}")
                keys = [keys[0]]
            while len(cur_dict[keys[0]]) != 0:
                generated_ids.append(keys[0])
                cur_dict = cur_dict[keys[0]]
                keys = list(cur_dict.keys())
                if len(keys) > 1:
                    print(f"More than 1: {keys}")
                    keys = [keys[0]]
            generated_ids.append(1)
        return generated_ids

    def clusterList_to_text(self, ids):
        ret_list = []
        for _ids in ids:
            _ids = self.remove_padding(_ids, pad_token_id, eos_token_id, is_gt=False)
            _corpus_list = self.clusterIdList2corpusList[tuple(_ids[1:])]
            ret_list.append(_corpus_list)
        return ret_list

    def _test_step(self, batch, batch_idx, return_elem=False):
       
        # for case where it resume test 
        test_input = []
        for _input in batch['input']:
            if _input in self.test_input_list: continue
            else: test_input.append(_input) 
        test_num = len(test_input)
        start_num = len(batch['input'])-test_num
        if test_num == 0: return None
        test_output = batch['output'][start_num:]
        test_source_ids = batch['source_ids'][start_num:]
        test_source_masks = batch['source_mask'][start_num:]
        test_target_ids = batch['target_ids'][start_num:]
        test_target_masks = batch['target_mask'][start_num:]
        assert len(test_output) == test_num, f'test_output: {len(test_output)}\ttest_num: {test_num}'
        _trie_list = [copy.deepcopy(self.trie) for _ in range(test_num)]
        
        #unique_pred = []; unique_ids = []
        unique_ids_list = [[] for _ in range(test_num)] 
        over = [0]*test_num
        for _iter in range(self.hparams.val_beam_size*2):
            print("="*80)
            print(f"iter: {_iter} // DONE: {over.count(1)} / {test_num}")
            print(unique_ids_list)
            print("="*80)
            _generated_ids = self.model.generate(
                test_source_ids, 
                attention_mask=test_source_masks,
                use_cache=True,
                decoder_attention_mask=test_target_masks,
                max_length=self.hparams.max_output_length,
                num_beams=self.hparams.val_beam_size,
                num_return_sequences=self.hparams.val_beam_size,
                prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get_list(
                    batch_id, sent.tolist(), scores, trie_list=_trie_list
                ),
                early_stopping=True,
                decoder_start_token_id=self.pad_token_id,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id
            )
            ### TODO
            ## total -> ids_to_text 사용 불가 (하나의 ids가 하나의 text에 matching되지 않음.)
            ## total -> list of clusterid -> 바로 하나의 text에 matching 시키기
            #_generated_text = self.ids_to_text(_generated_ids)

            #_generated_text = self.clusterList_to_text(_generated_ids)

            inum = len(_generated_ids) // self.hparams.val_beam_size
            assert inum == len(test_output) 
            # generated_text = [
            #     _generated_text[
            #         i * self.hparams.val_beam_size : (i + 1) * self.hparams.val_beam_size
            #     ]
            #     for i in range(inum)
            # ]
            generated_ids = [
                _generated_ids[
                    i * self.hparams.val_beam_size : (i+1) * self.hparams.val_beam_size
                ].detach().cpu().numpy().tolist()
                for i in range(inum)
            ]
            """ 
            for b_texts, b_ids in zip(generated_text, generated_ids):
                for _text, _ids in zip(b_texts, b_ids):
                    g_ids = [self.tokId2groupId[el] for el in _ids]
            print(f"prediction: {generated_text}")
            print("*"*80)
            """

            for bid, batch_ids in enumerate(generated_ids):
                if over[bid] == 1: continue
                ### iterate over batch (val_beam_size)
                remove_ids = []
                _trie_dict = _trie_list[bid]
                for _ids in batch_ids:
                    _ids = self.remove_padding(_ids, self.pad_token_id, self.eos_token_id, is_gt=False)
                    _ids = tuple(_ids)
                    if _ids in unique_ids_list[bid]:
                        print(f"ids already exists in the list")
                        continue
                    elif _ids not in self.clusterIdList:
                        print(f"{_ids} does not exists in clusterIdList")
                        continue
                    else:
                        remove_ids.append(_ids) 

                #_trie_dict = self._remove_prev_from_trie(_trie_dict, remove_ids, self.pad_token_id)
                #_trie_list[bid] = _trie_dict
                unique_ids_list[bid].extend(remove_ids)
                if len(unique_ids_list[bid]) >= self.hparams.val_beam_size:
                    over[bid] = 1
            if over.count(1) == test_num:
                break
            """
            if (over.count(1) == test_num) or (self.cnt_over == self.len_test_dataset):
                self.cnt_over += over.count(1)
                break
            """
            
        for i in range(len(unique_ids_list)):
            unique_ids_list[i] = unique_ids_list[i][:self.hparams.val_beam_size]
        #self._flush_first_beam_dict()
        #if self.print: print(f"## UNIQUE PRED: {unique_pred[:self.hparams.val_beam_size]}")
        """
        em_list, recall_list = self.calculate_scores(
            unique_pred_list, batch["output"], batch["input"], batch_idx
        )
        """
        if return_elem:
            assert (
                len(list(test_input))
                == len(list(unique_ids_list))
            )
            return {
                "input": list(test_input),
                "gt": list(test_output),
                "gt_tok": [el.detach().cpu().numpy().tolist() for el in test_target_ids],
                "pred_tok": list(unique_ids_list),
            }
        else:
            return em_list, recall_list


class T5AsyncTuner(T5AsyncBaseTuner):
    def __init__(self, args):
        super(T5AsyncTuner, self).__init__(args)

        assert self.hparams.gr_decoder_only_encoder_ckpt is None
        assert self.hparams.gr_decoder_only is not True

        self.input2output = json.load(open(os.path.join(self.hparams.dataset, self.hparams.dev_input2output), "r"))

        # If in training mode, load ckpt for training
        if self.hparams.do_train:
            if self.hparams.train_c_emb:
                assert not self.hparams.reload_dataloader_every_n_epochs
                self.model = contextualized_T5.from_pretrained(
                    self.hparams.model_name_or_path, config=self.config, ignore_mismatched_sizes=True
                )
            else:
                self.model = contextualized_T5.from_pretrained(
                    self.hparams.model_name_or_path, config=self.config
                )
            assert self.model.get_dim() == self.hparams.model_dim

            self.tokenizer = T5Tokenizer.from_pretrained(
                self.hparams.tokenizer_name_or_path
            )

            if self.hparams.periflow:
                self.connect_str, self.container_name = get_blob_info()
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)

            if self.print:
                print(f"@@@ Loading Model from {self.hparams.model_name_or_path}")
                print(f'@@@ Loading decoder embedding: {self.hparams.embedding_model}')

            self.em_score_list = []
            self.recall_score_list = []

        # If in testing mode, load ckpt for inference
        if self.hparams.do_test:
            self.model = contextualized_T5.from_pretrained(
                self.hparams.test_model_path, config=self.config, ignore_mismatched_sizes=True
            )
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.test_model_path)
            if self.print:
                print(f"@@@ Loading Model from {self.hparams.test_model_path}")


            self.test_save_name = os.path.join(self.hparams.output_dir, f"{self.hparams.test_name}_{self.hparams.tree_type}_mbs_{self.hparams.max_beam_search}_result_beam{self.hparams.val_beam_size}.json")               
            if os.path.exists(self.test_save_name):
                prev_f = json.load(open(self.test_save_name))
                print(f"@@@ Loading Previous file!! => #: {len(prev_f['input'])}")
                self.test_input_list = prev_f['input']
                self.test_gt_list = prev_f['gt']
                self.test_gt_tok_list = prev_f['gt_tok']
                self.test_pred_list = prev_f['pred']
                self.test_pred_tok_list = prev_f['pred_tok']
                self.test_em_score_list = prev_f['em']
                self.test_recall_score_list = prev_f['recall']
            else:
                print(f'@@@ Initialize Test!!')
                self.test_input_list = []
                self.test_gt_list = []
                self.test_gt_tok_list = []
                self.test_pred_list = []
                self.test_pred_tok_list = []
                self.test_em_score_list = []
                self.test_recall_score_list = []

        if self.hparams.freeze_encoder:
            if self.print:
                print(f"@@@ Freeze Encoder!")
                encoder = self.model.get_encoder()
                for n, p in encoder.named_parameters():
                    p.requires_grad=False

        #### Tokenizer for generation step!
        self.dec_tok = AutoTokenizer.from_pretrained(
            self.hparams.embedding_model
        )
        
        if self.hparams.tree_type == "nodeId":
            nodeId_sup = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.nodeId_sup), 'rb'))
            self.nodeId2groupdId = nodeId_sup['group_set']
            self.nodeId2tokId = nodeId_sup['token_set']
            self.groupId2nodeId = nodeId_sup['inv_group_set']
            self.tokId2nodeId = nodeId_sup['inv_token_set']

        self.cnt_over = 0
        self.len_test_dataset = len(self.test_dataloader())

    def _encode_list(self, title_list, context_list, model, tokenizer):

        if context_list is not None:        
            context_list = [" ".join([_title, _sen]).strip() for (_title, _sen) in zip(title_list, context_list)]
            title_tok = [len(tokenizer(_title, return_tensors='pt', add_special_tokens=False).input_ids[0]) for _title in title_list]
            #print("title_tok: ", title_tok)
        else:
            context_list = title_list
            title_tok = [len(tokenizer(_title, return_tensors='pt', add_special_tokens=False).input_ids[0]) for _title in title_list]

        _tok = tokenizer(
                    context_list, 
                    return_tensors='pt', 
                    add_special_tokens=False, 
                    max_length=self.hparams.max_context_length,
                    padding="max_length",
                    truncation=True
                )

        _input_ids = _tok['input_ids'].to(model.device)
        _attention_mask = _tok["attention_mask"].to(model.device)
        encoder = model.get_encoder().eval()
        model_ret = encoder(input_ids=_input_ids, attention_mask=_attention_mask, return_dict=True)
        assert len(title_tok) == len(model_ret['last_hidden_state'])
        last_hidden_state = [state[:toklen].detach().cpu().numpy() for (state, toklen) in zip(model_ret['last_hidden_state'], title_tok)]
        _tok_decode = [tokenizer.convert_ids_to_tokens(_ids)[:toklen] for (_ids, toklen) in zip(_input_ids, title_tok)]
        _input_ids = _input_ids.detach().cpu().numpy()
        return _tok_decode, _input_ids, last_hidden_state

    def _construct_sp(self, model, tokenizer, dump_path=None, fh=None):
        tokId_emb = {} # {tokid: emb}
        tok_Idlist_dict = defaultdict(list) # {tok_text: [Idlist of the tok]}
        tok_Id_dict = {} # {Id: tok_text}

        # tokId = 0 -> <pad> token 
        _tok_decode, _input_ids, last_hidden_state = self._encode_list(["<pad>", "</s>"], None, model, tokenizer)
        assert len(_tok_decode) == 2
        
        tok_Idlist_dict[_tok_decode[0][0]].append(0)
        tok_Id_dict[0] = _tok_decode[0][0] 
        assert _input_ids[0][0] == 0

        tok_Idlist_dict[_tok_decode[1][0]].append(1)
        tok_Id_dict[1] = _tok_decode[1][0]
        assert _input_ids[1][0] == 1

        if self.hparams.do_save is None: assert dump_path is None

        if self.hparams.do_save is None:
            tokId_emb[0] = last_hidden_state[0][0]
            tokId_emb[1] = last_hidden_state[1][0]
        elif self.hparams.do_save == "hdf5":
            self._add_id4hdf5(dump_path, 0, last_hidden_state[0][0])
            self._add_id4hdf5(dump_path, 1, last_hidden_state[1][0])
        elif self.hparams.do_save == "dat":
            fh[0][:] = last_hidden_state[0][0]
            fh[1][:] = last_hidden_state[1][0]
            fh.flush()
        else:
            raise NotImplementedError("Check the saving type")
        return tok_Idlist_dict, tok_Id_dict, tokId_emb 

    def _get_dtype(self):
        if self.hparams.fp16:
            return "float16"
        else:
            return "float32"

    def _get_emb_from_file(self, hf, _id, file_type):
        if file_type == "hdf5":
            return np.array(hf.get(str(_id)))
        elif file_type == "dat":
            return hf[_id][:]
            #return np.memmap(path, dtype=self._get_dtype(), mode="r+", shape=(self.hparams.tok_num, self.hparams.model_dim))[_id][:]
        else:
            assert False

    def _add_id4hdf5(self, dump_path, _id, _data):
        f = h5py.File(dump_path, 'w', libver='latest') 
        f.create_dataset(str(_id), data=_data)
        f.close()
        return

    def _dump_cluster_corpus(self, corpus, context, model, tokenizer):
        if self.hparams.do_save == "hdf5":
            dump_path = os.path.join(self.hparams.dump_path, "temp_tokId_emb.hdf5")
            if os.path.exists(dump_path): os.system(f'rm {dump_path}')
            fh = None
        elif self.hparams.do_save == "dat":
            dump_path = os.path.join(self.hparams.dump_path, "temp_tokId_emb.dat")
            if os.path.exists(dump_path): os.system(f'rm {dump_path}')
            fh = np.memmap(dump_path, dtype=self._get_dtype(), mode="w+", shape=(self.hparams.tok_num, self.hparams.model_dim))
            fh.flush()
        elif self.hparams.do_save == "pickle":
            assert False 
        else:
            raise NotImplementedError("Check the saving type")
        print(f"================= Saving at {dump_path}")

        tok_Idlist_dict, tok_Id_dict, _ = self._construct_sp(model, tokenizer, dump_path, fh) 
        cur_tokId=2; corpusId=0
        corpusId_tokenList_dict = {}; corpus_tokenList_dict = {}
        print(f"Done Dumping SP tokens!\nStart dumping corpus!")
        for i in tqdm(range(0, len(corpus), self.hparams.dump_batch_size)):
            _corpus = corpus[i:i+self.hparams.dump_batch_size]
            if context is not None:
                _context = context[i:i+self.hparams.dump_batch_size]
            else:
                _context = None
            tok_decode_list, _, last_hidden_state_list = self._encode_list(_corpus, _context, model, tokenizer)

            for elem, tok_decode, last_hidden_state in zip(_corpus, tok_decode_list, last_hidden_state_list):
                assert len(tok_decode) == len(last_hidden_state)
                _tok_list = []
                for i, (_tok, _last_hidden_state) in enumerate(zip(tok_decode, last_hidden_state)):
                    if _tok == "<pad>": break
                    tok_Id_dict[cur_tokId] = _tok
                    tok_Idlist_dict[_tok].append(cur_tokId)

                    if self.hparams.do_save == "hdf5":
                        self._add_id4hdf5(dump_path, cur_tokId, _last_hidden_state)
                    elif self.hparams.do_save == "dat":
                        fh[cur_tokId][:] = _last_hidden_state
                    else:
                        assert False
                    
                    _tok_list.append(cur_tokId)
                    cur_tokId += 1

                _tok_list.append(1)
                corpusId_tokenList_dict[corpusId] = _tok_list
                corpus_tokenList_dict[elem] = _tok_list
                corpusId += 1
        if self.hparams.do_save == "dat":
            fh.flush()

        return tok_Idlist_dict, tok_Id_dict, dump_path, corpusId_tokenList_dict, corpus_tokenList_dict 

    def encode_sp(self, sen, model, tokenizer):
        _tok = tokenizer(sen, return_tensors='pt', add_special_tokens=False, max_length=2000)
        _input_ids = _tok['input_ids'].cuda()
        _attention_mask = _tok["attention_mask"].cuda()
        _tok_decode = tokenizer.convert_ids_to_tokens(_input_ids[0])
        model = model.get_encoder().eval()
        model_ret = model(input_ids=_input_ids, attention_mask=_attention_mask, return_dict=True)
        last_hidden_state = model_ret['last_hidden_state'][0]
        last_hidden_state = last_hidden_state.detach().cpu().numpy()
        _input_ids = _input_ids.detach().cpu().numpy()
        return _tok_decode, _input_ids, last_hidden_state

    def encode_context(self, title, context, model, tokenizer):
        context = " ".join([title, context])
        context = context.strip()
        _tok_decode_context, _input_ids_context, last_hidden_state_context = self.encode_sp(context, model, tokenizer)
        _tok_decode_title, _input_ids_title, last_hidden_state_title = self.encode_sp(title, model, tokenizer)

        last_hidden_state_title = last_hidden_state_context[:len(_input_ids_title[0])]
        return _tok_decode_title, _input_ids_title, last_hidden_state_title

    def construct_one_sp(self, model, tokenizer):

        tokId_emb = {} # {tokid: emb}
        tok_Idlist_dict = {} # {tok_text: [Idlist of the tok]}
        tok_Id_dict = {} # {Id: tok_text}

        # tokId = 0 -> <pad> token 
        _tok_decode, _input_ids, last_hidden_state = self.encode_sp("<pad>", model, tokenizer)
        assert len(_tok_decode) == 1
        tok_Idlist_dict[_tok_decode[0]] = [0]
        tok_Id_dict[0] = _tok_decode[0] 
        assert _input_ids[0][0] == 0
        tokId_emb[0] = last_hidden_state[0]

        # tokId = 1 -> </s> token
        _tok_decode, _input_ids, last_hidden_state = self.encode_sp("</s>", model, tokenizer)
        assert _tok_decode[0] == "</s>"
        assert len(_tok_decode) == 1
        tok_Idlist_dict[_tok_decode[0]] = [1]
        tok_Id_dict[1] = _tok_decode[0] 
        assert _input_ids[0][0] == 1
        tokId_emb[1] = last_hidden_state[0]
        return tok_Idlist_dict, tok_Id_dict, tokId_emb

    # def _dump_one_corpus(self, corpus_list, context_list, model, tokenizer):

    #     if self.hparams.cluster_num != -1 and self.hparams.do_save:
    #         dump_path = os.path.join(self.hparams.dataset, "temp_tokId_emb.hdf5")
    #         if os.path.exists(dump_path): os.system(f'rm {dump_path}')
    #         tok_Idlist_dict, tok_Id_dict, tokId_emb = self.construct_one_sp(model, tokenizer)
    #         # Save to hdf5
    #         print(f'Save Special Tokens to {dump_path}!')
    #         self._add_id4hdf5(dump_path, 0, tokId_emb[0])
    #         self._add_id4hdf5(dump_path, 1, tokId_emb[1])
    #     else:
    #         dump_path = None
    #         tok_Idlist_dict, tok_Id_dict, tokId_emb = self.construct_one_sp(model, tokenizer)


    #     tokId = 2
    #     fileId = 1
    #     corpusId_tokenList_dict = {}; corpus_tokenList_dict = {}
    #     for corpusId in tqdm(range(len(corpus_list))):

    #         elem = corpus_list[corpusId] # title
    #         if context_list is not None:
    #             context = context_list[corpusId]
    #         else:
    #             context = ""
    #         _tok_decode, _input_ids, last_hidden_state = self.encode_context(elem, context, model, tokenizer)

    #         _tok_dict = {}
    #         assert len(_input_ids[0])==len(last_hidden_state)==len(_tok_decode)

    #         for tok_pos, (_text, _ids, _emb) in enumerate(zip(_tok_decode, _input_ids[0], last_hidden_state)):
    #             tok_Id_dict[tokId] = _text 
    #             if _text not in tok_Idlist_dict.keys():
    #                 tok_Idlist_dict[_text] = [tokId]
    #             else:
    #                 tok_Idlist_dict[_text].append(tokId)
    #             _tok_dict[tokId] = _emb
    #             if self.hparams.cluster_num != -1 and self.hparams.do_save:
    #                 self._add_id4hdf5(dump_path, tokId, _emb)
    #             else:
    #                 tokId_emb[tokId] = _emb
    #             tokId += 1
                
    #             # Add EOS Token 
    #             if tok_pos == len(_tok_decode)-1:
    #                 _tok_dict[1] = tokId_emb[1]

    #         corpusId_tokenList_dict[corpusId] = list(_tok_dict.keys()) 
    #         corpus_tokenList_dict[corpusId] = list(_tok_dict.keys()) 

    #     return tok_Idlist_dict, tok_Id_dict, tokId_emb, corpusId_tokenList_dict, corpus_tokenList_dict

    def _dump_corpus(self, corpus, context, model, tokenizer):

        tok_Idlist_dict, tok_Id_dict, tokId_emb = self._construct_sp(model, tokenizer)
        cur_tokId = 2; corpusId = 0
        corpusId_tokenList_dict = {}; corpus_tokenList_dict = {}
        print(f"NOT saving the file!!")
        print(f"Done Dumping SP tokens!\nStart dumping corpus!")
        for i in tqdm(range(0, len(corpus), self.hparams.dump_batch_size)):
            _corpus = corpus[i:i+self.hparams.dump_batch_size]
            if context is not None:
                _context = context[i:i+self.hparams.dump_batch_size]
            else:
                _context = None
            tok_decode_list, _, last_hidden_state_list = self._encode_list(_corpus, _context, model, tokenizer)

            for elem, tok_decode, last_hidden_state in zip(_corpus, tok_decode_list, last_hidden_state_list):
                assert len(tok_decode) == len(last_hidden_state)
                _tok_list = []
                for i, (_tok, _last_hidden_state) in enumerate(zip(tok_decode, last_hidden_state)):
                    if _tok == "<pad>": break
                    tok_Id_dict[cur_tokId] = _tok
                    tok_Idlist_dict[_tok].append(cur_tokId)
                    tokId_emb[cur_tokId] = _last_hidden_state
                    _tok_list.append(cur_tokId)
                    cur_tokId += 1

                _tok_list.append(1)
                corpusId_tokenList_dict[corpusId] = _tok_list
                corpus_tokenList_dict[elem] = _tok_list
                corpusId += 1

        return tok_Idlist_dict, tok_Id_dict, tokId_emb, corpusId_tokenList_dict, corpus_tokenList_dict 

    def _construct_group(self, tok_Idlist_dict):
        tokId_tokGroupId = {}
        tokGroupId_tokIdList = {}
        tokGroupId = 2 ## assert tokGroupId 1 is </s> for generate()
        tokTextList = list(tok_Idlist_dict.keys())
        assert len(tokTextList) == len(set(tokTextList))
        for tokText, tokIdList in tok_Idlist_dict.items():
            if tokText == "</s>":
                print(f"Found </s> and set it to 1!!!")
                tokGroupId_tokIdList[1] = tokIdList  
                for tokId in tokIdList:
                    assert tokId not in tokId_tokGroupId.keys()
                    tokId_tokGroupId[tokId] = 1 
            elif tokText == "<pad>":
                print(f"Found <pad> and set it to 0!!!")
                tokGroupId_tokIdList[0] = tokIdList  
                for tokId in tokIdList:
                    assert tokId not in tokId_tokGroupId.keys()
                    tokId_tokGroupId[tokId] = 0 
            else:
                tokGroupId_tokIdList[tokGroupId] = tokIdList
                for tokId in tokIdList:
                    assert tokId not in tokId_tokGroupId.keys()
                    tokId_tokGroupId[tokId] = tokGroupId
                tokGroupId += 1
        return tokId_tokGroupId, tokGroupId_tokIdList

    def _construct_group_prefix_tree(self, corpusId_tokenList_dict, tokId_tokGroupId):
        sys.setrecursionlimit(900000000)

        constrained_dict = {}
        for corpusId, tokIdList in corpusId_tokenList_dict.items():
            cur_dict = constrained_dict # cur_dict[-2]: the node number
            #tokIdList = list(corpusDict.keys())
            tokGroupIdList = [tokId_tokGroupId[el] for el in tokIdList]
            tokGroupIdList = [0] + tokGroupIdList
            
            for i in range(len(tokGroupIdList)-1):
                prev = tokGroupIdList[i]
                cur = tokGroupIdList[i+1]
                
                if i == len(tokGroupIdList)-2:
                    if prev in cur_dict.keys():
                        if cur not in cur_dict[prev].keys():
                            cur_dict[prev][cur] = {} 
                    else:
                        cur_dict[prev] = {cur: {}}
                else:
                    if prev in cur_dict.keys():
                        pass
                    else:
                        cur_dict[prev] = {}
                    cur_dict = cur_dict[prev] 
        return constrained_dict

    def _do_cluster(self, tokGroupId_tokIdList, tokId_tokGroupId, tokId_embs, tokId_tokText, load_file=False):
        assert self.hparams.cluster_num > 0
        tokText2clusterIdList = defaultdict(list)
        tokId2clusterId = {}
        tokGroupId2clusterIdList = defaultdict(list)
        clusterId2tokGroupId = {}
        clusterId2tokText = {}
        clusterId2clusterEmb = {}
        clusterId = 0

        if self.hparams.do_save == "hdf5":
            tokId_embs = h5py.File(tokId_embs, 'r')
        elif self.hparams.do_save == "dat":
            tokId_embs = np.memmap(tokId_embs, dtype=self._get_dtype(), mode="r+", shape=(self.hparams.tok_num, self.hparams.model_dim))
        else:
            assert False

        for tokGroupId, id_list in tqdm(tokGroupId_tokIdList.items()):
            text = tokId_tokText[id_list[0]]
            if not load_file:
                emb_list = [tokId_embs[id] for id in id_list]
            else:
                emb_list = [self._get_emb_from_file(hf=tokId_embs, _id=id, file_type=self.hparams.do_save) for id in id_list]

            # do cluster
            if len(emb_list) > self.hparams.cluster_num:
                df = pd.DataFrame(emb_list) 
                kmeans = KMeans(n_clusters=self.hparams.cluster_num, algorithm='auto') #, max_iter=50, n_init=3)
                kmeans.fit(df)
                predicts = np.array(kmeans.predict(df))
                centers = kmeans.cluster_centers_
                """ 
                kmeans = FaissKMeans(n_clusters=self.hparams.cluster_num)
                centers = kmeans.fit(np.array(emb_list), np.array(id_list))
                predicts = np.array(kmeans.predict(np.array(emb_list)))
                """
            else:
                predicts = np.array([i for i  in range(len(id_list))])
                centers = np.array(emb_list)
            
            clusterIdDict = {}
            for i, center in enumerate(centers):
                clusterIdDict[i] = clusterId 
                clusterId2tokText[clusterId] = text
                clusterId2clusterEmb[clusterId] = center 
                clusterId += 1

            for _predict, _id in zip(predicts, id_list):
                _group = tokId_tokGroupId[_id]
                _clusterId = clusterIdDict[_predict]
                tokGroupId2clusterIdList[_group].append(_clusterId)
                clusterId2tokGroupId[_clusterId] = _group
                tokId2clusterId[_id] = _clusterId
                tokText2clusterIdList[text].append(_clusterId)

            if text == "<pad>" or text == "</s>":
                assert len(id_list) == 1 and len(tokText2clusterIdList[text]) == 1
            if clusterId == 1: assert tokId2clusterId[0] == 0
            if clusterId == 2: assert tokId2clusterId[1] == 1

        return tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId, clusterId2clusterEmb



    def _dump_new_dataset(self, path=None):
        df = pd.read_csv(self.hparams.corpus_file)
        if "context" in df.keys():
            print(f'### Using corpus with paragraph: {len(df)}')
            corpus_file = df.fillna('')
            corpus = corpus_file["corpus"]
            context = corpus_file["context"]
            assert len(corpus) == len(context)
        else:
            print(f'### Using corpus withOUT paragraph: {len(df)}')
            corpus = list(df.fillna("")["corpus"])
            context = None

        if path == "base":
            _model = T5EncoderModel.from_pretrained('t5-large').cuda()
            if self.print: 
                print(f'=== Model in {_model.device} ===')
                print(f'*** Construct from Base Model!')
            _tokenizer = T5Tokenizer.from_pretrained('t5-large') 
        elif path == "test":
            _model = T5EncoderModel.from_pretrained(self.hparams.test_model_path).cuda()
            if self.print: 
                print(f'=== Model in {_model.device} ===')
                print(f'*** Construct from Test Model: {self.hparams.test_model_path}!')
            _tokenizer = T5Tokenizer.from_pretrained(self.hparams.test_model_path)
        elif path == "resume":
            _resume_path = os.path.join(self.hparams.output_dir, f"best_tfmr_{self.load_epoch}")
            _model = T5EncoderModel.from_pretrained(_resume_path).cuda()
            if self.print: 
                print(f'=== Model in {_model.device} ===')
                print(f'*** Resume from : {_resume_path}!')
            _tokenizer = T5Tokenizer.from_pretrained(_resume_path) 
        elif path is None:
            _model = self.model
            _tokenizer = self.tokenizer
        else:
            assert False

        print(f'$$$$ START dumping embedding!!')
        """
        if self.hparams.dump_batch_size == 1:
            if self.hparams.cluster_num == -1:
                tok_Idlist_dict, tok_Id_dict, tokId_emb, corpusId_tokenList_dict, corpus_tokenList_dict = self._dump_one_corpus(corpus, context, _model, _tokenizer) 
                os.makedirs(self.hparams.dataset, exist_ok=True)
                with open(os.path.join(self.hparams.dataset, 'temp_tokId_emb.pickle'), "wb") as f:
                    pickle.dump(tokId_emb, f)
                print(f'$$$$ DONE dumping embedding!!')
                tokId_tokGroupId, tokGroupId_tokIdList = self._construct_group(tok_Idlist_dict)
                print(f'0 in tokGroupId: {0 in tokGroupId_tokIdList.keys()}')
                groupId_tree = self._construct_group_prefix_tree(corpusId_tokenList_dict, tokId_tokGroupId)
                print(f'$$$$ DONE dumping group Info!!')
                if path is None: 
                    assert len(tokId_emb) == self.contextualized_emb_num, f"# of tokId_emb: {len(tokId_emb.keys())} // contextualized_emb_num: {self.contextualized_emb_num}"
                    self.model.set_contextualized_file(os.path.join(self.hparams.dataset, "temp_tokId_emb.pickle"))
                    self.model = self.model.train().to(self.device)
                del _model; del _tokenizer
                return tok_Id_dict, tokId_tokGroupId, tokGroupId_tokIdList, groupId_tree, tokId_emb, corpus_tokenList_dict
            else:
                tok_Idlist_dict, tok_Id_dict, tokId_emb, corpusId_tokenList_dict, corpus_tokenList_dict = self._dump_one_corpus(corpus, context, _model, _tokenizer) 
                #tok_Idlist_dict, tok_Id_dict, tokId_emb, corpusId_tokenList_dict, corpus_tokenList_dict = self._dump_corpus(corpus, context) 
                tokId_tokGroupId, tokGroupId_tokIdList = self._construct_group(tok_Idlist_dict)
                groupId_tree = self._construct_group_prefix_tree(corpusId_tokenList_dict, tokId_tokGroupId)
                print(f'$$$$ DONE dumping group Info!!')
                tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId, clusterId2clusterEmb = self._do_cluster(tokGroupId_tokIdList, tokId_tokGroupId, tokId_emb, tok_Id_dict, not_hdf5=True)
                corpus_clusterList_dict = self._construct_corpus2clusterList(corpus_tokenList_dict, tokId2clusterId)
                print(f'$$$$ DONE dumping cluster Info!!')
                with open(os.path.join(self.hparams.dump_path, 'temp_clusterId_emb.pickle'), "wb") as f:
                    pickle.dump(clusterId2clusterEmb, f)
                if path is None: 
                    assert len(clusterId2clusterEmb.keys()) == self.contextualized_emb_num, f"# of clusterId2clusterEmb: {len(clusterId2clusterEmb.keys())} // contextualized_emb_num: {self.contextualized_emb_num}"
                    self.model.set_contextualized_file(os.path.join(self.hparams.dump_path, "temp_clusterId_emb.pickle"))
                    self.model = self.model.train().to(self.device)
                del _model; del _tokenizer
                return clusterId2tokText, clusterId2tokGroupId, tokGroupId2clusterIdList, groupId_tree, clusterId2clusterEmb, corpus_clusterList_dict 

        else:
        """
        if self.hparams.cluster_num == -1:
            tok_Idlist_dict, tok_Id_dict, tokId_emb, corpusId_tokenList_dict, corpus_tokenList_dict = self._dump_corpus(corpus, context, _model, _tokenizer) 
            os.makedirs(self.hparams.dataset, exist_ok=True)
            with open(os.path.join(self.hparams.dump_path, 'temp_tokId_emb.pickle'), "wb") as f:
                pickle.dump(tokId_emb, f)
            print(f'$$$$ DONE dumping embedding!!')
            tokId_tokGroupId, tokGroupId_tokIdList = self._construct_group(tok_Idlist_dict)
            groupId_tree = self._construct_group_prefix_tree(corpusId_tokenList_dict, tokId_tokGroupId)
            print(f'$$$$ DONE dumping group Info!!')
            if path is None: 
                assert len(tokId_emb) == self.contextualized_emb_num, f"# of tokId_emb: {len(tokId_emb.keys())} // contextualized_emb_num: {self.contextualized_emb_num}"
                self.model.set_contextualized_file(os.path.join(self.hparams.dump_path, "temp_tokId_emb.pickle"))
                self.model = self.model.train().to(self.device)
            del _model; del _tokenizer
            return tok_Id_dict, tokId_tokGroupId, tokGroupId_tokIdList, groupId_tree, tokId_emb, corpus_tokenList_dict
        
        else:
            if self.hparams.do_save is not None:
                tok_Idlist_dict, tok_Id_dict, dump_path, corpusId_tokenList_dict, corpus_tokenList_dict = self._dump_cluster_corpus(corpus, context, _model, _tokenizer) 
                print(f'$$$$ DONE dumping embedding!!')
                #tokId_emb = h5py.File(dump_path, 'r')
                tokId_tokGroupId, tokGroupId_tokIdList = self._construct_group(tok_Idlist_dict)
                groupId_tree = self._construct_group_prefix_tree(corpusId_tokenList_dict, tokId_tokGroupId)
                print(f'$$$$ DONE dumping group Info!!')
                tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId, clusterId2clusterEmb = self._do_cluster(tokGroupId_tokIdList, tokId_tokGroupId, dump_path, tok_Id_dict, load_file=True)
                corpus_clusterList_dict = self._construct_corpus2clusterList(corpus_tokenList_dict, tokId2clusterId)
                print(f'$$$$ DONE dumping cluster Info!!')
                with open(os.path.join(self.hparams.dump_path, 'temp_clusterId_emb.pickle'), "wb") as f:
                    pickle.dump(clusterId2clusterEmb, f)
                if path is None: 
                    #assert len(clusterId2clusterEmb.keys()) == self.contextualized_emb_num, f"# of clusterId2clusterEmb: {len(clusterId2clusterEmb.keys())} // contextualized_emb_num: {self.contextualized_emb_num}"
                    self.model.set_contextualized_file(os.path.join(self.hparams.dump_path, "temp_clusterId_emb.pickle"))
                    self.model = self.model.train().to(self.device)
                del _model; del _tokenizer
                return clusterId2tokText, clusterId2tokGroupId, tokGroupId2clusterIdList, groupId_tree, clusterId2clusterEmb, corpus_clusterList_dict 
            else:
                tok_Idlist_dict, tok_Id_dict, tokId_emb, corpusId_tokenList_dict, corpus_tokenList_dict = self._dump_corpus(corpus, context, _model, _tokenizer) 
                tokId_tokGroupId, tokGroupId_tokIdList = self._construct_group(tok_Idlist_dict)
                groupId_tree = self._construct_group_prefix_tree(corpusId_tokenList_dict, tokId_tokGroupId)
                print(f'$$$$ DONE dumping group Info!!')
                tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId, clusterId2clusterEmb = self._do_cluster(tokGroupId_tokIdList, tokId_tokGroupId, tokId_emb, tok_Id_dict)
                corpus_clusterList_dict = self._construct_corpus2clusterList(corpus_tokenList_dict, tokId2clusterId)
                print(f'$$$$ DONE dumping cluster Info!!')
                with open(os.path.join(self.hparams.dump_path, 'temp_clusterId_emb.pickle'), "wb") as f:
                    pickle.dump(clusterId2clusterEmb, f)
                if path is None: 
                    #assert len(clusterId2clusterEmb.keys()) == self.contextualized_emb_num, f"# of clusterId2clusterEmb: {len(clusterId2clusterEmb.keys())} // contextualized_emb_num: {self.contextualized_emb_num}"
                    self.model.set_contextualized_file(os.path.join(self.hparams.dump_path, "temp_clusterId_emb.pickle"))
                    self.model = self.model.train().to(self.device)
                del _model; del _tokenizer
                return clusterId2tokText, clusterId2tokGroupId, tokGroupId2clusterIdList, groupId_tree, clusterId2clusterEmb, corpus_clusterList_dict 

    def _construct_corpus2clusterList(self, corpus_tokenList_dict, tokId2clusterId):
        corpus_clusterList_dict = {}
        for sen, tokenList in corpus_tokenList_dict.items():
            corpus_clusterList_dict[sen] = [tokId2clusterId[el] for el in tokenList] 
        return corpus_clusterList_dict

    def _get_dataset(self, split):
        if self.hparams.reload_dataloader_every_n_epochs:
            if split!="test" and self.current_epoch != 0 and self.current_epoch != self.load_epoch+1:
                if self.print: print(f"**** Dumping Dataset for Epoch {self.current_epoch}")
                self.tokId2tokText, self.tokId2groupId, self.groupId2tokId, self.trie, self.contextualized_tokid2emb, self.corpus_tokenList_dict = self._dump_new_dataset(path=None)
            
            dataset = GENREDataset(
                tokenizer=self.tokenizer,
                split=split,
                hparams=self.hparams,
                tokid2emb=self.contextualized_tokid2emb,
                corpus_tokenList_dict=self.corpus_tokenList_dict
            ) 
        else:
            dataset = GENREDataset(
                tokenizer=self.tokenizer,
                split=split,
                hparams=self.hparams,
                tokid2emb=self.contextualized_tokid2emb 
            )
        return dataset

    def _get_max_tokId_from_tokIdList(self, tokIdList, score):
        tokIdList = sorted(tokIdList)
        idx = score[tokIdList].detach().cpu().numpy().argmax()
        max_tokId = tokIdList[idx]
        return max_tokId

    def _get_tokIdList_from_nodeIdList(self, nodeIdList, score):                        
        assert self.hparams.tree_type == "nodeId"            
        tokIdList = []
        if self.hparams.max_beam_search:
            for nodeId in nodeIdList:
                max_tokId = self._get_max_tokId_from_tokIdList(list(self.nodeId2tokId[nodeId]), score)
                tokIdList.append(max_tokId)
            return list(set(tokIdList))
        else:                
            for nodeId in nodeIdList:
                tokIdList.extend(list(self.nodeId2tokId[nodeId]))                
            return list(set(tokIdList))
    
    def _get_tokIdList_from_groupIdList(self, groupIdList, score):
        assert self.hparams.tree_type == "groupId"
        tokIdList = []
        
        # for groupId Tree with max beam search
        if self.hparams.max_beam_search:
            for groupId in groupIdList:
                max_tokId = self._get_max_tokId_from_tokIdList(self.groupId2tokId[groupId], score)
                tokIdList.append(max_tokId)
            return list(set(tokIdList))
        # for normal groupId Tree
        else:
            for groupId in groupIdList:
                tokIdList.extend(self.groupId2tokId[groupId])
            return list(set(tokIdList))
   
    def _get_nodeId_from_tokId(self, tokId):
        nodeId = list(self.tokId2node/_teId[tokId])
        assert len(nodeId) == 1
        return nodeId[0] 
    
    def _get_groupId_from_tokId(self, tokId):
        return self.tokId2groupId[tokId]


    def _loss(self, batch):
        lm_labels = copy.deepcopy(batch["target_ids"])
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
        )
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
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

    def get_list(self, batch_id, input_ids, score, trie_list):
        # starts with pad token & groupId for pad token is -1
        assert input_ids[0] == 0
        assert trie_list is not None and isinstance(trie_list, list)
        return self._get_from_trie(input_ids, trie_list[batch_id], score)

    def get(self, batch_id, input_ids, score, trie_dict=None):
        # starts with pad token & groupId for pad token is -1
        assert input_ids[0] == 0
        if trie_dict is None:
            trie_dict = self.trie
        return self._get_from_trie(input_ids, trie_dict, score)

    """
    input_ids가 들어오면, 해당 tokId가 속한 groupId 찾고, 그걸 가지고 trie_dict 넘어간 다음
    해당 subtree의 key들(groupId) 를 모은 tokId return
    """
    def _get_from_trie(self, input_ids, trie_dict, score):
        #print(f"input_ids: {input_ids}")
        if self.hparams.tree_type == "groupId":
            if len(input_ids) == 0:
                possible_GroupList = list(trie_dict.keys())
                tokIdList = self._get_tokIdList_from_groupIdList(possible_GroupList, score)
                return tokIdList
            else:
                curGroupId = self._get_groupId_from_tokId(input_ids[0])
                if curGroupId in list(trie_dict.keys()):
                    return self._get_from_trie(input_ids[1:], trie_dict[curGroupId], score) 
                else:
                    return []
        elif self.hparams.tree_type == "nodeId":
            if input_ids[-1] == 1:
                return []
            else:
                NodeId = self._get_nodeId_from_tokId(input_ids[-1])
                next_nId_List = list(trie_dict[NodeId])
                tokIdList = self._get_tokIdList_from_nodeIdList(next_nId_List, score)
                return tokIdList
        elif self.hparams.tree_type == "clusterId":
            if len(input_ids) == 0:
                return list(trie_dict.keys()) 
            else:
                if input_ids[0] in list(trie_dict.keys()):
                    return self._get_from_trie(input_ids[1:], trie_dict[clusterId], score)
                else:
                    return []
        else:
            raise NotImplementedError('tree type should be either groupId_tree or nodeId_tree!')

    def calculate_scores(self, preds, gt_text, query, batch_idx):
        em_list = []
        recall_list = []
        for idx, (_query, _pred, _gt) in enumerate(zip(query, preds, gt_text)):
            _em = self._calculate_em(_pred[0], _gt)
            _recall = self._calculate_recall(_pred, _gt)
            if self.print and self.current_epoch==0 and idx == 0:
                print(f"$" * 50)
                print(f"query: {_query}\npreds: {_pred}\ngt: {_gt}")
                print(f"em: {_em} // recall: {_recall}")
                print(f"$" * 50)
                print(" ")
            em_list.append(_em)
            recall_list.append(_recall)
        return em_list, recall_list

    def _val_step(self, batch, batch_idx, return_elem=False):
        # calculates recall and em -> returns the list of each score
        _generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            #decoder_attention_mask=batch["target_mask"],
            max_length=self.hparams.max_output_length,
            num_beams=self.hparams.val_beam_size,
            num_return_sequences=self.hparams.val_beam_size,
            prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get(
                batch_id, sent.tolist(), scores
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
        generated_ids = [
            _generated_ids[
               i * self.hparams.val_beam_size : (i+1) * self.hparams.val_beam_size
            ].detach().cpu().numpy().tolist()
            for i in range(inum)
        ]

        gt_output = [self.input2output[_input] for _input in batch["input"]]
        em_list, recall_list = self.calculate_scores(
            generated_text, gt_output, batch["input"], batch_idx
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
                "gt_tok": list(batch["target_ids"].detach().cpu().numpy().tolist()),
                "pred": list(generated_text),
                "pred_tok": list(generated_ids),
                "em": list(em_list),
                "recall": list(recall_list),
            }
        else:
            return em_list, recall_list


    def _find_unique_path(self, seq, trie):
        _ids = seq.tolist()
        _ids = [int(elem) for elem in _ids]
        
        generated_ids = []
        if _ids[-1] == 0:
            for el in range(len(_ids)):
                if el != 0 and _ids[el] == 0:
                    continue
                else:
                    generated_ids.append(_ids[el])
        else:
            generated_ids = _ids 
        assert generated_ids[-1] != 0
        generated_ids = generated_ids[:-1]
        
        cur_dict = trie 
        for i in range(len(generated_ids) - 1):
            cur_dict = cur_dict[int(generated_ids[i])]
        if len(cur_dict[generated_ids[-1]]) == 0:
            add_list.append(generated_ids)
        else:
            cur_dict = cur_dict[int(generated_ids[-1])]
            # check if it is an unique path
            keys = list(cur_dict.keys())
            if len(keys) > 1:
                print(f"More than 1: {keys}")
                keys = [keys[0]]
            while len(cur_dict[keys[0]]) != 0:
                generated_ids.append(keys[0])
                cur_dict = cur_dict[keys[0]]
                keys = list(cur_dict.keys())
                if len(keys) > 1:
                    print(f"More than 1: {keys}")
                    keys = [keys[0]]
            generated_ids.append(1)
        return generated_ids

    
    def _test_step(self, batch, batch_idx, return_elem=False):
       
        # for case where it resume test 
        test_input = []
        for _input in batch['input']:
            if _input in self.test_input_list: continue
            else: test_input.append(_input) 
        test_num = len(test_input)
        start_num = len(batch['input'])-test_num
        if test_num == 0: return None

        test_output = batch['output'][start_num:]
        test_source_ids = batch['source_ids'][start_num:]
        test_source_masks = batch['source_mask'][start_num:]
        test_target_ids = batch['target_ids'][start_num:]
        test_target_masks = batch['target_mask'][start_num:]
        assert len(test_output) == test_num, f'test_output: {len(test_output)}\ttest_num: {test_num}'

        _trie_list = [copy.deepcopy(self.trie) for _ in range(test_num)]
        
        #unique_pred = []; unique_ids = []
        unique_pred_list = [[] for _ in range(test_num)] 
        unique_ids_list = [[] for _ in range(test_num)] 
        over = [0]*test_num
        
        for _iter in range(self.hparams.val_beam_size*2):
            print("="*80)
            print(f"iter: {_iter} // DONE: {over.count(1)} / {test_num}")
            print(unique_pred_list)
            print("="*80)
            _generated_ids = self.model.generate(
                test_source_ids, 
                attention_mask=test_source_masks,
                use_cache=True,
                decoder_attention_mask=test_target_masks,
                max_length=self.hparams.max_output_length,
                num_beams=self.hparams.val_beam_size,
                num_return_sequences=self.hparams.val_beam_size,
                prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get_list(
                    batch_id, sent.tolist(), scores, trie_list=_trie_list
                ),
                early_stopping=True,
            )

            _generated_text = self.ids_to_text(_generated_ids)
            inum = len(_generated_ids) // self.hparams.val_beam_size
            assert inum == len(test_output) 
            generated_text = [
                _generated_text[
                    i * self.hparams.val_beam_size : (i + 1) * self.hparams.val_beam_size
                ]
                for i in range(inum)
            ]
            generated_ids = [
                _generated_ids[
                    i * self.hparams.val_beam_size : (i+1) * self.hparams.val_beam_size
                ].detach().cpu().numpy().tolist()
                for i in range(inum)
            ]
            '''   
            for b_texts, b_ids in zip(generated_text, generated_ids):
                for _text, _ids in zip(b_texts, b_ids):
                    g_ids = [self.tokId2groupId[el] for el in _ids]
            ''' 
            for bid, (batch_text, batch_ids) in enumerate(zip(generated_text, generated_ids)):
                if over[bid] == 1: continue
                ### iterate over batch (val_beam_size)
                _upper_ids = []
                _trie_dict = _trie_list[bid]
                for _text, _ids in zip(batch_text, batch_ids):
                    if _ids in unique_ids_list[bid]:
                        continue
                    else:
                        if _text in unique_pred_list[bid]:
                            assert not self.hparams.tree_type == "nodeId"
                            _upper_ids.append([self.tokId2groupId[el] for el in _ids]) 
                        elif len(unique_pred_list[bid]) < self.hparams.val_beam_size:
                            unique_pred_list[bid].append(_text)
                            unique_ids_list[bid].append(_ids)
                            if self.hparams.tree_type == "groupId":
                                _upper_ids.append([self.tokId2groupId[el] for el in _ids]) 
                            elif self.hparams.tree_type == "nodeId":
                                temp = []
                                eos_pos = (np.array(_ids) == 1).nonzero()[0][0]
                                for el in _ids[:eos_pos]:
                                    assert len(self.tokId2nodeId[el]) == 1, self.tokId2nodeId[el]
                                    temp.append(list(self.tokId2nodeId[el])[0])
                                # find the end token
                                cur_nId = temp[-1]
                                next_nId = _trie_dict[cur_nId]
                                end_nId = list(next_nId.intersection(self.tokId2nodeId[1]))
                                assert len(end_nId) == 1
                                temp.append(end_nId[0])
                                _upper_ids.append(temp)
                            else:
                                assert False
                        else:
                            pass
                # remove from _trie_dict
                _trie_dict = self._remove_prev_from_trie(_trie_dict, _upper_ids)
                _trie_list[bid] = _trie_dict
                if len(unique_pred_list[bid]) >= self.hparams.val_beam_size:
                    over[bid] = 1
            if over.count(1) == test_num:
                break
            ''' 
            if (over.count(1) == test_num) or (self.cnt_over == self.len_test_dataset):
                self.cnt_over += over.count(1)
                break
            ''' 
            
        for i in range(len(unique_pred_list)):
            unique_pred_list[i] = unique_pred_list[i][:self.hparams.val_beam_size]

        #self._flush_first_beam_dict()
        #if self.print: print(f"## UNIQUE PRED: {unique_pred[:self.hparams.val_beam_size]}")
        em_list, recall_list = self.calculate_scores(
            unique_pred_list, batch["output"], batch["input"], batch_idx
        )
        if return_elem:
            assert (
                len(list(test_input))
                == len(list(generated_text))
                == len(list(em_list))
                == len(list(unique_pred_list))
                == len(list(unique_ids_list))
            )
            return {
                "input": list(test_input),
                "gt": list(test_output),
                "gt_tok": [el.detach().cpu().numpy().tolist() for el in test_target_ids],
                "pred": list(unique_pred_list),
                "pred_tok": list(unique_ids_list),
                "em": list(em_list),
                "recall": list(recall_list),
            }
        else:
            return em_list, recall_list
    
    """
    def _test_step(self, batch, batch_idx, return_elem=False):
       
        # for case where it resume test 
        test_input = []
        for _input in batch['input']:
            if _input in self.test_input_list: continue
            else: test_input.append(_input) 
        test_num = len(test_input)
        start_num = len(batch['input'])-test_num
        if test_num == 0: return None

        test_output = batch['output'][start_num:]
        test_source_ids = batch['source_ids'][start_num:]
        test_source_masks = batch['source_mask'][start_num:]
        test_target_ids = batch['target_ids'][start_num:]
        test_target_masks = batch['target_mask'][start_num:]
        assert len(test_output) == test_num, f'test_output: {len(test_output)}\ttest_num: {test_num}'

        _trie_list = [copy.deepcopy(self.trie) for _ in range(test_num)]
        
        #unique_pred = []; unique_ids = []
        unique_pred_list = [[] for _ in range(test_num)] 
        unique_ids_list = [[] for _ in range(test_num)] 
        over = [0]*test_num
        
        _generated_ids = self.model.generate(
            test_source_ids, 
            attention_mask=test_source_masks,
            use_cache=True,
            decoder_attention_mask=test_target_masks,
            max_length=self.hparams.max_output_length,
            num_beams=self.hparams.val_beam_size,
            num_return_sequences=self.hparams.val_beam_size,
            early_stopping=True,
        )

        _generated_text = self.ids_to_text(_generated_ids)
        inum = len(_generated_ids) // self.hparams.val_beam_size
        assert inum == len(test_output) 
        generated_text = [
            _generated_text[
                i * self.hparams.val_beam_size : (i + 1) * self.hparams.val_beam_size
            ]
            for i in range(inum)
        ]
        generated_ids = [
            _generated_ids[
                i * self.hparams.val_beam_size : (i+1) * self.hparams.val_beam_size
            ].detach().cpu().numpy().tolist()
            for i in range(inum)
        ]

        #self._flush_first_beam_dict()
        #if self.print: print(f"## UNIQUE PRED: {unique_pred[:self.hparams.val_beam_size]}")
        em_list, recall_list = self.calculate_scores(
            generated_text, batch["output"], batch["input"], batch_idx
        )
        if return_elem:
            assert (
                len(list(test_input))
                == len(list(generated_text))
                == len(list(em_list))
            )
            return {
                "input": list(test_input),
                "gt": list(test_output),
                "gt_tok": [el.detach().cpu().numpy().tolist() for el in test_target_ids],
                "pred": list(generated_text),
                "pred_tok": list(generated_ids),
                "em": list(em_list),
                "recall": list(recall_list),
            }
        else:
            return em_list, recall_list
    """

class T5JointTuner(T5BaseClass):
    def __init__(self, args):
        super(T5JointTuner, self).__init__()
        self.save_hyperparameters(args)
        if self.hparams.do_train:
            config = T5Config.from_pretrained(self.hparams.model_name_or_path)
            config.update({'total_emb': None})
            self.model = joint_T5.from_pretrained(
                self.hparams.model_name_or_path, config=config
            )
            self.emb_enc = T5EncoderModel.from_pretrained(
                self.hparams.model_name_or_path
            )
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.hparams.tokenizer_name_or_path
            )
            self.tokId2tokText = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2tokText), "rb"))

            # self.tokId2Emb = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), "rb"))

            # self.tokEmbList = [torch.tensor(emb).unsqueeze(0).to(self.device) for emb in self.tokId2Emb.values()]
            # #self.tokEmbList = [np.expand_dims(emb, axis=0) for emb in self.tokId2Emb.values()]
            # self.total_emb = torch.cat(self.tokEmbList, dim=0)

            # if self.hparams.freeze_emb_enc:
            # if self.print:
            print(f"@@@ Freeze Embedding Encoder!")
            for n, p in self.emb_enc.named_parameters():
                p.requires_grad=False
        
        if self.hparams.do_test:

            self.pad_tokenid = 0; self.end_tokenid = 1

            self.trie = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tree_path), "rb"))
            self.tokId2Emb = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), "rb"))
            self.groupId2tokId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.groupId2tokIdList), "rb"))
            self.tokId2groupId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2groupId), "rb"))
            self.tokId2tokText = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2tokText), "rb"))
            self.toktext2tokidlist = pickle.load(open("/data/project/rw/contextualized_GENRE/dataset/hotpot/title_context_overfit_200/tokText2tokIdList.pickle",'rb'))

            self.tokEmbList = [torch.tensor(emb).unsqueeze(0).to(self.device) for emb in self.tokId2Emb.values()]
            #self.tokEmbList = [np.expand_dims(emb, axis=0) for emb in self.tokId2Emb.values()]
            self.total_emb = torch.cat(self.tokEmbList, dim=0)
            self.end_emb = self.tokEmbList[self.end_tokenid] 
            self.pad_emb = self.tokEmbList[self.pad_tokenid]

            config = T5Config.from_pretrained(self.hparams.model_name_or_path)
            config.update({'total_emb': self.total_emb})

            self.model = joint_T5.from_pretrained(
                os.path.join(self.hparams.test_model_path, "model")
            )
            self.emb_enc = T5EncoderModel.from_pretrained(
                os.path.join(self.hparams.test_model_path, "emb_enc")
            )
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.hparams.test_model_path
            )

            self.test_em_score_list = []
            self.test_recall_score_list = []
            self.test_input_list = []
            self.test_gt_list = []
            self.test_pred_list = []
            
            self.test_save_name = os.path.join(self.hparams.output_dir, f"{self.hparams.test_name}_result_beam{self.hparams.val_beam_size}.json")               

        if self.print:
            print(f'@@@ Loading Model from {self.hparams.model_name_or_path}')

        self.loss_fct = nn.CrossEntropyLoss()
        self.val_loss = []; self.val_em = []
        os.makedirs(self.hparams.output_dir, exist_ok=True)
        if 'log.txt' in os.listdir(self.hparams.output_dir):
            print(f'+++ removing previous log file!')
            os.system(f'rm {os.path.join(self.hparams.output_dir, "log.txt")}')
        self.file = open(os.path.join(self.hparams.output_dir, "log.txt"), 'w') 
        print(f'+++ Writing logs in {os.path.join(self.hparams.output_dir, "log.txt")}')

    def _get_dataset(self, split):
        dataset = JOINTDataset(self.tokenizer, split, self.hparams)
        return dataset

    def forward(self, input_ids, attention_mask, decoder_inputs_embeds, decoder_attention_mask=None):
        return self.model(
                    input_ids, 
                    attention_mask=attention_mask, 
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    decoder_attention_mask=decoder_attention_mask
                ) 
        
    def _get_embedding(self, batch): ## embedding of query        
        last_hidden_state = self(
                            input_ids=batch["source_ids"], 
                            attention_mask=batch["source_mask"], 
                            decoder_inputs_embeds=batch["target_emb"],
                            decoder_attention_mask=batch["target_mask"],
                            ).last_hidden_state
        assert last_hidden_state.shape[1] == batch["target_emb"].shape[1]
        return last_hidden_state # [bs, target_ids num, 768]

    def _calculate_similarity(self, batch, batch_idx):
        preds = self._get_embedding(batch) # output embedding of model
        gts = batch['target_emb'] ## embedding of context 
        assert preds.shape == gts.shape
        # target_mask를 기반으로 pad token 아닌 애들만 냅두기
        preds, gts = self._remove_pad_tokens(batch['target_mask'], preds, gts, batch_idx)
        # print(f"sim socres: {torch.inner(preds, gts)}")
        return torch.inner(preds, gts)

    def _remove_pad_tokens(self, target_mask, preds, gts, batch_idx):
        non_mask = torch.count_nonzero(target_mask, dim=1)
        # print(f"non mask {non_mask}")
        assert non_mask.shape[0] == preds.shape[0] == gts.shape[0]
        pred_list = []; gt_list = []
        # print(f"preds: {preds}")
        # print("#"*50)
        # print(f"gts: {gts}")
        for _non_mask, _pred, _gt in zip(non_mask, preds, gts):
            # print(f'[Before] non_mask: {_non_mask}\tpred: {_pred}\tgt: {_gt}\n\n')
            _pred = _pred[:_non_mask, :]
            _gt = _gt[:_non_mask, :]
            # print(f'[After] non_mask: {_non_mask}\tpred: {_pred}\tgt: {_gt}\n\n')
            pred_list.append(_pred); gt_list.append(_gt)
        # print("#"*50)
        # print(f"preds_list {pred_list}")
        # print("#"*50)
        # print(f"gt_list: {gt_list}")
        pred_length = [len(pred) for pred in pred_list]
        gt_length = [len(gt) for gt in gt_list]
        if batch_idx == 0:
           self.file.write(f'After Removing Pad Tokens..\npred_length: {pred_length}\ngt_length: {gt_length}\n\n')
        ## TODO 0번쨰 dim 사이즈 안맞는데 에러가 안나는지 확인 
        # print(pred_length)
        
        preds = torch.cat(pred_list, dim=0)
        # print(preds.shape)
        gts = torch.cat(gt_list, dim=0)
        assert preds.shape == gts.shape
        # assert False
        return preds, gts 

    def _calculate_loss(self, batch, batch_idx, ret_em=False):
        sim = self._calculate_similarity(batch, batch_idx)
        labels = torch.arange(sim.size(0)).long().to(self.device) ## [length of sim]
        # print(labels.shape) 
        loss = self.loss_fct(sim, labels)
        if ret_em:
            sub_em = self._calculate_sub_em(sim, labels, batch_idx)
            if batch_idx == 0:
               self.file.write(f"loss: {loss}\tem: {sub_em}\n")
               self.file.write('='*80)
            print(f'loss: {loss}\tem: {sub_em}')
            return loss, sub_em
        return loss, None

    def _get_end_tok_emb(self):
        _, end_emb = self._encode_sp("</s>")
        return end_emb

    def _tokenize(self, sen):
        _tok = self.tokenizer(sen, return_tensors='pt', add_special_tokens=False, max_length=self.hparams.max_context_length)
        _tok_decode = self.tokenizer.convert_ids_to_tokens(_tok['input_ids'][0])
        return _tok, _tok_decode

    def _encode_sp(self, sen):
        _tok, _tok_decode = self._tokenize(sen)
        _input_ids = _tok['input_ids'].to(self.device)
        _attention_mask = _tok["attention_mask"].to(self.device)
        model_ret = self.emb_enc(input_ids=_input_ids, attention_mask=_attention_mask, return_dict=True)
        last_hidden_state = model_ret['last_hidden_state'][0]
        #last_hidden_state = last_hidden_state.detach()
        return _tok_decode, last_hidden_state

    def _emb_enc_forward(self, title, context, end_emb):
        if context == "":
            context = title 
        else:
            context = " ".join([title, context])
        context = context.strip()

        # get embedding of context
        _tok_decoder, last_hidden_state = self._encode_sp(context)

        # check number of tokens of title 
        _, _title_tok_list = self._tokenize(title)
        _title_tok_num = len(_title_tok_list)
        # print(f"title_tok {_title_tok_num} ")
        assert _title_tok_list == _tok_decoder[:_title_tok_num]
        title_hidden_state = last_hidden_state[:_title_tok_num]
        assert len(title_hidden_state) == _title_tok_num 

        tokText_emb = {}
        for _tok, _emb in zip(_title_tok_list, title_hidden_state):
            tokText_emb[_tok] = _emb.unsqueeze(0).to(self.device) 
        tokText_emb["</s>"] = end_emb.to(self.device)
        return tokText_emb ## title embeddings sliced 

    def _get_target_embs(self, batch):
        target_emb_list = []; target_mask_list = []
        end_emb = self._get_end_tok_emb()
        for _title, _context in zip(batch['title'], batch['context']):
            tokText_emb = self._emb_enc_forward(_title, _context, end_emb)
            # tokText_emb = self._emb_enc_forward(_title, "", end_emb)
            target_text = list(tokText_emb.keys())
            target_emb = list(tokText_emb.values())
            # print(target_text)
            assert len(target_text) == len(target_emb)
            if len(target_text) <= self.hparams.max_output_length:
                leftover = self.hparams.max_output_length-len(target_text)
                target_mask = torch.tensor([1]*len(target_text) + [0]*leftover).to(self.device)
                target_text = target_text + [-1]*leftover
                target_emb = target_emb + [torch.tensor([-1]*target_emb[0].shape[-1]).unsqueeze(0).to(self.device)]*leftover
            else:
                target_mask = torch.tensor([1]*self.hparams.max_output_length).to(self.device)
                target_text = target_text[:self.hparams.max_output_length]
                target_emb = target_emb[:self.hparams.max_output_length]
            assert len(target_text) == len(target_emb) == len(target_mask)

            target_emb = torch.cat(target_emb, dim=0).to(self.device) #[max_output_length, 768] 
            # print(f"5737: {target_emb.shape}")
            target_emb_list.append(target_emb.unsqueeze(0))
            target_mask_list.append(target_mask.unsqueeze(0))
        target_emb = torch.cat(target_emb_list, dim=0) #모든 token 일자로 붙여서 한번에 넣어줌
        # print(f"5738: {target_emb.shape}")
        target_mask = torch.cat(target_mask_list, dim=0) # [batch_size, all token length ,1024]
        # print(target_emb.shape)
        # assert False
        return target_emb, target_mask
    

    def _calculate_sub_em(self, sim, labels, batch_idx):
        # print(f"sim: {sim.shape}") # [27,27]
        top_score = torch.topk(sim, 1)
        indices = top_score.indices.squeeze() 
        if batch_idx == 0:
           self.file.write(f'indices: {indices}\nlabels: {labels}\n\n')
        # print(f"indices: {indices}")
        # print(f"labels: {labels}")        
        correct = torch.eq(indices, labels)
        # print(f"correct {correct}")
        correct = correct.sum()
        total = len(indices)
        # assert False
        return correct/total*100

    def training_step(self, batch, batch_idx):
        target_emb, target_mask = self._get_target_embs(batch)
        batch['target_emb'] = target_emb
        batch['target_mask'] = target_mask 
        loss, _ = self._calculate_loss(batch, batch_idx) 
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

    def validation_step(self, batch, batch_idx):
        target_emb, target_mask = self._get_target_embs(batch)
        batch['target_emb'] = target_emb
        batch['target_mask'] = target_mask 
        loss, sub_em = self._calculate_loss(batch, batch_idx, ret_em=True)
        self.log(
            "val_loss",
            loss, 
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.val_loss.append(loss.cpu())
        self.val_em.append(sub_em.cpu())
        return loss 

    def validation_epoch_end(self, outputs):
        avg_loss = np.mean(np.array(self.val_loss))
        avg_em = np.mean(np.array(self.val_em))
        self.val_loss = []; self.val_em = []
        self.log(
            "val_avg_loss",
            avg_loss, 
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_em",
            avg_em, 
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return
 
    def get(self, input_ids, trie_dict=None):
        if trie_dict is None:
            trie_dict = self.trie
            print('Trie_dict is None!')
            sys.exit()
        return self._get_from_trie(input_ids, trie_dict)

    def _get_from_trie(self, input_ids, trie_dict):
        assert self.hparams.tree_type == "groupId"
        if len(input_ids) == 0:
            possible_GroupList = list(trie_dict.keys())
            tokIdList = self._get_tokIdList_from_groupIdList(possible_GroupList)
            return tokIdList
        else:
            curGroupId = self._get_groupId_from_tokId(input_ids[0])
            if curGroupId in list(trie_dict.keys()):
                return self._get_from_trie(input_ids[1:], trie_dict[curGroupId]) 
            else:
                return []

    def _get_tokIdList_from_groupIdList(self, groupIdList):
        assert self.hparams.tree_type == "groupId"
        tokIdList = []

        assert not self.hparams.max_beam_search 
        """
        # for groupId Tree with max beam search
        if self.hparams.max_beam_search:
            for groupId in groupIdList:
                max_tokId = self._get_max_tokId_from_tokIdList(self.groupId2tokId[groupId], score)
                tokIdList.append(max_tokId)
            return list(set(tokIdList))
        # for normal groupId Tree
        """
        for groupId in groupIdList:
            tokIdList.extend(self.groupId2tokId[groupId])
        return list(set(tokIdList))
   

    def _get_groupId_from_tokId(self, tokId):
        return self.tokId2groupId[tokId]

    def ids_to_text(self, _generated_ids, skip_special_tokens=True):
        generated_ids = []
        for _ids in _generated_ids:
            _ids = copy.deepcopy(_ids)
            _text = [self.tokId2tokText[_id] for _id in _ids]
            generated_ids.append(self.tokenizer.convert_tokens_to_ids(_text))
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True
        )
        text = self.lmap(str.strip, gen_text)
        if not skip_special_tokens:
            while "<pad>" in text:
               text = text.replace('<pad>', '')
        return text

    def _calculate_beam_score(self, scores, next_possible_tokens, _beam_score, _decoder_inputs_ids, leftover, first=False):
        t_df = {'ids': [], 'score': []}
        # print(scores)
        # print(len(next_possible_tokens[0]))
        # assert len(scores) == len(next_possible_tokens), f"len of scores {scores}, next_possible_tokens {next_possible_tokens}"
        for batch_id, (_score, _n_tokens) in enumerate(zip(scores, next_possible_tokens)):
            _score = _score[_n_tokens]
            # print(len(_score))
            # print(len(_n_tokens))
            # print(len(_score))
            assert len(_score) == len(_n_tokens) and len(_score) > 0
            top_scores = torch.topk(_score, min(len(_score), self.hparams.val_beam_size))
            p_beam_scores = _beam_score[batch_id]
            p_tokid = _decoder_inputs_ids[batch_id]
            for top_id, (_val, _ind) in enumerate(zip(top_scores.values, top_scores.indices)):
                _tokId = _n_tokens[_ind.item()]
                t_df['ids'].append(list(p_tokid)+list([_tokId])) ## list(p_tokid) ??? 
                t_df['score'].append(p_beam_scores+_val.item())
        # print(t_df)
        t_df = pd.DataFrame(t_df).sort_values(by=['score'], ascending=False)[:leftover]
        # print(t_df)
        return t_df


    def _test_step(self, batch, trie_dict):
        end_token = [False]*self.hparams.val_beam_size
        leftover = self.hparams.val_beam_size
        end_df = {'ids': [], 'beam_score': []}

        ## training task는 잘한다 
        target_emb, target_mask = self._get_target_embs(batch)
        # print(f"target_emb {target_emb.shape}")
        batch['target_emb'] = target_emb
        batch['target_mask'] = target_mask 
        loss, sub_em = self._calculate_loss(batch,0, ret_em=True)
        # print(sub_em)

        # 처음에는 하나만 넣고 그 다음부터 val_beam_size 만큼
        _decoder_inputs_embeds = [[self.pad_emb]]
        _decoder_inputs_ids = [[self.pad_tokenid]]
        #########


        # _decoder_inputs_embeds = [[torch.tensor(self.tokId2Emb[self.tokId2groupId[tok['input_ids'][0][0]]]).unsqueeze(0).to(self.device)]]


        ## with first token answer## -> decoding은 정확하게 함 (by trie)
        # tok,tok_text = self._tokenize(batch['title'])
        # _decoder_inputs_embeds = [[self.pad_emb],[self.pad_emb]]

        # _decoder_inputs_embeds = [[self.pad_emb],[target_emb[0][0].unsqueeze(0).to(self.device)]]
        # _decoder_inputs_ids = [[self.pad_tokenid,self.toktext2tokidlist[tok_text[0]][0]]]
        ###########################
        assert _decoder_inputs_embeds[0][0].shape == self.pad_emb.shape, f"{_decoder_inputs_embeds[0][0].shape} != {self.pad_emb.shape}"
        # print(f"_decoer_inputs_embeds before {_decoder_inputs_embeds.shape}")
        _beam_score = [0]

        #original
        # _dec_input = torch.cat([torch.cat(embs, dim=0).unsqueeze(0).to(self.device) for embs in _decoder_inputs_embeds], dim=0) ### 
        #new
        _dec_input = torch.cat([torch.cat(embs, dim=0).unsqueeze(0).to(self.device) for embs in _decoder_inputs_embeds], dim=1) ### 

        # print(f"test dec_input: {_dec_input.shape}")
        # print(f"before enc_input {batch['source_ids'].shape}")
        _enc_input = torch.cat([batch['source_ids']], dim=0)
        # print(f"after enc_input {_enc_input.shape}")
        _enc_attention = torch.cat([batch['source_mask']], dim=0) 
        # print(_enc_input.size())

        # print(_dec_input.size())
        model_output = self(
            input_ids=_enc_input,
            attention_mask=_enc_attention,
            decoder_inputs_embeds=_dec_input,
        ).last_hidden_state # [bs, 1, 1024]
        # print(f"before: model output {model_output.shape}") 
        model_output = model_output[:, -1:, :] # [bs, 1, 1024]
        # print(f"after: model output {model_output.shape}")





        scores = torch.inner(model_output.to(self.device), self.total_emb.to(self.device))
        scores = nn.functional.log_softmax(scores, dim=2)
        scores = scores.squeeze(1)
        next_possible_tokens = np.array([self.get(ids, trie_dict) for ids in _decoder_inputs_ids]) # 
        # print(next_possible_tokens)
        # print(scores)
    
        t_df = self._calculate_beam_score(scores, next_possible_tokens, _beam_score, _decoder_inputs_ids, leftover, first=True)

        _decoder_inputs_ids = [] 
        _beam_score = [] 
        _decoder_inputs_embeds = []
        for bid, (ids, score) in enumerate(zip(t_df['ids'], t_df['score'])):
            if ids[-1] == self.end_tokenid:
                end_df['ids'].append(ids)
                end_df['beam_score'].append(score/len(ids))
                end_token[bid] = True 
                leftover -= 1
            else:
                _decoder_inputs_embeds.append([self.tokEmbList[_id] for _id in ids])
                _decoder_inputs_ids.append(ids)
                _beam_score.append(score)

################### used when beam size is larger than 1 ###########################
        while leftover > 0: ## left over 가 없어질때까지 위의 step 반복 
            _dec_input = torch.cat([torch.cat(embs, dim=0).unsqueeze(0).to(self.device) for embs in _decoder_inputs_embeds], dim=0) 
            _enc_input = torch.cat([batch['source_ids']]*leftover, dim=0)
            _enc_attention = torch.cat([batch['source_mask']]*leftover, dim=0)
            assert _dec_input.shape[0] == _enc_input.shape[0] == _enc_attention.shape[0], f"dec_input shape{_dec_input.shape}, _enc_input shape {_enc_input.shape}, _enc_attention shape {_enc_attention.shape}"

            model_output = self(
                input_ids=_enc_input,
                attention_mask=_enc_attention,
                decoder_inputs_embeds=_dec_input,
            ).last_hidden_state # [bs, 1, 768]
            # find the closest embedding
            assert model_output.shape[0] == leftover, f"model_output shape: {model_output.shape}"
            model_output = model_output[:, -1:, :]
            scores = torch.inner(model_output.to(self.device), self.total_emb.to(self.device))
            scores = nn.functional.log_softmax(scores, dim=2)
            assert scores.shape[1] == 1
            scores = scores.squeeze(1)

            next_possible_tokens = np.array([self.get(ids, trie_dict) for ids in _decoder_inputs_ids])
            assert len(next_possible_tokens) == leftover 

            t_df = self._calculate_beam_score(scores, next_possible_tokens, _beam_score, _decoder_inputs_ids, leftover)

            _decoder_inputs_ids = [] 
            _beam_score = [] 
            _decoder_inputs_embeds = []
            for bid, (ids, score) in enumerate(zip(t_df['ids'], t_df['score'])):
                if ids[-1] == self.end_tokenid:
                    end_df['ids'].append(ids)
                    end_df['beam_score'].append(score/len(ids))
                    end_token[bid] = True 
                    leftover -= 1
                else:
                    _decoder_inputs_embeds.append([self.tokEmbList[_id] for _id in ids])
                    _decoder_inputs_ids.append(ids)
                    _beam_score.append(score)

        # detokenize _decoder_inputs_ids
        for elem in end_df['ids']: assert end_df['ids'].count(elem) == 1 
        preds = [self.ids_to_text(end_df['ids'])]
        return preds[0], end_df 

    # TODO: batch size > 1
    # batch -> {"source_ids", "source_mask", "title", "context", "input", "output"}
    def test_step(self, batch, batch_idx):
        preds = []; scores = []
        _trie_dict = copy.deepcopy(self.trie)
        while len(preds) < self.hparams.val_beam_size:
            
            _preds, _end_df = self._test_step(batch, _trie_dict)
            remove_tokId = []
            for _pred, _ids, _score in zip(_preds, _end_df['ids'], _end_df['beam_score']):
                if _pred not in preds:
                    preds.append(_pred)
                    scores.append(_score)
                    remove_tokId.append([self.tokId2groupId[el] for el in _ids])
            _upper_ids = []
            _trie_dict = self._remove_prev_from_trie(_trie_dict, remove_tokId)
        
        preds = list(pd.DataFrame({'preds': preds, 'scores': scores}).sort_values(by=['scores'], ascending=False)[:self.hparams.val_beam_size]['preds'])
        # print(pd.DataFrame({'preds': preds, 'scores': scores}).sort_values(by=['scores'], ascending=False)[:self.hparams.val_beam_size])

        assert len(batch['output']) ==len(batch['input'])== 1
        self.test_em_score_list.append(self._calculate_em(preds[0], batch['output'][0]))
        self.test_recall_score_list.append(self._calculate_recall(preds, batch['output'][0]))
        self.test_input_list.extend(batch['input']) 
        self.test_gt_list.extend(batch['output']) 
        self.test_pred_list.append(preds)

        print('='*80)
        print(f"query: {batch['input']}")
        print(f'preds: {preds}')
        print(f"gt: {batch['output']}")
        print(f'EM: {self.test_em_score_list[-1]}\tRECALL: {self.test_recall_score_list[-1]}')
        print('='*80)

        self._save_test()
        return 

    def _save_test(self, epoch_end=False):
        os.makedirs(self.hparams.output_dir, exist_ok=True)
        _input = self.gather_list(self.test_input_list)
        _gt = self.gather_list(self.test_gt_list)
        _pred = self.gather_list(self.test_pred_list)
        _em = self.gather_list(self.test_em_score_list)
        _recall = self.gather_list(self.test_recall_score_list)
        assert len(_input) == len(_gt) == len(_pred) == len(_em) == len(_recall) 
        if self.print:
            with open(self.test_save_name, "w") as f:
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
            if epoch_end:
                print(
                    f"Saving in {self.test_save_name}!\nnumber of elements: {len(_input)}"
                )
                print(f"EM: {np.array(_em).mean()}")
                print(f"Recall: {np.array(_recall).mean()}")

    def test_epoch_end(self, outputs):
        self._save_test(epoch_end=True)

    def _get_params(self, no_decay, find_no_decay):
        ret_list = []
        if find_no_decay:
            for model in [self.model, self.emb_enc]:
                for n, p in model.named_parameters():
                    #if not torch.is_tensor(p): continue
                    if any(nd in n for nd in no_decay):
                        ret_list.append(p)
        else:            
            for model in [self.model, self.emb_enc]:
                for n, p in model.named_parameters():
                    #if not torch.is_tensor(p): continue
                    if not any(nd in n for nd in no_decay):
                        ret_list.append(p)
        return ret_list


    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": self._get_params(no_decay, find_no_decay=False),
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": self._get_params(no_decay, find_no_decay=True),
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
        # if not os.path.exists(save_path): os.makedirs(save_path, exist_ok=True)
        # if not os.path.exists(os.path.join(save_path, "model")): os.makedirs(os.path.join(save_path, "model"), exist_ok=True)
        # if not os.path.exists(os.path.join(save_path, "emb_enc")): os.makedirs(os.path.join(save_path, "emb_enc"), exist_ok=True)
        if self.global_rank==0:
            self.tokenizer.save_pretrained(save_path)
            self.model.save_pretrained(os.path.join(save_path, "model"))
            self.emb_enc.save_pretrained(os.path.join(save_path, "emb_enc"))
        

        if self.hparams.periflow:
            assert False
            target_path = save_path
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

class T5JointTuner_global(T5JointTuner):
    def __init__(self, args):
        super().__init__(args)
        self.contrastive_negative_selection = args.wandb_run_name.split('_')[-1]
    def _get_embedding(self, batch):            
        last_hidden_state = self(
                            input_ids=batch["source_ids"], 
                            attention_mask=batch["source_mask"], 
                            decoder_inputs_embeds=batch["target_emb"],
                            decoder_attention_mask=batch["target_mask"],
                            ).last_hidden_state
        assert last_hidden_state.shape[1] == batch["target_emb"].shape[1]
        return last_hidden_state # [bs, target_ids num, 768]

    def _calculate_similarity_global(self, batch, batch_idx):
        preds = batch['hidden_states'] # output embedding of model
        gts = batch['target_emb'] 
        assert preds.shape == gts.shape
        # target_mask를 기반으로 pad token 아닌 애들만 냅두기
        preds, gts = self._remove_pad_tokens(batch['target_mask'], preds, gts, batch_idx)
        return torch.inner(preds, gts)

    
    def _calculate_loss_global(self, batch, batch_idx, ret_em=False):

        world_size = self.trainer.world_size
############################################################################
        attention_mask_to_send = batch['target_mask'].detach()
        targets_to_send = batch['target_emb'].detach()
        hidden_states = self._get_embedding(batch) ## npm dir mㄷlm line 275 
        # print(f"target mask {attention_mask_to_send.unsqueeze(-1).shape}")
        # print(f"target emb {targets_to_send.shape}")
        # print(f"hidden state {hidden_states.shape}")
        in_tensor = torch.cat([
            attention_mask_to_send.unsqueeze(-1),
            targets_to_send,
            hidden_states], -1) # [bs, length, 2+hidden]
        # print(f"in tensor {in_tensor.shape}")
        # 모든 프로세스의 tensor(in_tensor) 를 모든 프로세스의 tensor_list(out_tensor) 에 복사합니다.
        out_tensor = [torch.zeros_like(in_tensor) for _ in range(torch.distributed.get_world_size())]
        out_tensor = my_all_gather2(out_tensor, in_tensor)
        out_tensor = torch.stack(out_tensor, 0) # [ws, bs, length, hidden+2]
        # print(f"out tensor {out_tensor.shape}")
        if self.contrastive_negative_selection == "half":
            H = self.trainer.world_size // 2
            out_tensor = out_tensor[:H] if self.global_rank < H else out_tensor[H:]
            # local_rank = self.global_rank if self.global_rank < H else self.global_rank - H
            # world_size = world_size // 2

        elif self.contrastive_negative_selection == "quarter":
            H = self.trainer.world_size // 4
            if self.global_rank < H:
                out_tensor = out_tensor[:H]
                # local_rank = self.global_rank
            elif self.global_rank < 2*H:
                out_tensor = out_tensor[H:2*H]
                # local_rank = self.global_rank - H
            elif self.global_rank < 3*H:
                out_tensor = out_tensor[2*H:3*H]
                # local_rank = self.global_rank - 2*H
            else:
                out_tensor = out_tensor[3*H:]
                # local_rank = self.global_rank - 3*H
            # world_size = world_size // 4
        assert out_tensor.shape[-1]==1 + 2*hidden_states.shape[-1] , print(f"out tensor {out_tensor.shape}, hidden states {hidden_states.shape}")

        all_attention_mask = out_tensor[:, :, :, 0].contiguous()
        all_targets = out_tensor[:, :, :, 1:1025].contiguous()
        all_hidden_states = out_tensor[:, :, :, 1025:].contiguous()

        all_attention_mask = all_attention_mask.reshape(-1, all_attention_mask.shape[-1])
        all_targets = all_targets.reshape(-1, all_targets.shape[-2] ,all_targets.shape[-1])
        all_hidden_states = all_hidden_states.reshape(-1, all_hidden_states.shape[-2], all_hidden_states.shape[-1])

        # all_targets = all_targets.int()
        # print(f"all attention mask {all_attention_mask.shape}")
        # print(f"all_targets {all_targets.shape}")
        # print(f"all_hidden_states {all_hidden_states.shape}")

        batch['target_mask'] = all_attention_mask
        batch['target_emb'] = all_targets
        batch['hidden_states'] = all_hidden_states

        
        # from now on, we don't really need to distinguish batch size and length
        # so do reshape
        # hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        # all_hidden_states = all_hidden_states.reshape(-1, all_hidden_states.shape[-1])
        # # targets = targets.reshape(-1)
        # all_targets = all_targets.reshape(-1)

        # print(f"all attention mask {all_attention_mask.shape}")
        # print(f"all_targets {all_targets.shape}")
        # print(f"all_hidden_states {all_hidden_states.shape}")


#####################################################################################
        ## sim, label 구성 
        sim = self._calculate_similarity_global(batch, batch_idx)
        # print(sim.size())
        labels = torch.arange(sim.size(0)).long().to(self.device) ## total batch 
        # print(labels)
        # assert False, f"{labels} \n {sim.size()} \n {sim.size(0)}"
        loss = self.loss_fct(sim, labels)

        if ret_em:
            sub_em = self._calculate_sub_em(sim, labels, batch_idx)
            if batch_idx == 0:
               self.file.write(f"loss: {loss}\tem: {sub_em}\n")
               self.file.write('='*80)
            print(f'loss: {loss}\tem: {sub_em}')
            return loss, sub_em
        return loss, None
    
    def training_step(self, batch, batch_idx):
        target_emb, target_mask = self._get_target_embs(batch)
        batch['target_emb'] = target_emb
        batch['target_mask'] = target_mask 
        # total_batch = self.all_gather(batch,sync_grads=True)

        loss, sub_em = self._calculate_loss_global(batch, batch_idx)
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
        
class T5MeanTuner(T5grTuner):
    def __init__(self, args):
        super(T5MeanTuner, self).__init__(args)
        assert self.hparams.train_c_emb == False
        assert self.hparams.tree_type == "groupId"

        self.corpus_EmbMean = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.corpus2EmbMean), 'rb'))
        self.totalEmbMean = torch.cat([torch.tensor(el).unsqueeze(0) for el in self.corpus_EmbMean.values()], dim=0)
        if self.hparams.fp16:
            self.totalEmbMean = self.totalEmbMean.half()
        self.config.update({"corpus_EmbMean": self.corpus_EmbMean})
        print(f"=== Shape of TotalEmbMean: {self.totalEmbMean.shape}")

        if self.hparams.do_train:
            self.model = split_T5.from_pretrained(
                self.hparams.model_name_or_path, config=self.config
            )
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.hparams.tokenizer_name_or_path
            )
            self.em_score_list = []; self.recall_score_list = [];
            self.first_em_score_list = []; self.first_recall_score_list = []

        if self.hparams.do_test:
            assert False

        self.loss_fct = nn.CrossEntropyLoss()

    def _get_dataset(self, split):
        dataset = MEANDataset(
            tokenizer=self.tokenizer,
            split=split,
            hparams=self.hparams,
            tokid2emb=self.contextualized_tokid2emb,
            corpus2EmbMean=self.corpus_EmbMean
        )
        return dataset

    def training_step(self, batch, batch_idx):
        first_loss, lm_loss = self._loss(batch)
        self.log(
            "title_loss",
            first_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "lm_loss",
            lm_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True
        )
        return first_loss+lm_loss 

    """
    different loss for the first token
    """
    def _loss(self, batch):
        # loss for lm tokens (second ~)
        lm_labels = copy.deepcopy(batch["target_ids"])
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        lm_labels[:, 0] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
            return_dict=False
        )
        logits = outputs.logits
        lm_loss = outputs[0]

        # loss for first token
        c_label = batch["c_label"].to(self.device)
        if self.hparams.fp16:
            c_label = c_label.half()
        first_logits = logits["first_logits"].squeeze(1)
        first_loss = self.loss_fct(first_logits.float(), c_label.float())

        return first_loss, lm_loss 

    def get(self, batch_id, input_ids, score, trie_dict=None):
        assert input_ids[0] == 0
        if trie_dict is None:
            trie_dict = self.trie 
        input_ids = input_ids[:, 2:]
        return self._get_from_trie(input_ids, trie_dict[0], score)

    def _get_from_trie(self, input_ids, trie_dict, score):
        if self.hparams.tree_type == "groupId":
            if len(input_ids) == 0:
                possible_GroupList = list(trie_dict.keys())
                tokIdList = self._get_tokIdList_from_groupIdList(possible_GroupList, score)
                return tokIdList
            else:
                curGroupId = self._get_groupId_from_tokId(input_ids[0])
                if curGroupId in list(trie_dict.keys()):
                    return self._get_from_trie(input_ids[1:], trie_dict[curGroupId], score) 
                else:
                    return [] 


    def _calculate_first_score(self, batch):
        outputs = self(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=None,
            decoder_input_ids=torch.tensor([[self.tokenizer.pad_token_id]]*(batch["source_ids"].shape[0])).to(self.device),
            return_dict=False
        )
        logits = outputs.logits["first_output"].squeeze(1)
        assert logits.shape[-1] == self.totalEmbMean.shape[-1], f"logits: {logits}"

        mean_emb = torch.topk(torch.inner(logits.to(self.device), self.totalEmbMean.to(self.device)), self.hparams.val_beam_size)
        #print(mean_emb)
        dec_embs = []
        _gt_mean_emb = torch.nonzero(batch["c_label"])
        gt_mean_emb = [] 
        for i in range(len(_gt_mean_emb)):
            assert _gt_mean_emb[i][0] == i 
            gt_mean_emb.append(_gt_mean_emb[i][1])
        assert len(gt_mean_emb) == len(mean_emb.indices), f"gr_mean_emb: {len(gr_mean_emb)} // mean_emb: {len(mean_emb.indices)}"
        for i, b_ind in enumerate(mean_emb.indices):
            _gt_mean_emb = gt_mean_emb[i]
            dec_embs.append(self.totalEmbMean[b_ind[0]].unsqueeze(0))
            if _gt_mean_emb == b_ind[0]:
                self.first_em_score_list.append(100)
            else:
                self.first_em_score_list.append(0)
            if _gt_mean_emb in b_ind:
                self.first_recall_score_list.append(100)
            else:
                self.first_recall_score_list.append(0)

        return torch.cat(dec_embs, dim=0).to(self.device)

    def _val_step(self, batch, batch_idx, return_elem=False):
        raise NotImplementedError(f"BUG! Check how to use decoder_inputs_embeds when generating")
        dec_input_embs = self._calculate_first_score(batch).unsqueeze(1)
        first_token_embs = [torch.tensor(self.contextualized_tokid2emb[0]).to(self.device).unsqueeze(0)]*(dec_input_embs.shape[0])
        first_token_embs = torch.cat(first_token_embs, dim=0).unsqueeze(1)
        assert dec_input_embs.shape == first_token_embs.shape
        dec_input_embs = torch.cat([first_token_embs, dec_input_embs], dim=1)
        # calculates recall and em -> returns the list of each score
        _generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            decoder_inputs_embeds=dec_input_embs,
            decoder_attention_mask=batch["target_mask"],
            max_length=self.hparams.max_output_length,
            num_beams=self.hparams.val_beam_size,
            num_return_sequences=self.hparams.val_beam_size,
            prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get(
                batch_id, sent.tolist(), scores
            ),
            early_stopping=True,
        )

        _generated_text = self.ids_to_text(_generated_ids)
        print(f'%%% TEXT: {_generated_text}')

        inum = len(_generated_ids) // self.hparams.val_beam_size
        assert inum == len(batch["output"])
        generated_text = [
            _generated_text[
                i * self.hparams.val_beam_size : (i + 1) * self.hparams.val_beam_size
            ]
            for i in range(inum)
        ]
        generated_ids = [
            _generated_ids[
               i * self.hparams.val_beam_size : (i+1) * self.hparams.val_beam_size
            ].detach().cpu().numpy().tolist()
            for i in range(inum)
        ]

        em_list, recall_list = self.calculate_scores(
            generated_text, batch["output"], batch["input"], batch_idx
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
                "gt_tok": list(batch["target_ids"].detach().cpu().numpy().tolist()),
                "pred": list(generated_text),
                "pred_tok": list(generated_ids),
                "em": list(em_list),
                "recall": list(recall_list),
            }
        else:
            return em_list, recall_list


    def _test_step(self, batch, batch_idx, return_elem=False):
        ret_dict = {"input": [], "gt": [], "gt_tok": [], "pred": [], "pred_tok": [], "em": [], "recall": []}
        return ret_dict


class T5Entail(T5BaseClass): 
    def __init__(self, args):
        super(T5Entail, self).__init__()
        if print: print(f'==========T5Entail===========')
        self.save_hyperparameters(args)
        if self.hparams.do_train:         
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.hparams.tokenizer_name_or_path
            )
        if self.hparams.do_test:
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.test_model_path)
        sp_tokens = ["[Ee]", "[Es]", "[Ve]", "[Vs]", "[HYPOT]", "[QUESTION]", "[ANSWER]", "[BECAUSE]", "[INFER]", "[AND]"]
        self.tokenizer.add_tokens(sp_tokens, special_tokens=True)
 
        if np_only(self.hparams):
            self.np_only = True 
        else:
            self.np_only = False

        ### Shared => Vanilla + NPD ###
        if self.np_only:
            self.shared_tokId2Emb = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), "rb"))
            self.shared_tokId2tokText = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2tokText), "rb"))
            self.shared_tokId2groupId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2groupId), "rb"))
            self.shared_groupId2tokId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.groupId2tokIdList), "rb"))
            self.contextualized_emb_num = len(self.shared_tokId2Emb.keys())
            self.groupId2tokId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.groupId2tokIdList), "rb"))
            self.contextualized_tokid2emb = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), "rb"))
            self.modelId2sharedId = None
            if self.hparams.model_type == "hyper-mem-np-only":
               assert self.shared_tokId2tokText[3] == "[Es]"
               assert self.shared_tokId2tokText[4] == "[Ee]"
               self.es_tokid = 3; self.ee_tokid = 4
            elif self.hparams.model_type == "hyper-wo-vd":
               assert self.shared_tokId2tokText[32101] == "[Es]"
               assert self.shared_tokId2tokText[32100] == "[Ee]"
               self.es_tokid = 32101; self.ee_tokid = 32100
               self.vs_tokid = 32103; self.ve_tokid = 32102
               self.hypot_tokid = 32104; self.question_tokid = 32105; self.answer_tokid = 32106; self.because_tokid = 32107; self.and_tokid = 32109; self.infer_tokid = 32108
               self.vanilla_tokid_list = [i for i in range(2,32000)] 

            self.vd_mask=None; self.npd_mask=None
        else:
            if os.path.exists(os.path.join(self.hparams.output_dir, "shared.modelId2sharedId.pickle")):
               self.shared_tokId2Emb = pickle.load(open(os.path.join(self.hparams.output_dir, "shared.tokId2emb.pickle"), "rb"))
               self.shared_tokId2tokText = pickle.load(open(os.path.join(self.hparams.output_dir, "shared.tokId2tokText.pickle"), "rb"))
               self.shared_tokId2groupId = pickle.load(open(os.path.join(self.hparams.output_dir, "shared.tokId2groupId.pickle"), "rb"))
               self.shared_groupId2tokId = pickle.load(open(os.path.join(self.hparams.output_dir, "shared.groupId2tokId.pickle"), "rb"))
               self.modelId2sharedId = pickle.load(open(os.path.join(self.hparams.output_dir, "shared.modelId2sharedId.pickle"), "rb"))
               self.groupId2tokId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.groupId2tokIdList), "rb"))
               self.contextualized_tokid2emb = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), "rb"))
            else:
               self.contextualized_tokid2emb = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), "rb"))
               self.groupId2tokId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.groupId2tokIdList), "rb"))
               self.tokId2groupId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2groupId), "rb"))
               self.tokId2tokText = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2tokText), "rb"))
               self.model_tokid2emb = self._get_model_tokid2emb() 
               self.model_tokid2text = {}
               for i in range(len(self.tokenizer)):
                  text = self.tokenizer.convert_ids_to_tokens(i)
                  self.model_tokid2text[i] = text 

               self.shared_tokId2Emb, self.shared_tokId2tokText, self.shared_tokId2groupId, self.shared_groupId2tokId, self.modelId2sharedId = self.combine_tokid2emb()

            sp_tokens_list = self.tokenizer.batch_encode_plus(sp_tokens)["input_ids"]
            self.sp_tokens_list = sp_tokens_list
            self.ee_tokid = self.modelId2sharedId[sp_tokens_list[0][0]]
            self.es_tokid = self.modelId2sharedId[sp_tokens_list[1][0]]
            self.ve_tokid = self.modelId2sharedId[sp_tokens_list[2][0]]
            self.vs_tokid = self.modelId2sharedId[sp_tokens_list[3][0]]
            self.hypot_tokid = self.modelId2sharedId[sp_tokens_list[4][0]]
            self.question_tokid = self.modelId2sharedId[sp_tokens_list[5][0]]
            self.answer_tokid = self.modelId2sharedId[sp_tokens_list[6][0]]
            self.because_tokid = self.modelId2sharedId[sp_tokens_list[7][0]]
            self.infer_tokid = self.modelId2sharedId[sp_tokens_list[8][0]]
            self.and_tokid = self.modelId2sharedId[sp_tokens_list[9][0]]

            if self.print:
               print(f'\n\nee: {self.ee_tokid}\nes: {self.es_tokid}\nve: {self.ve_tokid}\nvs: {self.vs_tokid}\nhypot: {self.hypot_tokid}\nquestion: {self.question_tokid}\nanswer: {self.answer_tokid}\nbecause: {self.because_tokid}\ninfer: {self.infer_tokid}\nand: {self.and_tokid}\n\n')
            
            self.vanilla_tokid_list = [self.modelId2sharedId[i] for i in range(2,32000)] 
            assert len(self.contextualized_tokid2emb) == self.modelId2sharedId[3], f"len(self.contextualized_tokid2emb): {len(self.contextualized_tokid2emb)}\nself.modelId2sharedId[3]: {self.modelId2sharedId[3]}"
            self.vd_mask = [1,1,1]+[0]*(len(self.contextualized_tokid2emb)-3)+[1]*(len(self.tokenizer)-3-6)+[0]*6 # remove sp except for ee, es, ve, vs 
            self.npd_mask = [1,1,1]+[1]*(len(self.contextualized_tokid2emb)-3)+[0]*(len(self.tokenizer)-3-6)+[0]*6 
            self.contextualized_emb_num = len(self.shared_tokId2Emb.keys())
            assert len(self.vd_mask) == len(self.npd_mask) == self.contextualized_emb_num
            assert self.contextualized_emb_num == len(self.tokenizer)+len(self.contextualized_tokid2emb)-3, f"Contextualized Emb Num: {self.contextualized_emb_num}\tlen(self.tokenizer): {len(self.tokenizer)}\tlen(self.contextualized_tokid2emb): {len(self.contextualized_tokid2emb)}\nadd => {len(self.tokenizer)+len(self.contextualized_tokid2emb)-3}"
       
        if self.hparams.dev_input2output is not None:
            self.dev_input2output = json.load(open(os.path.join(self.hparams.dataset, self.hparams.dev_input2output), "rb"))

        if self.hparams.cluster_num != -1:
            self.tokId2clusterId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2clusterId), 'rb'))
        else:
            self.tokId2clusterId = None 
        
        self.corpus_tokenList = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.corpus2tokIdList), "rb"))
        self.trie_dict = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tree_path), "rb"))
        
        print(f"$$$$$$ # of embedding: {self.contextualized_emb_num}")
        self.config = T5Config.from_pretrained(self.hparams.model_name_or_path)
        self.config.update({"fp16": self.hparams.fp16})
        self.config.update({"train_c_emb": self.hparams.train_c_emb}) 
        self.config.update({"do_test": self.hparams.do_test}) 
        self.config.update(
            {"contextualized_emb_num": self.contextualized_emb_num}
        )
        self.config.update({"freeze_vocab_emb": self.hparams.freeze_vocab_emb})
        if self.np_only: 
            self.config.update(
               {"contextualized_file": os.path.join(self.hparams.dataset, self.hparams.contextualized_file)} 
            )  # tokId_emb.pickle
            if self.hparams.model_type == "hyper-mem-np-only":
               self.config.update({"vanilla_emb_num": 0})
            elif self.hparams.model_type == "hyper-wo-vd":
               self.config.update({'vanilla_emb_num': len(self.tokenizer)})
            else:
               assert False
        else:
            self.config.update(
               {"contextualized_file": os.path.join(self.hparams.output_dir, "shared.tokId2emb.pickle")}
            )  # tokId_emb.pickle
            self.config.update({"vanilla_emb_num": len(self.tokenizer)})
        self.save_epoch = []

        # If in training mode, load ckpt for training
        if self.hparams.do_train:
            if self.hparams.train_c_emb:
                if self.hparams.w_enc:
                     self.model = entail_T5_w_enc.from_pretrained(
                              self.hparams.model_name_or_path, config=self.config, ignore_mismatched_sizes=True
                           )
                else:
                     self.model = entail_T5.from_pretrained(
                              self.hparams.model_name_or_path, config=self.config, ignore_mismatched_sizes=True
                          )
            else:
                if self.hparams.w_enc:
                     self.model = entail_T5_w_enc.from_pretrained(
                              self.hparams.model_name_or_path, config=self.config
                           )
                else:
                     self.model = entail_T5.from_pretrained(
                         self.hparams.model_name_or_path, config=self.config
                     )

            if self.print:
                print(f"@@@ Loading Model from {self.hparams.model_name_or_path}")
                print(f'@@@ Loading decoder embedding: {self.hparams.embedding_model}')
            self.em_score_list = []
            self.total_f1_score_list = []
            self.vs_f1_score_list = []
            self.es_f1_score_list = []


        # If in testing mode, load ckpt for inference
        if self.hparams.do_test:
            if self.hparams.w_enc:
                self.model = entail_T5_w_enc.from_pretrained(
                           self.hparams.test_model_path, config=self.config#, ignore_mismatched_sizes=True
                      )
            else:
                self.model = entail_T5.from_pretrained(
                   self.hparams.test_model_path, config=self.config#, ignore_mismatched_sizes=True
                )
            if self.print:
                print(f"@@@ Loading Model from {self.hparams.test_model_path}")
          
            ### TODO: check
            """
            if self.hparams.tie_enc_dec_vocab:
               if self.hparams.w_enc:
                   self.model.couple_total_vocab()
               else:
                   self.model.couple_encoder_decoder_model_vocab()
            """

            self.test_save_name = os.path.join(self.hparams.output_dir, f"{self.hparams.test_name}_{self.hparams.tree_type}_result_beam{self.hparams.val_beam_size}.json")               
            print(f'@@@ Initialize Test!!')
            self.test_input_list = []
            self.test_gt_list = []
            self.test_gt_tok_list = []
            self.test_pred_list = []
            self.test_pred_tok_list = []
            self.test_em_score_list = []
            self.test_total_f1_score_list = []
            self.test_vs_f1_score_list = []
            self.test_es_f1_score_list = []

        if self.hparams.freeze_encoder:
            if self.print:
                print(f"@@@ Freeze Encoder!")
                encoder = self.model.get_encoder()
                for n, p in encoder.named_parameters():
                    p.requires_grad=False
        
        self.cnt_over = 0
        self.len_test_dataset = len(self.test_dataloader())

    def combine_tokid2emb(self):
        shared_tokId2Emb = copy.deepcopy(self.contextualized_tokid2emb)
        shared_tokId2tokText = copy.deepcopy(self.tokId2tokText)
        shared_tokId2groupId = copy.deepcopy(self.tokId2groupId)
        shared_groupId2tokId = copy.deepcopy(self.groupId2tokId)
        modelId2sharedId = {}

        tokId = len(shared_tokId2Emb); groupId = len(shared_groupId2tokId)

        for id, text in self.model_tokid2text.items():
            if id==0 or id==1 or id==2:
               modelId2sharedId[id] = id  
               continue
            modelId2sharedId[id] = tokId 
            if text not in shared_tokId2tokText.values():
               assert tokId not in shared_tokId2Emb 
               assert groupId not in shared_groupId2tokId 
               shared_tokId2Emb[tokId] = self.model_tokid2emb[id]
               shared_tokId2tokText[tokId] = text 
               shared_tokId2groupId[tokId] = groupId 
               shared_groupId2tokId[groupId] = [tokId] 
               tokId +=1; groupId += 1
            else:
               assert tokId not in shared_tokId2Emb
               gId = self.text2gId(self.tokId2tokText, self.tokId2groupId, text)
               assert gId in shared_groupId2tokId
               shared_tokId2Emb[tokId] = self.model_tokid2emb[id]
               shared_tokId2tokText[tokId] = text 
               shared_tokId2groupId[tokId] = gId 
               shared_groupId2tokId[gId].append(tokId)
               tokId += 1

        print(f"# of Contextualized Token Embedding: {len(self.contextualized_tokid2emb)}")
        print(f"# of Model Token Embedding: {len(self.model_tokid2emb)}")
        print(f"# of Shared Token Embedding: {len(shared_tokId2Emb)}")

        os.makedirs(self.hparams.output_dir, exist_ok=True)
        with open(os.path.join(self.hparams.output_dir, "shared.tokId2emb.pickle"), "wb") as f:
           pickle.dump(shared_tokId2Emb, f)
        with open(os.path.join(self.hparams.output_dir, "shared.tokId2tokText.pickle"), "wb") as f:
           pickle.dump(shared_tokId2tokText, f)
        with open(os.path.join(self.hparams.output_dir, "shared.tokId2groupId.pickle"), "wb") as f:
           pickle.dump(shared_tokId2groupId, f)
        with open(os.path.join(self.hparams.output_dir, "shared.groupId2tokId.pickle"), "wb") as f:
           pickle.dump(shared_groupId2tokId, f)
        with open(os.path.join(self.hparams.output_dir, "shared.modelId2sharedId.pickle"), "wb") as f:
           pickle.dump(modelId2sharedId, f)

        return shared_tokId2Emb, shared_tokId2tokText, shared_tokId2groupId, shared_groupId2tokId, modelId2sharedId

    def text2gId(self, tokId2tokText, tokId2groupId, text):
        for t_id, t_text in tokId2tokText.items():
           if t_text == text:
               return tokId2groupId[t_id]

    def _get_model_tokid2emb(self):
        #model = T5Model.from_pretrained(self.hparams.model_name_or_path)    
        #vocab_weight = model.state_dict()["shared.weight"]
        model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        vocab_weight = model.encoder.embed_tokens.weight # 32110
        print(f'# of vocab => {len(self.tokenizer)}')
        vocab_num = len(self.tokenizer) #vocab_weight.size()[0]
        vocab_id2emb = {}
        for id in range(vocab_num):
           emb = vocab_weight[id].detach().cpu().numpy()
           vocab_id2emb[id] = emb
        return vocab_id2emb

    def _get_dataset(self, split):
        dataset = ENTAILDataset(
            tokenizer=self.tokenizer,
            split=split,
            hparams=self.hparams,
            ee_tokid=self.ee_tokid,#self.sp_tokens_list[0][0],
            es_tokid=self.es_tokid,#self.sp_tokens_list[1][0],
            corpus_tokenList=self.corpus_tokenList,
            modelId2sharedId=self.modelId2sharedId,
            tokId2clusterId=self.tokId2clusterId,
            vd_mask=self.vd_mask,
            npd_mask=self.npd_mask
        )
        return dataset

    def _get_tokIdList_from_groupIdList(self, groupIdList, score):
        assert self.hparams.tree_type == "groupId"
        tokIdList = []
        
        for groupId in groupIdList:
            tokIdList.extend(self.groupId2tokId[groupId])
            #tokIdList.extend(self.shared_groupId2tokId[groupId]) #### vanilla tokId도 list에 들어가는 상황
        return list(set(tokIdList))
   
    def _get_groupId_from_tokId(self, tokId):
        return self.shared_tokId2groupId[tokId]

    def forward(
        self,
        input_ids,
        attention_mask,
        loss_mask=None,
        decoder_attention_mask=None,
        decoder_input_ids=None,
        lm_labels=None,
        return_dict=True
    ):
        if lm_labels is None:
            assert decoder_input_ids is not None
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                return_dict=return_dict
            )
        if decoder_input_ids is None:
            assert lm_labels is not None
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=lm_labels,
                return_dict=return_dict,
                loss_mask=loss_mask
            ) 
    def _loss(self, batch):
        lm_labels = copy.deepcopy(batch["target_ids"])
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        if self.hparams.split_loss:
           outputs = self(
               input_ids=batch["source_ids"],
               attention_mask=batch["source_mask"],
               loss_mask=batch["target_loss_mask"],
               lm_labels=lm_labels,
               decoder_attention_mask=batch["target_mask"],
           )
        else:
           outputs = self(
               input_ids=batch["source_ids"],
               attention_mask=batch["source_mask"],
               lm_labels=lm_labels,
               decoder_attention_mask=batch["target_mask"]
           )
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        if self.hparams.tie_enc_dec_vocab:
            if self.hparams.w_enc:
                self.model.couple_total_vocab()
            else:
                self.model.couple_encoder_decoder_model_vocab()
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

    def get_list(self, batch_id, input_ids, score, trie_list):
        # starts with pad token & groupId for pad token is -1
        assert input_ids[0] == 0
        assert trie_list is not None and isinstance(trie_list, list)
        return self._get_from_trie(input_ids, trie_list[batch_id], score)

    def _find_idx(self, input_ids, idx):
        n_input_ids = np.array(input_ids)
        indices = np.where(n_input_ids == idx)
        return indices[0][-1]

    def _get_dec_status(self, input_ids):
        if self.hparams.model_type == "hyper-mem-np-only":
           assert self.es_tokid in input_ids
           es_idx = self._find_idx(input_ids, self.es_tokid)
           sub_input_ids = input_ids[es_idx+1:]
           if len(sub_input_ids) == 0:
               return "npd", [0]
           else:
               return "npd", [0]+sub_input_ids
        else:
           if self.vs_tokid in input_ids and self.es_tokid not in input_ids:
               return "vanilla", None
           elif self.vs_tokid not in input_ids and self.es_tokid in input_ids:
               es_idx = self._find_idx(input_ids, self.es_tokid)
               sub_input_ids = input_ids[es_idx+1:]
               if len(sub_input_ids) == 0:
                  return "npd", [0]
               else:
                  return "npd", [0]+sub_input_ids 
           elif self.vs_tokid in input_ids and self.es_tokid in input_ids:
               vs_idx = self._find_idx(input_ids, self.vs_tokid)
               es_idx = self._find_idx(input_ids, self.es_tokid)
               if vs_idx < es_idx:
                  sub_input_ids = input_ids[es_idx+1:]
                  if len(sub_input_ids) == 0:
                     return "npd", [0]
                  else:
                     return "npd", [0]+sub_input_ids
               elif es_idx < vs_idx:
                  return "vanilla", None 
               else:
                  assert False
           else:
               assert False

    def get(self, batch_id, input_ids, score, trie_dict=None):
        if len(input_ids) == 1:
            if self.hparams.model_type in ["hyper", "hyper-wo-vd"]:
               ret = [self.because_tokid, self.and_tokid, self.infer_tokid]
            elif self.hparams.model_type in ["hyper-mem", "hyper-mem-np-only"]:
               ret = [self.es_tokid]
            else:
               assert False
        elif input_ids[-1] in [self.ee_tokid]:
            if self.hparams.model_type in ["hyper", "hyper-wo-vd"]:
               ret = [self.because_tokid, self.and_tokid, self.infer_tokid]
            elif self.hparams.model_type in ["hyper-mem", "hyper-mem-np-only"]:
               ret = [1]
            else:
               assert False
        elif not self.hparams.model_type in ['hyper-mem-np-only'] and input_ids[-1] in [self.ve_tokid]:
            ret = [self.because_tokid, self.and_tokid, self.infer_tokid, 1]
        elif not self.hparams.model_type in ['hyper-mem-np-only'] and input_ids[-1] in [self.because_tokid, self.and_tokid, self.infer_tokid]:
            ret = [self.es_tokid, self.vs_tokid]
        else:
            if self.hparams.model_type in ["hyper-mem", "hyper-mem-np-only"]:
               dec_type, ids = self._get_dec_status(input_ids)
               assert dec_type == "npd"
               ret = self._get_from_trie(ids, self.trie_dict, 0)
            elif self.hparams.model_type in ["hyper", "hyper-wo-vd"]:
               dec_type, ids = self._get_dec_status(input_ids)
               if dec_type == "vanilla":
                  ret = self.vanilla_tokid_list+[self.ve_tokid]
                  assert 1 not in ret
               elif dec_type == "npd":
                  ret = self._get_from_trie(ids, self.trie_dict, 0)
               else:
                  assert False
            else:
               assert False

        return ret

    def _remove_eos(self, input_ids):
        return [el if el!=1 else self.ee_tokid for el in input_ids]

    """
    input_ids가 들어오면, 해당 tokId가 속한 groupId 찾고, 그걸 가지고 trie_dict 넘어간 다음
    해당 subtree의 key들(groupId) 를 모은 tokId return
    """
    def _get_from_trie(self, input_ids, trie_dict, score):
        #print(f"input_ids: {input_ids}")
        if self.hparams.tree_type == "groupId":
            if len(input_ids) == 0:
                possible_GroupList = list(trie_dict.keys())
                tokIdList = self._get_tokIdList_from_groupIdList(possible_GroupList, score)
                return list(self._remove_eos(tokIdList)) 
            else:
                curGroupId = self._get_groupId_from_tokId(input_ids[0])
                if curGroupId in list(trie_dict.keys()):
                    return self._get_from_trie(input_ids[1:], trie_dict[curGroupId], score) 
                else:
                    return []
        elif self.hparams.tree_type == "nodeId":
            if input_ids[-1] == 1:
                return []
            else:
                NodeId = self._get_nodeId_from_tokId(input_ids[-1])
                next_nId_List = list(trie_dict[NodeId])
                tokIdList = self._get_tokIdList_from_nodeIdList(next_nId_List, score)
                return tokIdList
        elif self.hparams.tree_type == "clusterId":
            if len(input_ids) == 0:
                return list(trie_dict.keys()) 
            else:
                if input_ids[0] in list(trie_dict.keys()):
                    return self._get_from_trie(input_ids[1:], trie_dict[clusterId], score)
                else:
                    return []
        else:
            raise NotImplementedError('tree type should be either groupId_tree or nodeId_tree!')

    def calculate_scores(self, preds, gt_text, query, batch_idx):
        em_list = []; total_f1 = []
        vs_f1 = []; es_f1 = []
        for idx, (_query, _pred, _gt) in enumerate(zip(query, preds, gt_text)):
            assert len(_pred) == 1, f"pred: {_pred}" 
            assert isinstance(_gt, str)
            _pred = _pred[0]
            _em = self._calculate_em(_pred, _gt)
            _es_f1, _vs_f1 = self._calculate_entail_f1(_pred, _gt)
            _total_f1 = self._calculate_tok_f1(_pred, _gt)
            if self.print and idx == 0:
                print(f"$" * 50)
                print(f"query: {_query}\npreds: {_pred}\ngt: {_gt}")
                print(f"em: {_em} // es_f1: {_es_f1} // vs_f1: {_vs_f1} // total_f1: {_total_f1}")
                print(f"$" * 50)
                print(" ")
            em_list.append(_em); total_f1.append(_total_f1)
            vs_f1.append(_vs_f1); es_f1.append(_es_f1)
        return em_list, total_f1, vs_f1, es_f1 


    def iter_calculate_scores(self, preds, gt_text):
        em_list = []; total_f1 = []
        vs_f1 = []; es_f1 = []
        for idx, (_pred, _gt) in enumerate(zip(preds, gt_text)):
            assert type(_pred) == list; assert type(_gt) == list 
            _pred = _pred[:len(_gt)]
            _em = set(_pred)==set(_gt) 
            _es_f1, _vs_f1 = self._calculate_iter_entail_f1(_pred, _gt)
            _total_f1 = self._calculate_iter_tok_f1(_pred, _gt)
            if self.print and idx == 0:
                print(f"$" * 50)
                print(f"preds: {_pred}\ngt: {_gt}")
                print(f"em: {_em} // es_f1: {_es_f1} // vs_f1: {_vs_f1} // total_f1: {_total_f1}")
                print(f"$" * 50)
                print(" ")
            em_list.append(_em); total_f1.append(_total_f1)
            vs_f1.append(_vs_f1); es_f1.append(_es_f1)
        return em_list, total_f1, vs_f1, es_f1 

    def _val_step(self, batch, batch_idx, return_elem=False):
        # calculates recall and em -> returns the list of each score
        _generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            #decoder_attention_mask=batch["target_mask"],
            max_length=self.hparams.max_output_length,
            num_beams=self.hparams.val_beam_size,
            num_return_sequences=self.hparams.ret_num,
            prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get(
                batch_id, sent.tolist(), scores
            ),
            early_stopping=True,
        )
        _generated_text = self.ids_to_text(_generated_ids, skip_special_tokens=True)
        inum = len(_generated_ids) // self.hparams.ret_num 
        assert inum == len(batch["output"])
        generated_text = [
            _generated_text[
                i * self.hparams.ret_num : (i + 1) * self.hparams.ret_num
            ]
            for i in range(inum)
        ]
        generated_ids = [
            _generated_ids[
               i * self.hparams.ret_num : (i+1) * self.hparams.ret_num
            ].detach().cpu().numpy().tolist()
            for i in range(inum)
        ]
        if print:
            print(f"GT: {batch['target_ids'][0]}")
            print(f"PRED: {generated_ids[0]}")
            print('='*80)
        em_list, total_f1, vs_f1, es_f1 = self.calculate_scores(
            generated_text, batch['output'], batch["input"], batch_idx
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
                "gt_tok": list(batch["target_ids"].detach().cpu().numpy().tolist()),
                "pred": list(generated_text),
                "pred_tok": list(generated_ids),
                "em": list(em_list),
                "total_f1": list(total_f1),
                "vs_f1": list(vs_f1),
                "es_f1": list(es_f1),
            }
        else:
            return em_list, total_f1, vs_f1, es_f1 

    def _find_unique_path(self, seq, trie):
        _ids = seq.tolist()
        _ids = [int(elem) for elem in _ids]
        
        generated_ids = []
        if _ids[-1] == 0:
            for el in range(len(_ids)):
                if el != 0 and _ids[el] == 0:
                    continue
                else:
                    generated_ids.append(_ids[el])
        else:
            generated_ids = _ids 
        assert generated_ids[-1] != 0
        generated_ids = generated_ids[:-1]
        
        cur_dict = trie 
        for i in range(len(generated_ids) - 1):
            cur_dict = cur_dict[int(generated_ids[i])]
        if len(cur_dict[generated_ids[-1]]) == 0:
            add_list.append(generated_ids)
        else:
            cur_dict = cur_dict[int(generated_ids[-1])]
            # check if it is an unique path
            keys = list(cur_dict.keys())
            if len(keys) > 1:
                print(f"More than 1: {keys}")
                keys = [keys[0]]
            while len(cur_dict[keys[0]]) != 0:
                generated_ids.append(keys[0])
                cur_dict = cur_dict[keys[0]]
                keys = list(cur_dict.keys())
                if len(keys) > 1:
                    print(f"More than 1: {keys}")
                    keys = [keys[0]]
            generated_ids.append(1)
        return generated_ids

    def _test_step(self, batch, batch_idx, return_elem=False):
       
        # for case where it resume test 
        test_input = []
        for _input in batch['input']:
            if _input in self.test_input_list: continue
            else: test_input.append(_input) 
        test_num = len(test_input)
        start_num = len(batch['input'])-test_num
        if test_num == 0: return None
        test_output = batch['output'][start_num:]
        test_source_ids = batch['source_ids'][start_num:]
        test_source_masks = batch['source_mask'][start_num:]
        test_target_ids = batch['target_ids'][start_num:]
        test_target_masks = batch['target_mask'][start_num:]
        assert len(test_output) == test_num, f'test_output: {len(test_output)}\ttest_num: {test_num}'
        _trie_list = [copy.deepcopy(self.trie_dict) for _ in range(test_num)]
        
        _generated_ids = self.model.generate(
             test_source_ids, 
             attention_mask=test_source_masks,
             use_cache=True,
             max_length=self.hparams.max_output_length,
             num_beams=self.hparams.val_beam_size,
             num_return_sequences=self.hparams.ret_num,
             prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get(
                 batch_id, sent.tolist(), scores
             ),
             early_stopping=True,
        )
        _generated_text = self.ids_to_text(_generated_ids, skip_special_tokens=True)
        inum = len(_generated_ids) // self.hparams.ret_num
        assert inum == len(batch['output']) 
        generated_text = [
            _generated_text[
                i * self.hparams.ret_num: (i + 1) * self.hparams.ret_num
            ]
            for i in range(inum)
        ]
        generated_ids = [
            _generated_ids[
                i * self.hparams.ret_num : (i+1) * self.hparams.ret_num
            ].detach().cpu().numpy().tolist()
            for i in range(inum)
        ]
        em_list, total_f1, vs_f1, es_f1 = self.calculate_scores(
            generated_text, batch["output"], batch["input"], batch_idx
        )
        if return_elem:
            assert (
                len(list(test_input))
                == len(list(generated_text))
                == len(list(em_list))
            )
            return {
                "input": list(test_input),
                "gt": list(test_output),
                "pred": list(generated_text),
                "pred_tok": list(generated_ids),
                "em": list(em_list),
                "total_f1": list(total_f1),
                "vs_f1": list(vs_f1),
                "es_f1": list(es_f1),
            }
        else:
            return em_list, total_f1, vs_f1, es_f1 


    def ids_to_text(self, _generated_ids, skip_special_tokens=True):
        generated_ids = []
        for _ids in _generated_ids:
            _ids = copy.deepcopy(_ids)
            _ids = _ids.detach().cpu().numpy()
            _text = [self.shared_tokId2tokText[_id] for _id in _ids]
            generated_ids.append(self.tokenizer.convert_tokens_to_ids(_text))
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True
        )
        text = []
        if not skip_special_tokens:
            for elem in gen_text: 
               rep_elem = elem.replace('<pad>', '')
               text.append(rep_elem)
        else:
            text = gen_text
        text = self.lmap(str.strip, text)
        return text



    def validation_step(self, batch, batch_idx):
        if self.hparams.save_model_only:
           self.convert_checkpoint("new_ckpt")
           print(f"Done saving checkpoint.. {self.current_epoch}")
           sys.exit()

        em_score, total_f1_score, vs_f1_score, es_f1_score = self._val_step(batch, batch_idx)
        self.em_score_list.extend(list(em_score))
        self.total_f1_score_list.extend(list(total_f1_score)) 
        self.vs_f1_score_list.extend(list(vs_f1_score)) 
        self.es_f1_score_list.extend(list(es_f1_score)) 

    def validation_epoch_end(self, outputs):
        avg_em = np.mean(np.array(self.em_score_list))
        avg_total_f1 = np.mean(np.array(self.total_f1_score_list))
        avg_vs_f1 = np.mean(np.array(self.vs_f1_score_list))
        avg_es_f1 = np.mean(np.array(self.es_f1_score_list))
        self.em_score_list = []
        self.total_f1_score_list = []
        self.vs_f1_score_list = []
        self.es_f1_score_list = []

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
            "val_total_f1",
            avg_total_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_vs_f1",
            avg_vs_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_es_f1",
            avg_es_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return


    def _iter_test_step(self, batch, batch_idx, return_elem=False):
        test_input = []
        for _input in batch["input"]:
            if _input in self.test_input_list: continue
            else: test_input.append(_input)
        test_num = len(test_input)
        start_num = len(batch['input'])-test_num 

        if test_num == 0: return None 
        test_output = [self.dev_input2output[el] for el in test_input] 
        test_source_ids = batch["source_ids"][start_num:]
        test_source_masks = batch["source_mask"][start_num:]
        test_target_ids = batch["target_ids"][start_num:]
        test_target_masks = batch["target_mask"][start_num:]
        assert len(test_output) == test_num, f"test_output: {len(test_output)}\ntest_num: {test_num}"

        _test_input = test_input

        test_pred = [[] for _ in range(len(batch["input"]))]
        iternum = np.array([len(el) for el in test_output]).max()

        for iter_idx in range(iternum):
           _generated_ids = self.model.generate(
                     test_source_ids,
                     attention_mask=test_source_masks,
                     use_cache=True,
                     max_length=self.hparams.max_output_length,
                     num_beams=self.hparams.val_beam_size,
                     num_return_sequences=self.hparams.ret_num,
                     prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get(batch_id, sent.tolist(), scores),
                     early_stopping=True
                 )
           _generated_text = self.ids_to_text(_generated_ids)
           inum = len(_generated_ids) // self.hparams.ret_num 
           assert inum == len(batch['output'])
         
           generated_text = [
               _generated_text[
                  i*self.hparams.ret_num:(i+1)*self.hparams.ret_num
               ]
               for i in range(inum)
           ]


           generated_ids = [
               _generated_ids[
                  i*self.hparams.ret_num:(i+1)*self.hparams.ret_num
               ]
               for i in range(inum)
           ]

           for idx, elem in enumerate(generated_text):
               test_pred[idx].append(elem[0])
           test_source_ids, test_source_masks, _test_input = self.tokenize4next_hop(_test_input, generated_text)

        em_list, total_f1, vs_f1, es_f1 = self.iter_calculate_scores(
            test_pred, test_output
         )

        if return_elem:
            assert len(list(test_input))==len(list(generated_text))==len(list(em_list))
            return {
               "input": list(test_input),
               "gt": list(test_output),
               "pred": list(generated_text),
               "pred_tok": list(generated_ids),
               "em": list(em_list),
               "total_f1": list(total_f1),
               "vs_f1": list(vs_f1),
               "es_f1": list(es_f1),
            }
        else:
            return em_list, total_f1, vs_f1, es_f1

    def tokenize4next_hop(self, input_list, pred_list):
        sen_list = [f"{_input} {_pred[0]}" for _input, _pred in zip(input_list, pred_list)]
        tok_ret = self.tokenizer.batch_encode_plus(
                  sen_list,
                  max_length=self.hparams.max_input_length,
                  padding="max_length",
                  truncation=True,
                  return_tensors="pt"
              )
        return tok_ret["input_ids"].to(self.device), tok_ret["attention_mask"].to(self.device), sen_list

    def test_step(self, batch, batch_idx):
        if self.hparams.do_iteration:
           ret_dict = self._iter_test_step(batch, batch_idx, return_elem=True)
        else:
           ret_dict = self._test_step(batch, batch_idx, return_elem=True)
        if ret_dict is None: return None

        self.test_input_list.extend(ret_dict["input"])
        self.test_gt_list.extend(ret_dict["gt"])
        self.test_pred_list.extend(ret_dict["pred"])
        self.test_pred_tok_list.extend(ret_dict["pred_tok"])
        self.test_em_score_list.extend(ret_dict["em"])
        self.test_total_f1_score_list.extend(ret_dict["total_f1"])
        self.test_vs_f1_score_list.extend(ret_dict["vs_f1"])
        self.test_es_f1_score_list.extend(ret_dict["es_f1"])
        self._save_test() 

    def _save_test(self, epoch_end=False):
        os.makedirs(self.hparams.output_dir, exist_ok=True)
        _input = self.gather_list(self.test_input_list) 
        _gt = self.gather_list(self.test_gt_list)
        _pred = self.gather_list(self.test_pred_list)
        _pred_tok = self.gather_list(self.test_pred_tok_list)
        _em = self.gather_list(self.test_em_score_list)
        _total_f1 = self.gather_list(self.test_total_f1_score_list)
        _test_vs = self.gather_list(self.test_vs_f1_score_list)
        _test_es = self.gather_list(self.test_es_f1_score_list)
        assert len(_input) == len(_gt) == len(_pred) == len(_em) == len(_total_f1) == len(_test_vs) == len(_test_es) == len(_pred_tok)
        if self.print:
            with open(self.test_save_name, "w") as f:
                 json.dump(
                     {
                         "input": _input,
                         "gt": _gt,
                         "pred": _pred,
                         "pred_tok": _pred_tok,
                         "em": _em,
                         "total_f1": _total_f1,
                         "vs_f1": _test_vs,
                         "es_f1": _test_es,
                     },
                     f,
                 )
            if epoch_end:
                 print(
                     f"Saving in {self.test_save_name}!\nnumber of elements: {len(_input)}"
                 )
                 print(f"EM: {np.array(_em).mean()}")
                 print(f"Total F1: {np.array(_total_f1).mean()}")
                 print(f"Vs F1: {np.array(_test_vs).mean()}")
                 print(f"Es F1: {np.array(_test_es).mean()}")
    
    def test_epoch_end(self, outputs):
        self._save_test(epoch_end=True)

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
        try:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            self.save_epoch.append(self.current_epoch)
        except:
            print(f"Fail to save ... {self.current_epoch}")
        if len(self.save_epoch) > 5:
            rm_file = f"best_tfmr_{self.save_epoch[0]}"
            os.system(f"rm -rf {os.path.join(self.hparams.output_dir, rm_file)}")
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

    def convert_checkpoint(self, sub_path=None):
        if sub_path is None:
           save_path = os.path.join(
               self.hparams.output_dir, f"best_tfmr_{self.current_epoch}"
           )
        else:
           save_path = os.path.join(
               self.hparams.output_dir, sub_path     
           )
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.save_epoch.append(self.current_epoch)
        if len(self.save_epoch) > 5:
            rm_file = f"best_tfmr_{self.save_epoch[0]}"
            os.system(f"rm -rf {os.path.join(self.hparams.output_dir, rm_file)}")
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


class T5Split(T5BaseClass): 
    def __init__(self, args):
        super(T5Split, self).__init__()
        if print: print(f'==========T5Split===========')
        self.save_hyperparameters(args)
        if self.hparams.do_train:         
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.hparams.tokenizer_name_or_path
            )
        if self.hparams.do_test:
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.test_model_path)
        sp_tokens = ["[Ee]", "[Es]", "[Ve]", "[Vs]", "[HYPOT]", "[QUESTION]", "[ANSWER]", "[BECAUSE]", "[INFER]", "[AND]"]
        self.tokenizer.add_tokens(sp_tokens, special_tokens=True)
 
        if np_only(self.hparams):
            raise NotImplementedError('np_only case is not Implemented Yet!') 
            self.np_only = True 
        else:
            self.np_only = False

        ### Shared => Vanilla + NPD ###
        if self.np_only:
            self.shared_tokId2Emb = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), "rb"))
            self.shared_tokId2tokText = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2tokText), "rb"))
            self.shared_tokId2groupId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2groupId), "rb"))
            self.shared_groupId2tokId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.groupId2tokIdList), "rb"))
            self.contextualized_emb_num = len(self.shared_tokId2Emb.keys())
            self.total_emb_num = self.contextualized_emb_num
            self.groupId2tokId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.groupId2tokIdList), "rb"))
            self.contextualized_tokid2emb = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), "rb"))
            self.modelId2sharedId = None
            self.vd_mask=None; self.npd_mask=None
        else:
            if os.path.exists(os.path.join(self.hparams.output_dir, "shared.modelId2sharedId.pickle")):
               self.shared_tokId2Emb = pickle.load(open(os.path.join(self.hparams.output_dir, "shared.tokId2emb.pickle"), "rb"))
               self.shared_tokId2tokText = pickle.load(open(os.path.join(self.hparams.output_dir, "shared.tokId2tokText.pickle"), "rb"))
               self.shared_tokId2groupId = pickle.load(open(os.path.join(self.hparams.output_dir, "shared.tokId2groupId.pickle"), "rb"))
               self.shared_groupId2tokId = pickle.load(open(os.path.join(self.hparams.output_dir, "shared.groupId2tokId.pickle"), "rb"))
               self.modelId2sharedId = pickle.load(open(os.path.join(self.hparams.output_dir, "shared.modelId2sharedId.pickle"), "rb"))
               self.groupId2tokId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.groupId2tokIdList), "rb"))
               self.contextualized_tokid2emb = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), "rb"))
            else:
               self.contextualized_tokid2emb = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), "rb"))
               self.groupId2tokId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.groupId2tokIdList), "rb"))
               self.tokId2groupId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2groupId), "rb"))
               self.tokId2tokText = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2tokText), "rb"))
               self.model_tokid2emb = self._get_model_tokid2emb() 
               self.model_tokid2text = {}
               for i in range(len(self.tokenizer)):
                  text = self.tokenizer.convert_ids_to_tokens(i)
                  self.model_tokid2text[i] = text 

               self.shared_tokId2Emb, self.shared_tokId2tokText, self.shared_tokId2groupId, self.shared_groupId2tokId, self.modelId2sharedId = self.combine_tokid2emb()

            sp_tokens_list = self.tokenizer.batch_encode_plus(sp_tokens)["input_ids"]
            self.sp_tokens_list = sp_tokens_list
            self.ee_tokid = self.modelId2sharedId[sp_tokens_list[0][0]]
            self.es_tokid = self.modelId2sharedId[sp_tokens_list[1][0]]
            self.ve_tokid = self.modelId2sharedId[sp_tokens_list[2][0]]
            self.vs_tokid = self.modelId2sharedId[sp_tokens_list[3][0]]
            self.hypot_tokid = self.modelId2sharedId[sp_tokens_list[4][0]]
            self.question_tokid = self.modelId2sharedId[sp_tokens_list[5][0]]
            self.answer_tokid = self.modelId2sharedId[sp_tokens_list[6][0]]
            self.because_tokid = self.modelId2sharedId[sp_tokens_list[7][0]]
            self.infer_tokid = self.modelId2sharedId[sp_tokens_list[8][0]]
            self.and_tokid = self.modelId2sharedId[sp_tokens_list[9][0]]

            if self.print:
               print(f'\n\nee: {self.ee_tokid}\nes: {self.es_tokid}\nve: {self.ve_tokid}\nvs: {self.vs_tokid}\nhypot: {self.hypot_tokid}\nquestion: {self.question_tokid}\nanswer: {self.answer_tokid}\nbecause: {self.because_tokid}\ninfer: {self.infer_tokid}\nand: {self.and_tokid}\n\n')
            
            self.vanilla_tokid_list = [self.modelId2sharedId[i] for i in range(2,32000)]
            if self.print:
               print(f"Vanilla TokId List starts from .. {self.modelId2sharedId[3]} to {self.modelId2sharedId[32000]}")
            assert len(self.contextualized_tokid2emb) == self.modelId2sharedId[3], f"len(self.contextualized_tokid2emb): {len(self.contextualized_tokid2emb)}\nself.modelId2sharedId[3]: {self.modelId2sharedId[3]}"
            self.total_emb_num = len(self.shared_tokId2Emb.keys())
            self.contextualized_emb_num = len(self.contextualized_tokid2emb)-3
            assert self.total_emb_num == len(self.tokenizer)+len(self.contextualized_tokid2emb)-3, f"Total Emb Num: {self.total_emb_num}\tlen(self.tokenizer): {len(self.tokenizer)}\tlen(self.contextualized_tokid2emb): {len(self.contextualized_tokid2emb)}\nadd => {len(self.tokenizer)+len(self.contextualized_tokid2emb)-3}"
       
        if self.hparams.dev_input2output is not None:
            self.dev_input2output = json.load(open(os.path.join(self.hparams.dataset, self.hparams.dev_input2output), "rb"))

        if self.hparams.cluster_num != -1:
            self.tokId2clusterId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2clusterId), 'rb'))
        else:
            self.tokId2clusterId = None 
        
        self.corpus_tokenList = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.corpus2tokIdList), "rb"))
        self.trie_dict = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tree_path), "rb"))
       
        if 'log.txt' in os.listdir(self.hparams.output_dir):
            print(f'+++ removing previous log file!')
            os.system(f'rm {os.path.join(self.hparams.output_dir, "log.txt")}')
        self.file = open(os.path.join(self.hparams.output_dir, "log.txt"), 'w') 
        print(f'+++ Writing logs in {os.path.join(self.hparams.output_dir, "log.txt")}')
 

        if self.print: print(f"$$$$$$ # of embedding: {self.total_emb_num}")
        self.config = T5Config.from_pretrained(self.hparams.model_name_or_path)
        self.config.update({"fp16": self.hparams.fp16})
        self.config.update({"train_c_emb": self.hparams.train_c_emb}) 
        self.config.update({"do_test": self.hparams.do_test}) 
        self.config.update(
            {"contextualized_emb_num": self.contextualized_emb_num}
        )
        self.config.update({"freeze_vocab_emb": self.hparams.freeze_vocab_emb})
        self.config.update({'change_lm_head': self.hparams.change_lm_head})
        self.config.update({'change_dec': self.hparams.change_dec})
        self.config.update({'change_enc': self.hparams.change_enc})
        self.config.update({'original_T5': self.hparams.original_T5})
        self.config.update({'tie_vocab_emb': self.hparams.tie_vocab_emb})
        self.config.update({'tie_np_emb': self.hparams.tie_np_emb})
        self.config.update(
           {"contextualized_file": os.path.join(self.hparams.dataset, self.hparams.contextualized_file)}
        )  # tokId_emb.pickle
        self.config.update(
           {'model_vocab_file': os.path.join(self.hparams.dataset, "modelId2Emb.pickle")}      
        )
        self.config.update({"vanilla_emb_num": len(self.tokenizer)})
        self.config.update({"total_emb_num": self.total_emb_num})
        self.config.update({'loss_weight': self.hparams.loss_weight})
        self.save_epoch = []

        # If in training mode, load ckpt for training
        if self.hparams.do_train:
            if self.hparams.train_c_emb:
                self.model = entail_T5_split.from_pretrained(
                              self.hparams.model_name_or_path, config=self.config, ignore_mismatched_sizes=True
                          )
            else:
                self.model = entail_T5_split.from_pretrained(
                    self.hparams.model_name_or_path, config=self.config
                )

            if self.print:
                print(f"@@@ Loading Model from {self.hparams.model_name_or_path}")
                print(f'@@@ Loading decoder embedding: {self.hparams.embedding_model}')

            self.em_score_list = []
            self.total_f1_score_list = []
            self.vs_f1_score_list = []
            self.es_f1_score_list = []

            
        # If in testing mode, load ckpt for inference
        if self.hparams.do_test:
            self.model = entail_T5_split.from_pretrained(
               self.hparams.test_model_path, config=self.config#, ignore_mismatched_sizes=True
            )
           
            if self.print:
                print(f"@@@ Loading Model from {self.hparams.test_model_path}")
          
            self.test_save_name = os.path.join(self.hparams.output_dir, f"{self.hparams.test_name}_{self.hparams.tree_type}_result_beam{self.hparams.val_beam_size}.json")               
            if self.print: print(f'@@@ Initialize Test!!')
            self.test_input_list = []
            self.test_gt_list = []
            self.test_gt_tok_list = []
            self.test_pred_list = []
            self.test_pred_tok_list = []
            self.test_em_score_list = []
            self.test_total_f1_score_list = []
            self.test_vs_f1_score_list = []
            self.test_es_f1_score_list = []

        if self.hparams.freeze_encoder:
            if self.print:
                print(f"@@@ Freeze Encoder!")
                encoder = self.model.get_encoder()
                for n, p in encoder.named_parameters():
                    p.requires_grad=False
        
        self.cnt_over = 0


        if self.print:
            print(f"*** Dump NP Embeddings")
            dec_embeddings = self.model.decoder.np_embed_tokens.weight
            np_vocab = np.memmap(os.path.join(self.hparams.output_dir, f"train_{self.hparams.do_train}_np_vocab.dat"), dtype="float32", mode="w+", shape=(dec_embeddings.shape[0], 1024))
            for i in tqdm(range(dec_embeddings.shape[0])):
               np_vocab[i][:] = dec_embeddings[i][:].detach().numpy()
           
            print(f"*** Dump Decoder Embeddings")
            dec_shared = self.model.decoder.embed_tokens.weight 
            dec_vocab = np.memmap(os.path.join(self.hparams.output_dir, f"train_{self.hparams.do_train}_dec_vocab.dat"), dtype="float32", mode="w+", shape=(dec_shared.shape[0], 1024))
            for i in tqdm(range(dec_shared.shape[0])):
               dec_vocab[i][:] = dec_shared[i][:].detach().numpy()
           
            print(f"*** Dump Encoder Embeddings")
            enc_shared = self.model.encoder.embed_tokens.weight 
            enc_vocab = np.memmap(os.path.join(self.hparams.output_dir, f"train_{self.hparams.do_train}_enc_vocab.dat"), dtype="float32", mode="w+", shape=(enc_shared.shape[0], 1024))
            for i in tqdm(range(enc_shared.shape[0])):
               enc_vocab[i][:] = enc_shared[i][:].detach().numpy()
            # sys.exit()



        vocab_num = self.model.config.vocab_size 
        self.vd_mask = [1,1,1]+[0]*(len(self.contextualized_tokid2emb)-3)+[1]*(len(self.tokenizer)-3)+[0]*(vocab_num-len(self.tokenizer)) # remove sp except for ee, es, ve, vs 
        ### TODO: npd mask에서 Ee, Es는 open 할지 말지 
        self.npd_mask = [1,1,1]+[1]*(len(self.contextualized_tokid2emb)-3)+[0]*(len(self.tokenizer)-3-6)+[0]*(vocab_num-len(self.tokenizer)+6) 
        assert len(self.vd_mask) == len(self.npd_mask) == len(self.contextualized_tokid2emb)-3+vocab_num 

        self.len_test_dataset = len(self.test_dataloader())


    def combine_tokid2emb(self):
        shared_tokId2Emb = copy.deepcopy(self.contextualized_tokid2emb)
        shared_tokId2tokText = copy.deepcopy(self.tokId2tokText)
        shared_tokId2groupId = copy.deepcopy(self.tokId2groupId)
        shared_groupId2tokId = copy.deepcopy(self.groupId2tokId)
        modelId2sharedId = {}

        tokId = len(shared_tokId2Emb); groupId = len(shared_groupId2tokId)

        for id, text in self.model_tokid2text.items():
            if id==0 or id==1 or id==2:
               modelId2sharedId[id] = id  
               continue
            modelId2sharedId[id] = tokId 
            if text not in shared_tokId2tokText.values():
               assert tokId not in shared_tokId2Emb 
               assert groupId not in shared_groupId2tokId 
               shared_tokId2Emb[tokId] = self.model_tokid2emb[id]
               shared_tokId2tokText[tokId] = text 
               shared_tokId2groupId[tokId] = groupId 
               shared_groupId2tokId[groupId] = [tokId] 
               tokId +=1; groupId += 1
            else:
               assert tokId not in shared_tokId2Emb
               gId = self.text2gId(self.tokId2tokText, self.tokId2groupId, text)
               assert gId in shared_groupId2tokId
               shared_tokId2Emb[tokId] = self.model_tokid2emb[id]
               shared_tokId2tokText[tokId] = text 
               shared_tokId2groupId[tokId] = gId 
               shared_groupId2tokId[gId].append(tokId)
               tokId += 1

        print(f"# of Contextualized Token Embedding: {len(self.contextualized_tokid2emb)}")
        print(f"# of Model Token Embedding: {len(self.model_tokid2emb)}")
        print(f"# of Shared Token Embedding: {len(shared_tokId2Emb)}")

        os.makedirs(self.hparams.output_dir, exist_ok=True)
        with open(os.path.join(self.hparams.output_dir, "shared.tokId2emb.pickle"), "wb") as f:
           pickle.dump(shared_tokId2Emb, f)
        with open(os.path.join(self.hparams.output_dir, "shared.tokId2tokText.pickle"), "wb") as f:
           pickle.dump(shared_tokId2tokText, f)
        with open(os.path.join(self.hparams.output_dir, "shared.tokId2groupId.pickle"), "wb") as f:
           pickle.dump(shared_tokId2groupId, f)
        with open(os.path.join(self.hparams.output_dir, "shared.groupId2tokId.pickle"), "wb") as f:
           pickle.dump(shared_groupId2tokId, f)
        with open(os.path.join(self.hparams.output_dir, "shared.modelId2sharedId.pickle"), "wb") as f:
           pickle.dump(modelId2sharedId, f)

        return shared_tokId2Emb, shared_tokId2tokText, shared_tokId2groupId, shared_groupId2tokId, modelId2sharedId

    def text2gId(self, tokId2tokText, tokId2groupId, text):
        for t_id, t_text in tokId2tokText.items():
           if t_text == text:
               return tokId2groupId[t_id]

    def _get_model_tokid2emb(self):
        #model = T5Model.from_pretrained(self.hparams.model_name_or_path)    
        #vocab_weight = model.state_dict()["shared.weight"]
        model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        vocab_weight = model.encoder.embed_tokens.weight # 32110
        print(f'# of vocab => {len(self.tokenizer)}')
        vocab_num = len(self.tokenizer) #vocab_weight.size()[0]
        vocab_id2emb = {}
        for id in range(vocab_num):
           emb = vocab_weight[id].detach().cpu().numpy()
           vocab_id2emb[id] = emb
        with open(os.path.join(self.hparams.dataset, "modelId2Emb.pickle"), 'wb') as f:
           pickle.dump(vocab_id2emb, f)
        return vocab_id2emb

    def _get_dataset(self, split):
        dataset = ENTAILDataset(
            tokenizer=self.tokenizer,
            split=split,
            hparams=self.hparams,
            ee_tokid=self.ee_tokid if self.np_only else self.sp_tokens_list[0][0],
            es_tokid=self.es_tokid if self.np_only else self.sp_tokens_list[1][0],
            corpus_tokenList=self.corpus_tokenList,
            modelId2sharedId=self.modelId2sharedId,
            tokId2clusterId=self.tokId2clusterId,
            vd_mask=self.vd_mask,
            npd_mask=self.npd_mask
        )
        return dataset

    def _get_tokIdList_from_groupIdList(self, groupIdList, score):
        assert self.hparams.tree_type == "groupId"
        tokIdList = []
        
        for groupId in groupIdList:
            tokIdList.extend(self.groupId2tokId[groupId])
            #tokIdList.extend(self.shared_groupId2tokId[groupId]) #### vanilla tokId도 list에 들어가는 상황
        return list(set(tokIdList))
   
    def _get_groupId_from_tokId(self, tokId):
        return self.shared_tokId2groupId[tokId]

    def forward(
        self,
        input_ids,
        attention_mask,
        vd_loss_mask=None,
        npd_loss_mask=None,
        decoder_attention_mask=None,
        decoder_input_ids=None,
        lm_labels=None,
        return_dict=True
    ):
        if lm_labels is None:
            assert decoder_input_ids is not None
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                return_dict=return_dict
            )
        if decoder_input_ids is None:
            assert lm_labels is not None
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=lm_labels,
                return_dict=return_dict,
                vd_loss_mask=vd_loss_mask,
                npd_loss_mask=npd_loss_mask
            )

    def _loss(self, batch):
        lm_labels = copy.deepcopy(batch["target_ids"])
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        if self.hparams.split_loss or self.hparams.loss_weight is not None:
           outputs = self(
               input_ids=batch["source_ids"],
               attention_mask=batch["source_mask"],
               vd_loss_mask=batch["target_vd_loss_mask"],
               npd_loss_mask=batch["target_npd_loss_mask"],
               lm_labels=lm_labels,
               decoder_attention_mask=batch["target_mask"],
           )
        else:
           outputs = self(
               input_ids=batch["source_ids"],
               attention_mask=batch["source_mask"],
               lm_labels=lm_labels,
               decoder_attention_mask=batch["target_mask"]
           )
        loss = outputs[0]
        logits = outputs[1]
        if self.print:
           self.file.write(f"Epoch: {self.current_epoch}\tStep: {self.global_step}\nlogits[:50]: {logits[:50]}\nlogits[-50:]: {logits[-50:]}\n\n")
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

    def get_list(self, batch_id, input_ids, score, trie_list):
        # starts with pad token & groupId for pad token is -1
        assert input_ids[0] == 0
        assert trie_list is not None and isinstance(trie_list, list)
        return self._get_from_trie(input_ids, trie_list[batch_id], score)

    def _find_idx(self, input_ids, idx):
        n_input_ids = np.array(input_ids)
        indices = np.where(n_input_ids == idx)
        return indices[0][-1]

    def _get_dec_status(self, input_ids):
        if self.hparams.model_type in ["hyper-mem-np-only", "hyper-split-mem"]:
           assert self.es_tokid in input_ids
           es_idx = self._find_idx(input_ids, self.es_tokid)
           sub_input_ids = input_ids[es_idx+1:]
           if len(sub_input_ids) == 0:
               return "npd", [0]
           else:
               return "npd", [0]+sub_input_ids
        else:
           if self.vs_tokid in input_ids and self.es_tokid not in input_ids:
               return "vanilla", None
           elif self.vs_tokid not in input_ids and self.es_tokid in input_ids:
               es_idx = self._find_idx(input_ids, self.es_tokid)
               sub_input_ids = input_ids[es_idx+1:]
               if len(sub_input_ids) == 0:
                  return "npd", [0]
               else:
                  return "npd", [0]+sub_input_ids 
           elif self.vs_tokid in input_ids and self.es_tokid in input_ids:
               vs_idx = self._find_idx(input_ids, self.vs_tokid)
               es_idx = self._find_idx(input_ids, self.es_tokid)
               if vs_idx < es_idx:
                  sub_input_ids = input_ids[es_idx+1:]
                  if len(sub_input_ids) == 0:
                     return "npd", [0]
                  else:
                     return "npd", [0]+sub_input_ids
               elif es_idx < vs_idx:
                  return "vanilla", None 
               else:
                  assert False
           else:
               assert False

    def get(self, batch_id, input_ids, score, trie_dict=None):
        if len(input_ids) == 1:
            if self.hparams.model_type in ["hyper", "hyper-wo-vd", "hyper-split"]:
               ret = [self.because_tokid, self.and_tokid, self.infer_tokid]
            elif self.hparams.model_type in ["hyper-mem", "hyper-mem-np-only", "hyper-split-mem"]:
               ret = [self.es_tokid]
            else:
               assert False
        elif input_ids[-1] in [self.ee_tokid]:
            if self.hparams.model_type in ["hyper", "hyper-wo-vd", "hyper-split"]:
               ret = [self.because_tokid, self.and_tokid, self.infer_tokid]
            elif self.hparams.model_type in ["hyper-mem", "hyper-mem-np-only", "hyper-split-mem"]:
               ret = [1]
            else:
               assert False
        elif not self.hparams.model_type in ['hyper-mem-np-only'] and input_ids[-1] in [self.ve_tokid]:
            assert self.hparams.model_type not in ['hyper-mem', 'hyper-split-mem']
            ret = [self.because_tokid, self.and_tokid, self.infer_tokid, 1]
        elif not self.hparams.model_type in ['hyper-mem-np-only'] and input_ids[-1] in [self.because_tokid, self.and_tokid, self.infer_tokid]:
            assert self.hparams.model_type not in ['hyper-mem', 'hyper-split-mem']
            ret = [self.es_tokid, self.vs_tokid]
        else:
            if self.hparams.model_type in ["hyper-mem", "hyper-mem-np-only", "hyper-split-mem"]:
               dec_type, ids = self._get_dec_status(input_ids)
               assert dec_type == "npd"
               ret = self._get_from_trie(ids, self.trie_dict, 0)
            elif self.hparams.model_type in ["hyper", "hyper-wo-vd", "hyper-split"]:
               dec_type, ids = self._get_dec_status(input_ids)
              
               if dec_type == "vanilla":
                  ret = self.vanilla_tokid_list+[self.ve_tokid]
                  assert 1 not in ret
               elif dec_type == "npd":
                  ret = self._get_from_trie(ids, self.trie_dict, 0)
                  # ret = self.vanilla_tokid_list+[self.ee_tokid]
               else:
                  assert False

            else:
               assert False

        return ret

    def _remove_eos(self, input_ids):
        return [el if el!=1 else self.ee_tokid for el in input_ids]

    """
    input_ids가 들어오면, 해당 tokId가 속한 groupId 찾고, 그걸 가지고 trie_dict 넘어간 다음
    해당 subtree의 key들(groupId) 를 모은 tokId return
    """
    def _get_from_trie(self, input_ids, trie_dict, score):
        #print(f"input_ids: {input_ids}")
        if self.hparams.tree_type == "groupId":
            if len(input_ids) == 0:
                possible_GroupList = list(trie_dict.keys())
                tokIdList = self._get_tokIdList_from_groupIdList(possible_GroupList, score)
                return list(self._remove_eos(tokIdList)) 
            else:
                curGroupId = self._get_groupId_from_tokId(input_ids[0])
                if curGroupId in list(trie_dict.keys()):
                    return self._get_from_trie(input_ids[1:], trie_dict[curGroupId], score) 
                else:
                    return []
        elif self.hparams.tree_type == "nodeId":
            assert False
            if input_ids[-1] == 1:
                return []
            else:
                NodeId = self._get_nodeId_from_tokId(input_ids[-1])
                next_nId_List = list(trie_dict[NodeId])
                tokIdList = self._get_tokIdList_from_nodeIdList(next_nId_List, score)
                return tokIdList
        elif self.hparams.tree_type == "clusterId":
            assert False
            if len(input_ids) == 0:
                return list(trie_dict.keys()) 
            else:
                if input_ids[0] in list(trie_dict.keys()):
                    return self._get_from_trie(input_ids[1:], trie_dict[clusterId], score)
                else:
                    return []
        else:
            raise NotImplementedError('tree type should be either groupId_tree or nodeId_tree!')

    def calculate_scores(self, preds, gt_text, query, batch_idx):
        em_list = []; total_f1 = []
        vs_f1 = []; es_f1 = []
        for idx, (_query, _pred, _gt) in enumerate(zip(query, preds, gt_text)):
            assert len(_pred) == 1, f"pred: {_pred}" 
            assert isinstance(_gt, str)
            _pred = _pred[0]
            _em = self._calculate_em(_pred, _gt)
            _es_f1, _vs_f1 = self._calculate_entail_f1(_pred, _gt)
            _total_f1 = self._calculate_tok_f1(_pred, _gt)
            if self.print and idx == 0:
                print(f"$" * 50)
                print(f"query: {_query}\npreds: {_pred}\ngt: {_gt}")
                print(f"em: {_em} // es_f1: {_es_f1} // vs_f1: {_vs_f1} // total_f1: {_total_f1}")
                print(f"$" * 50)
                print(" ")
            em_list.append(_em); total_f1.append(_total_f1)
            vs_f1.append(_vs_f1); es_f1.append(_es_f1)
        return em_list, total_f1, vs_f1, es_f1 


    def iter_calculate_scores(self, preds, gt_text):
        em_list = []; total_f1 = []
        vs_f1 = []; es_f1 = []
        for idx, (_pred, _gt) in enumerate(zip(preds, gt_text)):
            assert type(_pred) == list; assert type(_gt) == list 
            _pred = _pred[:len(_gt)]
            _em = set(_pred)==set(_gt) 
            _es_f1, _vs_f1 = self._calculate_iter_entail_f1(_pred, _gt)
            _total_f1 = self._calculate_iter_tok_f1(_pred, _gt)
            if self.print and idx == 0:
                print(f"$" * 50)
                print(f"preds: {_pred}\ngt: {_gt}")
                print(f"em: {_em} // es_f1: {_es_f1} // vs_f1: {_vs_f1} // total_f1: {_total_f1}")
                print(f"$" * 50)
                print(" ")
            em_list.append(_em); total_f1.append(_total_f1)
            vs_f1.append(_vs_f1); es_f1.append(_es_f1)
        return em_list, total_f1, vs_f1, es_f1 

    def _val_step(self, batch, batch_idx, return_elem=False):
        # calculates recall and em -> returns the list of each score
        _generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            #decoder_attention_mask=batch["target_mask"],
            max_length=self.hparams.max_output_length,
            num_beams=self.hparams.val_beam_size,
            num_return_sequences=self.hparams.ret_num,
            prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get(
                batch_id, sent.tolist(), scores
            ),
            early_stopping=True,
        )
        print(f"Generated Ids: {_generated_ids[0]}")
        _generated_text = self.ids_to_text(_generated_ids, skip_special_tokens=True)
        print(f"Generated Text: {_generated_text[0]}")
        inum = len(_generated_ids) // self.hparams.ret_num 
        assert inum == len(batch["output"])
        generated_text = [
            _generated_text[
                i * self.hparams.ret_num : (i + 1) * self.hparams.ret_num
            ]
            for i in range(inum)
        ]
        generated_ids = [
            _generated_ids[
               i * self.hparams.ret_num : (i+1) * self.hparams.ret_num
            ].detach().cpu().numpy().tolist()
            for i in range(inum)
        ]
        if print:
            print(f"GT: {batch['target_ids'][0]}")
            print(f"PRED: {generated_ids[0]}")
            print('='*80)
        em_list, total_f1, vs_f1, es_f1 = self.calculate_scores(
            generated_text, batch['output'], batch["input"], batch_idx
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
                "gt_tok": list(batch["target_ids"].detach().cpu().numpy().tolist()),
                "pred": list(generated_text),
                "pred_tok": list(generated_ids),
                "em": list(em_list),
                "total_f1": list(total_f1),
                "vs_f1": list(vs_f1),
                "es_f1": list(es_f1),
            }
        else:
            return em_list, total_f1, vs_f1, es_f1 

    def _find_unique_path(self, seq, trie):
        _ids = seq.tolist()
        _ids = [int(elem) for elem in _ids]
        
        generated_ids = []
        if _ids[-1] == 0:
            for el in range(len(_ids)):
                if el != 0 and _ids[el] == 0:
                    continue
                else:
                    generated_ids.append(_ids[el])
        else:
            generated_ids = _ids 
        assert generated_ids[-1] != 0
        generated_ids = generated_ids[:-1]
        
        cur_dict = trie 
        for i in range(len(generated_ids) - 1):
            cur_dict = cur_dict[int(generated_ids[i])]
        if len(cur_dict[generated_ids[-1]]) == 0:
            add_list.append(generated_ids)
        else:
            cur_dict = cur_dict[int(generated_ids[-1])]
            # check if it is an unique path
            keys = list(cur_dict.keys())
            if len(keys) > 1:
                print(f"More than 1: {keys}")
                keys = [keys[0]]
            while len(cur_dict[keys[0]]) != 0:
                generated_ids.append(keys[0])
                cur_dict = cur_dict[keys[0]]
                keys = list(cur_dict.keys())
                if len(keys) > 1:
                    print(f"More than 1: {keys}")
                    keys = [keys[0]]
            generated_ids.append(1)
        return generated_ids

    def _test_step(self, batch, batch_idx, return_elem=False):
       
        # for case where it resume test 
        test_input = []
        for _input in batch['input']:
            if _input in self.test_input_list: continue
            else: test_input.append(_input) 
        test_num = len(test_input)
        start_num = len(batch['input'])-test_num
        if test_num == 0: return None
        test_output = batch['output'][start_num:]
        test_source_ids = batch['source_ids'][start_num:]
        test_source_masks = batch['source_mask'][start_num:]
        test_target_ids = batch['target_ids'][start_num:]
        test_target_masks = batch['target_mask'][start_num:]
        assert len(test_output) == test_num, f'test_output: {len(test_output)}\ttest_num: {test_num}'
        _trie_list = [copy.deepcopy(self.trie_dict) for _ in range(test_num)]
        
        _generated_ids = self.model.generate(
             test_source_ids, 
             attention_mask=test_source_masks,
             use_cache=True,
             max_length=self.hparams.max_output_length,
             num_beams=self.hparams.val_beam_size,
             num_return_sequences=self.hparams.ret_num,
             prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get(
                 batch_id, sent.tolist(), scores
             ),
             early_stopping=True,
        ) # _generated_ids -> clusterId
        _generated_text = self.ids_to_text(_generated_ids, skip_special_tokens=True)
        inum = len(_generated_ids) // self.hparams.ret_num
        assert inum == len(batch['output']) 
        generated_text = [
            _generated_text[
                i * self.hparams.ret_num: (i + 1) * self.hparams.ret_num
            ]
            for i in range(inum)
        ]
        generated_ids = [
            _generated_ids[
                i * self.hparams.ret_num : (i+1) * self.hparams.ret_num
            ].detach().cpu().numpy().tolist()
            for i in range(inum)
        ]
        print(f"Text: {generated_text[0]}")
        print(f"Ids: {generated_ids[0]}")
        print(f"GT: {batch['output'][0]}")
        em_list, total_f1, vs_f1, es_f1 = self.calculate_scores(
            generated_text, batch["output"], batch["input"], batch_idx
        )
        if return_elem:
            assert (
                len(list(test_input))
                == len(list(generated_text))
                == len(list(em_list))
            )
            return {
                "input": list(test_input),
                "gt": list(test_output),
                "pred": list(generated_text),
                "pred_tok": list(generated_ids),
                "em": list(em_list),
                "total_f1": list(total_f1),
                "vs_f1": list(vs_f1),
                "es_f1": list(es_f1),
            }
        else:
            return em_list, total_f1, vs_f1, es_f1 


    def ids_to_text(self, _generated_ids, skip_special_tokens=True):
        generated_ids = []
        for _ids in _generated_ids:
            _ids = copy.deepcopy(_ids)
            _ids = _ids.detach().cpu().numpy()
            _text = [self.shared_tokId2tokText[_id] for _id in _ids]
            generated_ids.append(self.tokenizer.convert_tokens_to_ids(_text))
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True
        )
        text = []
        if not skip_special_tokens:
            for elem in gen_text: 
               rep_elem = elem.replace('<pad>', '')
               text.append(rep_elem)
        else:
            text = gen_text
        text = self.lmap(str.strip, text)
        return text



    def validation_step(self, batch, batch_idx):
        if self.hparams.save_model_only:
           self.convert_checkpoint("new_ckpt")
           print(f"Done saving checkpoint.. {self.current_epoch}")
           sys.exit()

        em_score, total_f1_score, vs_f1_score, es_f1_score = self._val_step(batch, batch_idx)
        self.em_score_list.extend(list(em_score))
        self.total_f1_score_list.extend(list(total_f1_score)) 
        self.vs_f1_score_list.extend(list(vs_f1_score)) 
        self.es_f1_score_list.extend(list(es_f1_score)) 

    def validation_epoch_end(self, outputs):
        avg_em = np.mean(np.array(self.em_score_list))
        avg_total_f1 = np.mean(np.array(self.total_f1_score_list))
        avg_vs_f1 = np.mean(np.array(self.vs_f1_score_list))
        avg_es_f1 = np.mean(np.array(self.es_f1_score_list))
        self.em_score_list = []
        self.total_f1_score_list = []
        self.vs_f1_score_list = []
        self.es_f1_score_list = []

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
            "val_total_f1",
            avg_total_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_vs_f1",
            avg_vs_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_es_f1",
            avg_es_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return


    def _iter_test_step(self, batch, batch_idx, return_elem=False):
        test_input = []
        for _input in batch["input"]:
            if _input in self.test_input_list: continue
            else: test_input.append(_input)
        test_num = len(test_input)
        start_num = len(batch['input'])-test_num 

        if test_num == 0: return None 
        test_output = [self.dev_input2output[el] for el in test_input] 
        test_source_ids = batch["source_ids"][start_num:]
        test_source_masks = batch["source_mask"][start_num:]
        test_target_ids = batch["target_ids"][start_num:]
        test_target_masks = batch["target_mask"][start_num:]
        assert len(test_output) == test_num, f"test_output: {len(test_output)}\ntest_num: {test_num}"

        _test_input = test_input

        test_pred = [[] for _ in range(len(batch["input"]))]
        iternum = np.array([len(el) for el in test_output]).max()

        for iter_idx in range(iternum):
           _generated_ids = self.model.generate(
                     test_source_ids,
                     attention_mask=test_source_masks,
                     use_cache=True,
                     max_length=self.hparams.max_output_length,
                     num_beams=self.hparams.val_beam_size,
                     num_return_sequences=self.hparams.ret_num,
                     prefix_allowed_tokens_fn=lambda batch_id, sent, scores: self.get(batch_id, sent.tolist(), scores),
                     early_stopping=True
                 )
           _generated_text = self.ids_to_text(_generated_ids)
           inum = len(_generated_ids) // self.hparams.ret_num 
           assert inum == len(batch['output'])
         
           generated_text = [
               _generated_text[
                  i*self.hparams.ret_num:(i+1)*self.hparams.ret_num
               ]
               for i in range(inum)
           ]


           generated_ids = [
               _generated_ids[
                  i*self.hparams.ret_num:(i+1)*self.hparams.ret_num
               ]
               for i in range(inum)
           ]

           for idx, elem in enumerate(generated_text):
               test_pred[idx].append(elem[0])
           test_source_ids, test_source_masks, _test_input = self.tokenize4next_hop(_test_input, generated_text)

        em_list, total_f1, vs_f1, es_f1 = self.iter_calculate_scores(
            test_pred, test_output
         )

        if return_elem:
            assert len(list(test_input))==len(list(generated_text))==len(list(em_list))
            return {
               "input": list(test_input),
               "gt": list(test_output),
               "pred": list(generated_text),
               "pred_tok": list(generated_ids),
               "em": list(em_list),
               "total_f1": list(total_f1),
               "vs_f1": list(vs_f1),
               "es_f1": list(es_f1),
            }
        else:
            return em_list, total_f1, vs_f1, es_f1

    def tokenize4next_hop(self, input_list, pred_list):
        sen_list = [f"{_input} {_pred[0]}" for _input, _pred in zip(input_list, pred_list)]
        tok_ret = self.tokenizer.batch_encode_plus(
                  sen_list,
                  max_length=self.hparams.max_input_length,
                  padding="max_length",
                  truncation=True,
                  return_tensors="pt"
              )
        return tok_ret["input_ids"].to(self.device), tok_ret["attention_mask"].to(self.device), sen_list

    def test_step(self, batch, batch_idx):
        if self.hparams.do_iteration:
           ret_dict = self._iter_test_step(batch, batch_idx, return_elem=True)
        else:
           ret_dict = self._test_step(batch, batch_idx, return_elem=True)
        if ret_dict is None: return None

        self.test_input_list.extend(ret_dict["input"])
        self.test_gt_list.extend(ret_dict["gt"])
        self.test_pred_list.extend(ret_dict["pred"])
        self.test_pred_tok_list.extend(ret_dict["pred_tok"])
        self.test_em_score_list.extend(ret_dict["em"])
        self.test_total_f1_score_list.extend(ret_dict["total_f1"])
        self.test_vs_f1_score_list.extend(ret_dict["vs_f1"])
        self.test_es_f1_score_list.extend(ret_dict["es_f1"])
        self._save_test() 

    def _save_test(self, epoch_end=False):
        os.makedirs(self.hparams.output_dir, exist_ok=True)
        _input = self.gather_list(self.test_input_list) 
        _gt = self.gather_list(self.test_gt_list)
        _pred = self.gather_list(self.test_pred_list)
        _pred_tok = self.gather_list(self.test_pred_tok_list)
        _em = self.gather_list(self.test_em_score_list)
        _total_f1 = self.gather_list(self.test_total_f1_score_list)
        _test_vs = self.gather_list(self.test_vs_f1_score_list)
        _test_es = self.gather_list(self.test_es_f1_score_list)
        assert len(_input) == len(_gt) == len(_pred) == len(_em) == len(_total_f1) == len(_test_vs) == len(_test_es) == len(_pred_tok)
        if self.print:
            with open(self.test_save_name, "w") as f:
                 json.dump(
                     {
                         "input": _input,
                         "gt": _gt,
                         "pred": _pred,
                         "pred_tok": _pred_tok,
                         "em": _em,
                         "total_f1": _total_f1,
                         "vs_f1": _test_vs,
                         "es_f1": _test_es,
                     },
                     f,
                 )
            if epoch_end:
                 print(
                     f"Saving in {self.test_save_name}!\nnumber of elements: {len(_input)}"
                 )
                 print(f"EM: {np.array(_em).mean()}")
                 print(f"Total F1: {np.array(_total_f1).mean()}")
                 print(f"Vs F1: {np.array(_test_vs).mean()}")
                 print(f"Es F1: {np.array(_test_es).mean()}")
    
    def test_epoch_end(self, outputs):
        self._save_test(epoch_end=True)

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
        try:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            self.save_epoch.append(self.current_epoch)
        except:
            print(f"Fail to save ... {self.current_epoch}")
        if len(self.save_epoch) > 5:
            rm_file = f"best_tfmr_{self.save_epoch[0]}"
            os.system(f"rm -rf {os.path.join(self.hparams.output_dir, rm_file)}")
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

    def convert_checkpoint(self, sub_path=None):
        if sub_path is None:
           save_path = os.path.join(
               self.hparams.output_dir, f"best_tfmr_{self.current_epoch}"
           )
        else:
           save_path = os.path.join(
               self.hparams.output_dir, sub_path     
           )
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.save_epoch.append(self.current_epoch)
        if len(self.save_epoch) > 5:
            rm_file = f"best_tfmr_{self.save_epoch[0]}"
            os.system(f"rm -rf {os.path.join(self.hparams.output_dir, rm_file)}")
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

