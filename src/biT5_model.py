import os
import sys
import json
import torch
import faiss
import numpy
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributed as dist 
import pickle
import string
import re

from data import GENREDataset 

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader 
from transformers import T5Tokenizer, T5EncoderModel , T5Model
from transformers import BertTokenizer, Adafactor, AutoTokenizer
import faiss.contrib.torch_utils

class BiT5Model(pl.LightningModule):
   def __init__(self, hparams):
      super(BiT5Model, self).__init__()

      if torch.cuda.current_device() == 0:
         self.print=True
      else:
         self.print=False 

      self.save_hyperparameters(hparams)

      if self.hparams.do_test:
         assert not self.hparams.do_train
         raise NotImplementedError("Test Code is not Implemented Yet") 
      else:
         assert self.hparams.do_train
         self.model = T5EncoderModel.from_pretrained(self.hparams.model_name_or_path)
         self.tokenizer= T5Tokenizer.from_pretrained(self.hparams.model_name_or_path)

      if self.hparams.do_test:
         assert not self.hparams.do_train
         raise NotImplementedError("Test Code is not Implemented Yet") 
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

      if self.hparams.do_train:
         assert self.hparams.do_train
         self.model = T5EncoderModel.from_pretrained(self.hparams.model_name_or_path)
         self.tokenizer= T5Tokenizer.from_pretrained(self.hparams.model_name_or_path)
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

         # if self.hparams.periflow:
         #       self.connect_str, self.container_name = get_blob_info()
         #       self.blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)
      
      self.corpus = self.construct_corpus()

      self.dev_input2output = self.construct_dev_input2output()
      self.tokId2corpus = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2corpus), "rb"))
      self.em_score_list = []
      self.recall_score_list = []

      if self.hparams.cluster_num < 0:
         self.corpus2tokId = self._get_corpus2tokId()
      else:
         self.corpus2tokId = self._get_corpus2clusterId()

      self.loss_fct = nn.CrossEntropyLoss()
      self.decoder_input_ids, self.decoder_attention_mask = self.get_decoder_input()

   def construct_dev_input2output(self):
      input2output = defaultdict(list)
      f = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.dev_file), "rb"))
      for _input, _output in zip(f["input"], f["output"]):
         input2output[_input].append(_output)
      return input2output
   
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

   def get_dataset(self, split):
      dataset = GENREDataset(
         tokenizer=self.tokenizer,
         split=split,
         hparams=self.hparams,
         tokid2emb=self.contextualized_tokid2emb 
      )
      return dataset
   
   def train_dataloader(self):
      train_dataset = self.get_dataset("train")
      dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
      return dataloader

   def val_dataloader(self):
      val_dataset = self.get_dataset("validation")
      dataloader = DataLoader(val_dataset, shuffle=False, batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
      return dataloader 

   def test_dataloader(self):
      return

   def get_decoder_input(self):
      _tok = self.tokenizer("</s>", return_tensors='pt', add_special_tokens=False)
      _input_ids = _tok['input_ids'].to(self.device)
      _attention_mask = _tok['attention_mask'].to(self.device)
      return _input_ids, _attention_mask

   def construct_corpus(self):
      # index = faiss.IndexFlatIP(1024)
      # # # add corpus to Index  
      # corp = np.memmap(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), dtype="float32", shape=(37,1024), mode="readonly")
      # # if self.print: print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Construct FAISS index!")
      # for elem in tqdm(corp):
      #    # elem = np.expand_dims(elem, axis=0)
      #    elem = torch.unsqueeze(torch.from_numpy(elem),0)
      #    index.add(elem)
      # if self.print: print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Done FAISS index!")
      # if self.print: print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Construct FAISS index!")

      print('!!! Loading Embedding !!!')
      if self.hparams.faiss_file.endswith('dat'):
         self.faiss = True
         self.contextualized_tokid2emb = np.memmap(os.path.join(self.hparams.dataset, self.hparams.contextualized_file),dtype="float32" ,mode="readonly", shape=(3700000, 1024))#37000000
         if "faiss.index" in os.listdir(self.hparams.dataset):
               index = faiss.read_index(os.path.join(self.hparams.dataset, "faiss.index"))
         else:
               print(f'BUILD Faiss Index!! ')
               index = faiss.IndexFlatIP(1024)
               # self.contextualized_token = []
               for elem in tqdm(self.contextualized_tokid2emb):
                  # emb = np.expand_dims(emb, axis=0)
                  emb = torch.unsqueeze(torch.from_numpy(elem),0)
                  index.add(emb)
               print(index.ntotal)
               
         gpu_index = faiss.index_cpu_to_all_gpus(index)

         return gpu_index
      elif self.hparams.faiss_file.endswith('pickle'):
         assert False
         self.faiss = False 
         self.contextualized_tokid2emb = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), "rb"))
         corpus = torch.tensor(list(self.contextualized_tokid2emb.values()))#.to(self.device)
         if self.hparams.fp16:
               assert False
               self.corpus = self.corpus.half()
         # self.contextualized_token = list(self.contextualized_tokid2emb.keys())
         return corpus
      else:
         # assert False
         self.faiss = True
         # self.contextualized_tokid2emb = pickle.load(
         #    open(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), "rb")
         # )
         self.contextualized_tokid2emb = np.memmap(os.path.join(self.hparams.dataset, self.hparams.contextualized_file),dtype="float32" ,mode="readonly", shape=(3700000, 1024))#37000000

         print("starting to load faiss index")
         assert torch.cuda.is_available(), f"Cuda availability {torch.cuda.is_available()}"
         index = faiss.read_index(os.path.join(self.hparams.dataset, self.hparams.faiss_file))
         gpu_index = faiss.index_cpu_to_all_gpus(index)

      return gpu_index

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

   # def _get_embedding(self, batch): ## query output 평균내서 하나로 만들기 
   #    # query
   #    # assert False, print(f"the output is {batch}")
   #    query_outputs = self(input_ids=batch["source_ids"])
   #    # print(query_outputs.keys())

   #    query_output = torch.mean(query_outputs.last_hidden_state,dim=1)
   #    # assert False, print(f"query size: {query_outputs.last_hidden_state.size()} \n mean query_size: {query_output.size()}")

   #    return query_output
   
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

      # key_outputs = torch.cat([torch.tensor(self.contextualized_tokid2emb[key_input_id[0]]).unsqueeze(0) for key_input_id in key_input_ids], 0) ## 나중에 이걸로 바꿔서 사용하세요
      key_outputs = torch.cat([torch.tensor(self.contextualized_tokid2emb[key_input_id[0]%3700000]).unsqueeze(0) for key_input_id in key_input_ids], 0) ## 나중에 이걸로 바꿔서 사용하세요


      if self.hparams.fp16:
         key_outputs = key_outputs.half()
      return query_output, key_outputs

   ## int8, in8 계산이 아니고, fp32 그대로 사용하면 문제 없는지 확인 
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
   
   def sort_index(self,scores,ind):
      # _scores = zip(scores,ind)
      # index_scores = sorted(_scores, key=lambda x:x[1])

      # return [s for s,i in index_scores]
      result = [0 for _ in ind]
      for s,i in zip(scores,ind):
         result[i] = s
      return result

   # def _calculate_similarity(self, batch):
   #    z1 = self._get_embedding(batch).detach().cpu() ## 1. cpu가 아닌 gpu에 있는 tensor를 index.search를 못하게 막아두었다 
   #    ## get scores and index from faiss search
      

   #    b_scores, I = self.corpus.search(z1, self.corpus.ntotal) #  self.hparams.val_beam_size
   #    _scores = zip(b_scores,I)
   #    # __scores = [(s,i) for s,i in _scores]
   #    # assert False, print(f'\n z1 is {z1.size()} \n b_scores is {len(b_scores)},{len(b_scores[0])}\n b_scores is {len(b_scores)},{len(b_scores[0])},\n I is {len(I)},{len(I[0])}')
   #    # index_scores = sorted(_scores, key=lambda x:x[1])
   #    result=[self.sort_index(s,i) for s,i in _scores]
   #    sim = torch.tensor(result).to(self.device) ## use scores sorted by index; index_scores[1]-> index in ascending order 
   #    # sim = torch.inner(z1, self.corpus) # self.corpus too large to calculate? ## index 순서로 sort 해서 score  -> torch tensor로 바꿔서 전달 
   #    return sim
   

   def _calculate_similarity(self, batch, total=False):
      z1, z2 = self._get_embedding(batch) # z1: query, z2: corpus 개수

      if total:
         # sim = torch.inner(z1, self.corpus.to(self.device)) # sim: [query 개수, corpus 개수]
         # print(self.corpus.ntotal)
         sim, ind = self.corpus.search(z1.detach().cpu(),2048) #self.corpus.ntotal
         sim.requires_grad_()
      else:
         z1 = z1.to(self.device)
         z2 = z2.to(self.device)
         sim = torch.inner(z1, z2) 
      return sim

   # def _common_step(self, batch):
   #    _sim = self._calculate_similarity(batch)
   #    labels = torch.zeros_like(_sim)
   #    for i, ids in enumerate(batch["target_ids"]):
   #       labels[i][ids] = 1
   #    labels = torch.tensor(labels).float().to(self.device)
   #    loss = self.loss_fct(_sim, labels)
   #    return loss
   
   def _common_step(self, batch):
      _sim = self._calculate_similarity(batch)
      labels = torch.arange(_sim.size(0)).to(self.device)
      loss = self.loss_fct(_sim, labels)
      return loss
   

   def _total_common_step(self, batch, all=False):
      _sim = self._calculate_similarity(batch, total=True).to(self.device)
      labels = torch.zeros_like(_sim)
      # for i, (ids, corpus) in enumerate(zip(batch['target_ids'], batch['output'])):
      #    all_ids = self.corpus2tokId[corpus]
      #    assert ids[0] in all_ids, f'ids: {ids} // all_ids: {all_ids}'
      #    for ids in all_ids:
      #          labels[i][ids] = 1
         #assert torch.count_nonzero(labels[i]).item() == len(all_ids)
      labels = torch.tensor(labels).float().to(self.device)
      loss = self.loss_fct(_sim, labels)
      # loss.requires_grad(True)
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
      
      # self.log(
      #    "train_loss",
      #    loss, 
      #    on_step=True,
      #    on_epoch=True,
      #    prog_bar=False,
      #    logger=True,
      #    sync_dist=True,
      # )
      return loss 
   
   # def validation_step(self, batch, batch_idx):
   #    _sim = self._calculate_similarity(batch)
   #    # z1 = self._get_embedding(batch)
   #    # S, indices = self.corpus.search(z1, self.hparams.val_beam_size) # z1.detach().cpu().numpy()
   #    em_score, recall_score = self._calculate_score(_sim, batch)
   #    self.em_score_list.extend(list(em_score))
   #    self.recall_score_list.extend(list(recall_score))
   #    return
   

   def validation_step(self, batch, batch_idx):
      query_output, _ = self._get_embedding(batch)
      if self.faiss:
         _, indices = self.corpus.search(query_output.detach().cpu(), self.hparams.val_beam_size)
      else:
         scores = torch.inner(query_output.to('cpu'), self.corpus)
         top_scores = torch.topk(scores, self.hparams.val_beam_size)
         indices = top_scores.indices # [# of query, self.hparams.val_beam_size]
         #scores = torch.inner(query_output.to(self.device), self.corpus.to(self.device)) # [# of query, # of corpus] 
      assert len(indices) == len(batch['output'])

      em_score, recall_score, = self._calculate_score(indices, batch, batch_idx)
      self.em_score_list.extend(list(em_score))
      self.recall_score_list.extend(list(recall_score))

   def validation_epoch_end(self, outputs):
      avg_em = np.mean(np.array(self.em_score_list))
      avg_recall = np.mean(np.array(self.recall_score_list))
      self.em_score_list = []; self.recall_score_list = []
      # self.log(
      #    "val_em",
      #    avg_em,
      #    on_step=False,
      #    on_epoch=True,
      #    prog_bar=True,
      #    logger=True,
      #    sync_dist=True,
      # )
      # self.log(
      #    "val_recall",
      #    avg_recall,
      #    on_step=False,
      #    on_epoch=True,
      #    prog_bar=True,
      #    logger=True,
      #    sync_dist=True,
      # )
      return 

   def _calculate_score(self, indices, batch,batch_idx):
      em_list = []; recall_list = []
      for idx, (ind_list, _input) in enumerate(zip(indices, batch["input"])):
         _output = self.dev_input2output[_input]
         _predict = []; _pred_idx = []
         for ind in ind_list:
            if ind == 0 or ind == 1: ## tokId
               _predict.append(None)      
            else:
               # _predict.append(self.tokId2corpus[ind])
               _predict.append(None)
         em_list.append(self._calculate_em(_predict[0], _output))
         recall_list.append(self._calculate_recall(_predict, _output))
      return em_list, recall_list
   


   def _calculate_em(self, pred, gt):
      if pred is None: return 0
      if type(gt) == list:
         gt = [self.normalize_answer(el) for el in gt]
         if self.normalize_answer(pred) in gt:
            return 100
         else:
            return 0 
      else:
         assert False

   def _calculate_recall(self, pred, gt):
      if type(gt) == list:
         gt = [self.normalize_answer(el) for el in gt]
         for elem in pred:
            if elem is None: continue 
            if self.normalize_answer(elem) in gt:
               return 100
      else: 
         assert False 
      return 0

   def test_step(self, batch, batch_idx):
      return 

   def test_epoch_end(self, outputs):
      return 

   def configure_optimizers(self):
      no_decay = ["bias", "LayerNorm.weight"]
      optimizer_grouped_parameters = [
         {
            "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.hparams.weight_decay,
         },
         {
            "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
         }
      ]
      
      optimizer = Adafactor(optimizer_grouped_parameters, lr=self.hparams.learning_rate, warmup_init=False, scale_parameter=False, relative_step=False)
      self.opt = optimizer 

      if self.hparams.lr_scheduler == "constant":
         return [optimizer]
      elif self.hparams.lr_scheduler == "exponential":
         len_data = len(self.train_dataloader())
         denominator = self.hparams.n_gpu
         steps_per_epoch=((len_data//denominator)+1)//self.hparams.gradient_accumulation_steps
         scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate, steps_per_epoch=steps_per_epoch, pct_start=0.1, epochs=self.hparams.num_train_epochs, anneal_strategy="linear", cycle_momentum=False)
         return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": "learning_rate"}]
      else:
         assert False

   def on_save_checkpoint(self, checkpoint):
      save_path = os.path.join(self.hparams.output_dir, f"best_tfmr_epoch_{self.current_epoch}")
      self.model.save_pretrained(save_path)
      self.tokenizer.save_pretrained(save_path)
      return




