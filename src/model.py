import os
import re
import sys
import json
import uuid
import copy
import torch
import string
import pickle
import numpy 
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributed as dist

from data import GENREDataset
from blob import get_blob_info, upload_directory_to_blob

from transformers import T5Config, T5Tokenizer, T5Model, T5ForConditionalGeneration, BertTokenizer, Adafactor, AutoTokenizer
from torch.utils.data import DataLoader
from itertools import chain

from azure.storage.blob import (
    BlobServiceClient,
    BlobClient,
    ContainerClient,
    __version__,
)

class T5BaseClass(pl.LightningModule):
    def __init__(self):
        super(T5BaseClass, self).__init__()
        
        if torch.cuda.current_device() == 0:
            self.print = True 
        else:
            self.print = False 

    def _get_dataset(self, split):
        dataset = GENREDataset(
            tokenizer=self.tokenizer,
            split=split,
            hparams=self.hparams,
            tokid2emb=self.contextualized_tokid2emb 
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
        assert len(pred) == self.hparams.val_beam_size
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

class T5BiEncoder(T5BaseClass):
    def __init__(self, args):
        super(T5BiEncoder, self).__init__()
        self.save_hyperparameters(args)

        if self.hparams.do_train:
            self.model = T5Model.from_pretrained(self.hparams.model_name_or_path)
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_name_or_path)
            if self.print:
                print(f'@@@ In Training Mode...')
                print(f'@@@ Loading Model from {self.hparams.model_name_or_path}')
            self.em_score_list = []
            self.recall_score_list = []

            if self.hparams.periflow:
                self.connect_str, self.container_name = get_blob_info()
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)

        if self.hparams.do_test:
            self.model = T5Model.from_pretrained(self.hparams.test_model_path)
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

        self.contextualized_tokid2emb = pickle.load(open(self.hparams.contextualized_file, "rb"))
        self.contextualized_tensor = torch.tensor(list(self.contextualized_tokid2emb.values())).to(self.device)
        if self.hparams.fp16:
            self.contextualized_tensor = self.contextualized_tensor.half()
        self.contextualized_token = list(self.contextualized_tokid2emb.keys())
        self.tokId2corpus = pickle.load(open(self.hparams.tokId2corpus, "rb"))

        self.loss_fct = nn.CrossEntropyLoss()
        self.decoder_input_ids, self.decoder_attention_mask = self.get_decoder_input()

    def get_decoder_input(self):
        _tok = self.tokenizer("</s>", return_tensors='pt', add_special_tokens=False)
        _input_ids = _tok['input_ids'].to(self.device)
        _attention_mask = _tok['attention_mask'].to(self.device)
        return _input_ids, _attention_mask

    def forward(self, input_ids, attention_mask=None):
        bs = input_ids.shape[0]
        d_input = torch.cat([self.decoder_input_ids]*bs, 0).to(self.device)
        d_att = torch.cat([self.decoder_attention_mask]*bs, 0).to(self.device)

        outputs = self.model(input_ids, 
                        attention_mask=attention_mask,
                        decoder_input_ids=d_input, 
                        decoder_attention_mask=d_att
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
        query_output = query_outputs.last_hidden_state[:, 0]

        key_outputs = torch.cat([torch.tensor(self.contextualized_tokid2emb[key_input_id[0]]).unsqueeze(0) for key_input_id in key_input_ids], 0) 
        if self.hparams.fp16:
            key_outputs = key_outputs.half()
        return query_output, key_outputs

    def _calculate_similarity(self, batch):
        z1, z2 = self._get_embedding(batch)
        z1 = z1.to(self.device)
        z2 = z2.to(self.device)
        sim = torch.inner(z1, z2)
        return sim

    def _common_step(self, batch):
        _sim = self._calculate_similarity(batch)
        labels = torch.arange(_sim.size(0)).long().to(self.device)
        loss = self.loss_fct(_sim, labels)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch)
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

    def _calculate_score(self, indices, batch, batch_idx):
        em_list = []; recall_list = []; pred_list = []; pred_tok_list = []
        for idx, (_indice_list, _output, _input) in enumerate(zip(indices, batch['output'], batch['input'])):
            _predict = []; _pred_idx = []
            for _indice in _indice_list:
                tokId = self.contextualized_token[_indice]
                _pred_idx.append(tokId)
                if tokId == 0 or tokId == 1:
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
 
    def test_epoch_end(self, outputs):
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
            filename = f"{self.hparams.test_name}_group_{self.hparams.groupId_tree}_mbs_{self.hparams.max_beam_search}_result_beam{self.hparams.val_beam_size}.json"
            with open(
                os.path.join(
                    self.hparams.output_dir,
                    filename 
                ),
                "w",
            ) as f:
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
            print(
                f"Saving in {os.path.join(self.hparams.output_dir, filename)}!\nnumber of elements: {len(_input)}"
            )
            print(f"EM: {np.array(_em).mean()}")
            print(f"Recall: {np.array(_recall).mean()}")

class T5FineTuner(T5BaseClass):
    def __init__(self, args):
        super(T5FineTuner, self).__init__()
        self.save_hyperparameters(args)
        
        config = T5Config.from_pretrained(self.hparams.model_name_or_path)
        config.update({"fp16": self.hparams.fp16})
        config.update(
            {"contextualized_emb_num": self.hparams.contextualized_emb_num}
        )
        config.update(
            {"contextualized_file": os.path.join(self.hparams.dataset, self.hparams.contextualized_file)}
        )  # tokId_emb.pickle
        config.update({"freeze_vocab_emb": self.hparams.freeze_vocab_emb})

        # If in training mode, load ckpt for training
        if self.hparams.do_train:

            self.model = T5ForConditionalGeneration.from_pretrained(
                self.hparams.model_name_or_path, config=config
            )
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.hparams.model_name_or_path
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
            #raise NotImplementedError("Test Code is not implemented yet!")
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.hparams.test_model_path, config=config
            )
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.test_model_path)
            if self.print:
                print(f"@@@ Loading Model from {self.hparams.test_model_path}")
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
        ### REMEMBER!!! Values in trie_dict is "GroupID" not "tokId"
        self.group_trie = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.groupId_tree), "rb"))
        self.node_trie = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.nodeId_tree), "rb"))
        self.contextualized_tokid2emb = pickle.load(
            open(os.path.join(self.hparams.dataset, self.hparams.contextualized_file), "rb")
        )
        assert (
            len(self.contextualized_tokid2emb.keys())
            == int(self.hparams.contextualized_emb_num)
        ), f"contextualized_emb_num: {self.hparams.contextualized_emb_num} and length of keys: {len(self.contextualized_tokid2emb.keys())}"
        
        self.groupId2tokId= pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.groupId2tokIdList), "rb"))
        self.tokId2groupId = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2groupId), 'rb'))
        self.tokId2tokText = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.tokId2tokText), 'rb'))
        nodeId_sup = pickle.load(open(os.path.join(self.hparams.dataset, self.hparams.nodeId_sup), 'rb'))
        self.nodeId2groupdId = nodeId_sup['group_set']
        self.nodeId2tokId = nodeId_sup['token_set']
        self.groupId2nodeId = nodeId_sup['inv_group_set']
        self.tokId2nodeId = nodeId_sup['inv_token_set']
        #self.eos_list = list(self.groupId2tokId[1])

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

    def lmap(self, f, x):
        return list(map(f, x))

    def ids_to_text(self, _generated_ids):
        generated_ids = []
        for _ids in _generated_ids:
            _ids = copy.deepcopy(_ids)
            _ids = _ids.detach().cpu().numpy()
            _text = [self.tokId2tokText[_id] for _id in _ids]
            generated_ids.append(self.dec_tok.convert_tokens_to_ids(_text))
        gen_text = self.dec_tok.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def forward(
        self,
        input_ids,
        attention_mask,
        lm_labels,
        decoder_attention_mask,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=None,
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
            if self.hparams.tree_type == "groupId":
                trie_dict = self.group_trie
            elif self.hparams.tree_type == "nodeId":
                trie_dict = self.node_trie
            else:
                assert False
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

    def _val_step(self, batch, batch_idx, return_elem=False):
        # calculates recall and em -> returns the list of each score
        _generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
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
        if self.hparams.tree_type == "groupId":
            _trie_list = [copy.deepcopy(self.group_trie) for _ in range(self.hparams.eval_batch_size)]
            #_trie_dict = copy.deepcopy(self.group_trie)
        elif self.hparams.tree_type == "nodeId":
            _trie_list = [copy.deepcopy(self.node_trie) for _ in range(self.hparams.eval_batch_size)]
            #_trie_dict = copy.deepcopy(self.node_trie)
        else:
            assert False
        
        #unique_pred = []; unique_ids = []
        unique_pred_list = [[] for _ in range(self.hparams.eval_batch_size)] 
        unique_ids_list = [[] for _ in range(self.hparams.eval_batch_size)] 
        over = [0]*self.hparams.eval_batch_size
        while True:
            
            _generated_ids = self.model.generate(
                batch["source_ids"],
                attention_mask=batch["source_mask"],
                use_cache=True,
                decoder_attention_mask=batch["target_mask"],
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
            """ 
            for b_texts, b_ids in zip(generated_text, generated_ids):
                for _text, _ids in zip(b_texts, b_ids):
                    g_ids = [self.tokId2groupId[el] for el in _ids]
            """
            for bid, (batch_text, batch_ids) in enumerate(zip(generated_text, generated_ids)):
                if over[bid] == 1: continue
                ### iterate over batch (val_beam_size)
                _upper_ids = []
                _trie_dict = _trie_list[bid]
                for _text, _ids in zip(batch_text, batch_ids):
                    if _text not in unique_pred_list[bid] and len(unique_pred_list[bid])<self.hparams.val_beam_size:
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
                # remove from _trie_dict
                _trie_dict = self._remove_prev_from_trie(_trie_dict, _upper_ids)
                _trie_list[bid] = _trie_dict
                if len(unique_pred_list[bid]) >= self.hparams.val_beam_size:
                    over[bid] = 1
            if over.count(1) == self.hparams.eval_batch_size:
                break

        for i in range(len(unique_pred_list)):
            unique_pred_list[i] = unique_pred_list[i][:self.hparams.val_beam_size]

        #self._flush_first_beam_dict()
        #if self.print: print(f"## UNIQUE PRED: {unique_pred[:self.hparams.val_beam_size]}")
        em_list, recall_list = self.calculate_scores(
            unique_pred_list, batch["output"], batch["input"], batch_idx
        )
        if return_elem:
            assert (
                len(list(batch["input"]))
                == len(list(generated_text))
                == len(list(em_list))
                == len(list(unique_pred_list))
                == len(list(unique_ids_list))
            )
            return {
                "input": list(batch["input"]),
                "gt": list(batch["output"]),
                "gt_tok": list(batch["target_ids"].detach().cpu().numpy().tolist()),
                "pred": list(unique_pred_list),
                "pred_tok": list(unique_ids_list),
                "em": list(em_list),
                "recall": list(recall_list),
            }
        else:
            return em_list, recall_list


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
        ret_dict = self._test_step(batch, batch_idx, return_elem=True)
        self.test_input_list.extend(ret_dict["input"])
        self.test_gt_list.extend(ret_dict["gt"])
        self.test_gt_tok_list.extend(ret_dict["gt_tok"])
        self.test_pred_list.extend(ret_dict["pred"])
        self.test_pred_tok_list.extend(ret_dict["pred_tok"])
        self.test_em_score_list.extend(ret_dict["em"])
        self.test_recall_score_list.extend(ret_dict["recall"])

    def test_epoch_end(self, outputs):
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
            with open(
                os.path.join(
                    self.hparams.output_dir,
                    f"{self.hparams.tree_type}_mbs_{self.hparams.max_beam_search}_result_beam{self.hparams.val_beam_size}.json",
                ),
                "w",
            ) as f:
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
            print(
                f"Saving in {os.path.join(self.hparams.output_dir)}!\nnumber of elements: {len(_input)}"
            )
            print(f"EM: {np.array(_em).mean()}")
            print(f"Recall: {np.array(_recall).mean()}")
