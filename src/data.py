import os
import sys
import copy
import torch
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict
from torch.utils.data import Dataset

class SentenceDataset(Dataset):
    def __init__(self, tokenizer, split, hparams):
        self.hparams = hparams 
        self.tokenizer = tokenizer

        if split == "train":
            data_path = self.hparams.train_file
        elif split == "validation":
            data_path = self.hparams.dev_file
        elif split == "test":
            data_path = self.hparams.test_file
        else:
            raise NotImplementedError("Check the split type in SentenceDataset!")

        assert data_path.endswith('.csv')
        self.dataset = load_dataset("csv", data_files=os.path.join(self.hparams.dataset, data_path))['train']
        
        #column_names = self.dataset.column_names
        self.sent0_cname = "input" 
        self.sent1_cname = "output" 

    def __len__(self):
        return len(self.dataset[self.sent0_cname]) 

    def convert_to_features(self, query, key, idx):
        #input_ = example_batch[self.sent0_cname] + example_batch[self.sent1_cname]
        output_ = '<pad>' 

        query_ = self.tokenizer.batch_encode_plus(
            [query],
            max_length=self.args.query_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        key_ = self.tokenizer.batch_encode_plus(
            [key],
            max_length=self.args.key_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [output_],
            return_tensors="pt"
        )

        if idx == 0:
            print(f"##########\nquery: {query}\nkey: {key}\noutput: {output_}\n\n")
        return query_, key_, target

    def get_corpus_tokens(self):
        print(f'### Get Corpus Token')
        ret_dict = {'corpus_sen': [], 'corpus_ids': [], 'corpus_mask': [], 'target_ids': [], 'target_mask': []}
        for key in tqdm(self.corpus):
            key_ = self.tokenizer.batch_encode_plus(
                    [key],
                    max_length=self.args.key_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                    )
            target =self.tokenizer.batch_encode_plus(
                    ['<pad>'],
                    return_tensors='pt'
                    )
            ret_dict['corpus_sen'].append(key)
            ret_dict['corpus_ids'].append(key_['input_ids'])
            ret_dict['corpus_mask'].append(key_['attention_mask'])
            ret_dict['target_ids'].append(target['input_ids'][:, :1])
            ret_dict['target_mask'].append(target['attention_mask'][:, :1])
        return ret_dict

    def __getitem__(self, idx):
        query = self.dataset[self.sent0_cname][idx]
        key = self.dataset[self.sent1_cname][idx]
        query_, key_, target = self.convert_to_features(query, key, idx)

        query_ids = query_['input_ids'].squeeze()
        query_mask = query_['attention_mask'].squeeze()
       
        key_ids = key_['input_ids'].squeeze()
        key_mask = key_['attention_mask'].squeeze()

        target_ids = target['input_ids'].squeeze()[:1]
        target_mask = target['attention_mask'].squeeze()[:1]

        return {
            "query_ids": query_ids,
            "query_mask": query_mask,
            "key_ids": key_ids,
            "key_mask": key_mask,
            "decoder_ids": target_ids,
            "decoder_mask": target_mask,
            "query": query,
            "key": key 
        }        


class JOINTDataset(Dataset):
    def __init__(self, tokenizer, split, hparams):
        self.hparams = hparams
        if split == "train":
            data_path = self.hparams.train_file
        elif split == "validation":
            data_path = self.hparams.dev_file
        elif split == "test":
            data_path = self.hparams.test_file
        else:
            raise NotImplementedError(f"Inappropriate split type: {split}")
        
        assert data_path.endswith(".csv"), "Only csv file is possible!"
        self.dataset = pd.read_csv(os.path.join(self.hparams.dataset, data_path)) # key - input, output, output_tokid, output_tokemb
        self.len = len(self.dataset)
        if torch.cuda.current_device() == 0:
            print(
                f"@@@ Loading from {os.path.join(self.hparams.dataset, data_path)}: {self.len}"
            )

        self.tokenizer = tokenizer

    def __len__(self):
        return self.len 

    def convert_to_features(self, batch, idx):
        input_ = batch['query']
        title_ = batch['title']
        context_ = batch['context']
        if type(context_) != str:
            context_ = ""

        source = self.tokenizer.batch_encode_plus(
            [input_],
            max_length=self.hparams.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return source, input_, title_, context_ 

    def __getitem__(self, idx):
        source, input_, title_, context_ = self.convert_to_features(
            self.dataset.iloc[idx], idx
        )
        source_ids = source["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()

        #print(f'source_ids: {source_ids}\nsource_mask: {src_mask}\ntitle: {title_}\ncontext: {context_}\ninput: {input_}\n\n')
        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "title": title_, 
            "context": context_, 
            "input": input_,
            "output": title_,
        }

class TESTDataset(Dataset):
    def __init__(self, tokenizer, split, hparams, tokid2emb, corpus_tokenList_dict=None):
        self.hparams = hparams
        self.split = split
        if split == "train":
            data_path = self.hparams.train_file
        elif split == "validation":
            data_path = self.hparams.dev_file
        elif split == "test":
            data_path = self.hparams.test_file
        else:
            raise NotImplementedError(f"Inappropriate split type: {split}")

        assert data_path.endswith(".pickle"), "Only pickle file is possible!"
        data_dict = pickle.load(open(os.path.join(self.hparams.dataset, data_path), "rb")) # key - input, output, output_tokid, output_tokemb
        
        if split == "test": 
            self.dataset = {'input': [], 'output': [], 'output_tokid': []}
            for _input, _output, _output_tok_id in zip(data_dict['input'], data_dict['output'], data_dict['output_tokid']):
                if self.hparams.model_type=="multihop":
                    _input = _input.split('<P1>')[0]
                    while _input.endswith(" "): _input = _input[:-1]
               
                if self.hparams.do_title:
                    _input = _input.split("<title>")[0]
                    while _input.endswith(" "): _input = _input[:-1]

                if _input in self.dataset['input']: 
                    continue 
                else:
                    self.dataset['input'].append(_input)
                    self.dataset['output'].append(_output)
                    self.dataset['output_tokid'].append(_output_tok_id)
            self.dataset = pd.DataFrame(self.dataset)
        else:
            self.dataset = pd.DataFrame(data_dict)

        print(f"# of dataset: {len(self.dataset)}")
        self.len = len(self.dataset)
        if torch.cuda.current_device() == 0:
            print(
                f"@@@ Loading from {os.path.join(self.hparams.dataset, data_path)}: {self.len}"
            )

        self.tokenizer = tokenizer
        self.tokid2emb = tokid2emb

    def __len__(self):
        return self.len

    def convert_to_features(self, batch, idx):
        input_ = batch["input"]
        output_ = batch["output"]

        if self.hparams.change_enc:
            assert False 
        else:
            source = self.tokenizer.batch_encode_plus(
               [input_],
               max_length=self.hparams.max_input_length,
               padding="max_length",
               truncation=True,
               return_tensors="pt",
            )

        if self.hparams.change_dec:
            target = batch["output_tokid"]  # load from file
            if len(target) > self.hparams.max_output_length:
               target = target[: self.hparams.max_output_length]
               att = [1] * self.hparams.max_output_length
            else:
               _leftover = self.hparams.max_output_length - len(target)
               att = [1] * len(target) + [0] * _leftover
               target = target + [0] * _leftover
            assert (
               len(target) == self.hparams.max_output_length
               and len(att) == self.hparams.max_output_length
            ), print(f"length of target: {len(target)}\nlength of attention:  {len(att)}")
           
            target_idx = torch.tensor([target])
            att = torch.tensor([att])
            target = {"input_ids": target_idx, "attention_mask": att}
        else:
            target = self.tokenizer.batch_encode_plus(
               [output_],
               max_length=self.hparams.max_output_length,
               padding="max_length",
               truncation=True,
               return_tensors="pt"
            )

        if idx == 0 and torch.cuda.current_device() == 0:
            print(f"=" * 80)
            print(f"input: {input_}")
            print(f"output: {output_}")
            print(f"source: {source}")
            print(f"target: {target}")
            print(f"=" * 80)

        return source, target, input_, output_


    def __getitem__(self, idx):
        source, target, input_, output_ = self.convert_to_features(
            self.dataset.iloc[idx], idx
        )
        source_ids = source["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "target_ids": target_ids,
            "source_mask": src_mask,
            "target_mask": target_mask,
            "input": input_,
            "output": output_,
        }



class GENREDataset(Dataset):
    def __init__(self, tokenizer, split, hparams, tokid2emb, corpus_tokenList_dict=None):
        self.hparams = hparams
        self.split = split
        if split == "train":
            data_path = self.hparams.train_file
        elif split == "validation":
            data_path = self.hparams.dev_file
        elif split == "test":
            data_path = self.hparams.test_file
        else:
            raise NotImplementedError(f"Inappropriate split type: {split}")

        assert data_path.endswith(".pickle"), "Only pickle file is possible!"
        data_dict = pickle.load(open(os.path.join(self.hparams.dataset, data_path), "rb")) # key - input, output, output_tokid, output_tokemb
        
        if self.hparams.reload_dataloader_every_n_epochs and corpus_tokenList_dict is not None:
            for i, _output in enumerate(data_dict["output"]):
                data_dict["output_tokid"][i] = corpus_tokenList_dict[_output]

        if split == "test": 
            self.dataset = {'input': [], 'output': [], 'output_tokid': []}
            for _input, _output, _output_tok_id in zip(data_dict['input'], data_dict['output'], data_dict['output_tokid']):
                if self.hparams.model_type=="multihop":
                    _input = _input.split('<P1>')[0]
                    while _input.endswith(" "): _input = _input[:-1]
               
                if self.hparams.do_title:
                    _input = _input.split("<title>")[0]
                    while _input.endswith(" "): _input = _input[:-1]

                if _input in self.dataset['input']: 
                    continue 
                else:
                    self.dataset['input'].append(_input)
                    self.dataset['output'].append(_output)
                    self.dataset['output_tokid'].append(_output_tok_id)
            self.dataset = pd.DataFrame(self.dataset)
        else:
            self.dataset = pd.DataFrame(data_dict)

        print(f"# of dataset: {len(self.dataset)}")
        self.len = len(self.dataset)
        if torch.cuda.current_device() == 0:
            print(
                f"@@@ Loading from {os.path.join(self.hparams.dataset, data_path)}: {self.len}"
            )

        self.tokenizer = tokenizer
        self.tokid2emb = tokid2emb
        self.source_dict = {}

    def __len__(self):
        return self.len

    def convert_to_features(self, batch, idx):
        input_ = batch["input"]
        output_ = batch["output"]

        if input_ not in self.source_dict.keys():
            source = self.tokenizer.batch_encode_plus(
                [input_],
                max_length=self.hparams.max_input_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.source_dict[input_] = source 
        else:
            source = self.source_dict[input_]

        target = batch["output_tokid"]  # load from file
        if len(target) > self.hparams.max_output_length:
            target = target[: self.hparams.max_output_length]
            att = [1] * self.hparams.max_output_length
        else:
            _leftover = self.hparams.max_output_length - len(target)
            att = [1] * len(target) + [0] * _leftover
            target = target + [0] * _leftover
        assert (
            len(target) == self.hparams.max_output_length
            and len(att) == self.hparams.max_output_length
        ), print(f"length of target: {len(target)}\nlength of attention:  {len(att)}")
        
        target_idx = torch.tensor([target])
        att = torch.tensor([att])
        target = {"input_ids": target_idx, "attention_mask": att}

        if idx == 0 and torch.cuda.current_device() == 0:
            print(f"=" * 80)
            print(f"input: {input_}")
            print(f"output: {output_}")
            print(f"source: {source}")
            print(f"target: {target}")
            print(f"=" * 80)

        return source, target, input_, output_


    def __getitem__(self, idx):
        source, target, input_, output_ = self.convert_to_features(
            self.dataset.iloc[idx], idx
        )
        source_ids = source["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "target_ids": target_ids,
            "source_mask": src_mask,
            "target_mask": target_mask,
            "input": input_,
            "output": output_,
        }

class MEANDataset(Dataset):
    def __init__(self, tokenizer, split, hparams, tokid2emb, corpus2EmbMean):
        self.hparams = hparams
        if split == "train":
            data_path = self.hparams.train_file
        elif split == "validation":
            data_path = self.hparams.dev_file
        elif split == "test":
            data_path = self.hparams.test_file
        else:
            raise NotImplementedError(f"Inappropriate split type: {split}")

        assert data_path.endswith(".pickle"), "Only pickle file is possible!"
        data_dict = pickle.load(open(os.path.join(self.hparams.dataset, data_path), "rb")) # key - input, output, output_tokid, output_tokemb
        self.dataset = pd.DataFrame(data_dict)
        self.len = len(self.dataset)
        if torch.cuda.current_device() == 0:
            print(
                f"@@@ Loading from {os.path.join(self.hparams.dataset, data_path)}: {self.len}"
            )

        self.tokenizer = tokenizer
        self.tokid2emb = tokid2emb
        self.corpusList = list(corpus2EmbMean.keys())
        self.source_dict = {}

    def __len__(self):
        return self.len

    def convert_to_features(self, batch, idx):
        input_ = batch["input"]
        output_ = batch["output"]
        corpusId = self.corpusList.index(output_)

        # label for corpusId
        c_label = torch.zeros(len(self.corpusList))
        c_label[corpusId] = 1

        if input_ not in self.source_dict.keys():
            source = self.tokenizer.batch_encode_plus(
                [input_],
                max_length=self.hparams.max_input_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.source_dict[input_] = source 
        else:
            source = self.source_dict[input_]

        target = batch["output_tokid"]  # load from file

        max_output_length = self.hparams.max_output_length - 1 
        if len(target) > max_output_length:
            target = target[: max_output_length]
            att = [1] * max_output_length
        else:
            att = [1] * len(target) + [0] * (max_output_length-len(target)) 
            target = target + [self.tokenizer.pad_token_id] * (max_output_length-len(target)) 

        # add mean token
        target = [corpusId]+target
        att = [1]+att

        assert (
            len(target) == self.hparams.max_output_length
            and len(att) == self.hparams.max_output_length
        ), print(f"length of target: {len(target)}\nlength of attention:  {len(att)}")

        target_idx = torch.tensor([target])
        att = torch.tensor([att])
        target = {"input_ids": target_idx, "attention_mask": att}

        if idx == 0 and torch.cuda.current_device() == 0:
            print(f"=" * 80)
            print(f"input: {input_}")
            print(f"output: {output_}")
            print(f"source: {source}")
            print(f"target: {target}")
            print(f"=" * 80)

        return source, target, input_, output_, c_label

    def __getitem__(self, idx):
        source, target, input_, output_, c_label = self.convert_to_features(
            self.dataset.iloc[idx], idx
        )
        source_ids = source["input_ids"].squeeze()
        target_ids = target["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "target_ids": target_ids,
            "source_mask": src_mask,
            "target_mask": target_mask,
            "c_label": c_label,
            "input": input_,
            "output": output_,
        }

class ENTAILDataset(Dataset):
    def __init__(self, tokenizer, split, hparams, ee_tokid, es_tokid, corpus_tokenList, modelId2sharedId, tokId2clusterId, vd_mask, npd_mask):
        self.hparams = hparams
        self.split = split
        if split == "train":
            data_path = self.hparams.train_file
        elif split == "validation":
            data_path = self.hparams.dev_file
        elif split == "test":
            data_path = self.hparams.test_file
        else:
            raise NotImplementedError(f"Inappropriate split type: {split}")

        assert data_path.endswith(".csv"), "Only csv file is possible!"
        self.dataset = pd.read_csv(os.path.join(self.hparams.dataset, data_path))
        print(f"# of dataset: {len(self.dataset)}")
        self.len = len(self.dataset)
        if torch.cuda.current_device() == 0:
            print(
                f"@@@ Loading from {os.path.join(self.hparams.dataset, data_path)}: {self.len}"
            )

        self.tokenizer = tokenizer
        self.ee_tokid=ee_tokid
        self.es_tokid=es_tokid
        self.corpus_tokenList=corpus_tokenList
        self.modelId2sharedId=modelId2sharedId
        self.tokId2clusterId=tokId2clusterId
        self.vd_mask=vd_mask
        self.npd_mask=npd_mask
        if self.modelId2sharedId is not None:
           self.zeros=[0]*len(self.npd_mask)

    def __len__(self):
        return self.len

    def _iter_split(self, sen_list, sp):
        ret_list = []
        for sen in sen_list:
            ret_list.extend(sen.split(sp))
        return ret_list

    def get_ret_sen(self, sen, is_input):

        if is_input:
           sen = sen.split("[Es]")
           sen = sen[1:]
           ret_sen_list = []
           for elem in sen:
               assert "Ee" in elem 
               elem = elem.replace("[Ee]", "")
               while elem.startswith(" "): elem = elem[1:]
               while elem.endswith(" "): elem = elem[:-1]
               ret_sen_list.append(elem)
        else:
           if type(sen) != list: sen = [sen]
           sen = self._iter_split(sen, "[BECAUSE]") 
           sen = self._iter_split(sen, "[AND]") 
           sen = self._iter_split(sen, "[INFER]")

           ret_sen_list = []
           for elem in sen:
               if "[Es]" in elem:
                  assert "[Ee]" in elem
                  elem = elem.replace("[Es]", "")
                  elem = elem.replace("[Ee]", "")
                  while elem.startswith(" "): elem = elem[1:]
                  while elem.endswith(" "): elem = elem[:-1]
                  ret_sen_list.append(elem)
        return ret_sen_list

    def convert_to_features(self, batch, idx):
        input_ = batch["input"]
        output_ = batch["output"]

        source = self.tokenizer.batch_encode_plus(
            [input_],
            max_length=self.hparams.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if self.hparams.model_type == "hyper-mem-np-only": 
           target = self._get_id_np_only(output_)
        else:
           target = self._get_id(output_, self.hparams.max_output_length, is_input=False)

        if idx == 0 and torch.cuda.current_device() == 0:
            print(f"=" * 80)
            print(f"input: {input_}")
            print(f"output: {output_}")
            print(f"source: {source}")
            print(f"target: {target}")
            print(f"=" * 80)

        return source, target, input_, output_

    def _remove_sp(self, sen):
        sen = sen.replace('[Ee]', '')
        sen = sen.replace('[Es]', '')
        while sen.endswith(' '): sen = sen[:-1] 
        while sen.startswith(' '): sen = sen[1:] 
        return sen

    def _get_id_np_only(self, sen):

        target_idx = [self.tokId2clusterId[el] for el in self.corpus_tokenList[self._remove_sp(sen)]]
        target_idx = [self.es_tokid]+target_idx+[self.ee_tokid]
        if len(target_idx) > self.hparams.max_output_length:
            target_idx = target_idx[:self.hparams.max_output_length]
            att = [1]*self.hparams.max_output_length 
        else:
            _leftover = self.hparams.max_output_length - len(target_idx)
            att = [1]*len(target_idx) + [0]*_leftover 
            target_idx = target_idx+[0]*_leftover 
        assert len(target_idx)==len(att)==self.hparams.max_output_length 
        target_idx = torch.tensor([target_idx])
        att = torch.tensor([att])
        target = {"input_ids": target_idx, "attention_mask": att}
        return target

    def _get_id(self, sen_list, max_len, is_input):
        ### output_tokid => parse and add
        _target = np.array(self.tokenizer.batch_encode_plus(
            [sen_list],
            max_length=max_len,
            padding="max_length",
            truncation=True,
        )["input_ids"][0])
        _target = self._remove_zeros(_target)
        ret_sen_list = self.get_ret_sen(sen_list, is_input=is_input)
        ret_tokid_list = [] 
        for sen in ret_sen_list:
            ret_tokid_list.append(self.corpus_tokenList[sen])

        es_tokid_list = list(np.where(_target==self.es_tokid)[0])
        ee_tokid_list = list(np.where(_target==self.ee_tokid)[0])

        target = []
        target_loss_mask = []
        temp = copy.deepcopy(_target)
        idx = 0

        assert len(es_tokid_list) <= len(ret_tokid_list)
        assert len(es_tokid_list)>0 and len(ee_tokid_list)>0

        for i, (es, ee, idlist) in enumerate(zip(es_tokid_list, ee_tokid_list, ret_tokid_list)):
            es = es-idx
            ee = ee-idx
            # vanilla part
            if self.modelId2sharedId is not None:
               _ids = [self.modelId2sharedId[el] for el in temp[:es+1]]
            else:
               _ids = temp[:es+1]
            target.extend(_ids)
            target_loss_mask.extend([self.vd_mask]*len(_ids))
            # npd part
            if self.hparams.cluster_num != -1:
               assert idlist[-1] == 1
               _ids = [self.tokId2clusterId[el] for el in idlist[:-1]]
            else:
               _ids = idlist[:-1]
            target.extend(_ids)
            target_loss_mask.extend([self.npd_mask]*len(_ids))

            temp = temp[ee:]
            idx += ee
        if self.modelId2sharedId is not None:
           _ids = [self.modelId2sharedId[el] for el in temp]
        else:
           _ids = temp 
        target.extend(_ids)
        target_loss_mask.extend([self.vd_mask]*len(_ids))

        if len(target) > max_len:
            target = target[: max_len]
            att = [1] * max_len 
            target_loss_mask = target_loss_mask[:max_len]

        else:
            _leftover = max_len - len(target)
            att = [1] * len(target) + [0] * _leftover
            target = target + [0] * _leftover
            target_loss_mask = target_loss_mask + [self.zeros]*_leftover 
            
        target_idx = torch.tensor([target])
        att_idx = torch.tensor([att])
      
        if self.modelId2sharedId is not None:
           assert (
              len(target) == len(att) == len(target_loss_mask) == max_len
          ), print(f"length of target: {len(target)}\nlength of attention:  {len(att)}\nlength of target_loss_mask: {len(target_loss_mask)}")
           target_loss_mask = torch.tensor([target_loss_mask])
           target = {"input_ids": target_idx, "attention_mask": att_idx, "loss_mask": target_loss_mask}
        else:
           target = {"input_ids": target_idx, "attention_mask": att_idx}
        return target


    def convert_to_features_w_enc(self, batch, idx):
        input_ = batch["input"]
        output_ = batch["output"]

        source = self._get_id(input_, self.hparams.max_input_length, is_input=True)
        target = self._get_id(output_, self.hparams.max_output_length, is_input=False)

        if idx == 0 and torch.cuda.current_device() == 0:
            print(f"=" * 80)
            print(f"input: {input_}")
            print(f"output: {output_}")
            print(f"source: {source}")
            print(f"target: {target}")
            print(f"=" * 80)

        return source, target, input_, output_



    def _remove_zeros(self, ids):
        if 0 in ids:
            idx = list(ids).index(0)
            return ids[:idx]
        else:
            return ids

    def __getitem__(self, idx):
        if self.hparams.w_enc:
           source, target, input_, output_ = self.convert_to_features_w_enc(
                     self.dataset.iloc[idx], idx
                 ) 
        else:
           source, target, input_, output_ = self.convert_to_features(
                     self.dataset.iloc[idx], idx
                 )
        source_ids = source["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()
        if "loss_mask" in target:
           target_loss_mask = target["loss_mask"].squeeze()

           return {
               "source_ids": source_ids,
               "target_ids": target_ids,
               "source_mask": src_mask,
               "target_mask": target_mask,
               "target_loss_mask": target_loss_mask,
               "input": input_,
               "output": output_,
           }

        else:
           return {
               "source_ids": source_ids,
               "target_ids": target_ids,
               "source_mask": src_mask,
               "target_mask": target_mask,
               "input": input_,
               "output": output_,
           }

