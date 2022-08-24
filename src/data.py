import os
import sys
import torch
import pickle
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
                if _input in self.dataset['input']: 
                    continue 
                else:
                    self.dataset['input'].append(_input)
                    self.dataset['output'].append(_output)
                    self.dataset['output_tokid'].append(_output_tok_id)
            self.dataset = pd.DataFrame(self.dataset)
        else:
            self.dataset = pd.DataFrame(data_dict)

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
