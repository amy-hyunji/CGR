import os
import sys
import torch
import pickle
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

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
        target_ids = target["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()
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
