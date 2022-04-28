import os
import torch
import pickle
import pandas as pd

from torch.utils.data import Dataset


class GENREDataset(Dataset):
    def __init__(self, tokenizer, split, hparams, tokid2emb):
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

    def __len__(self):
        return self.len

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
        target = batch["output_tokid"]  # load from file
        if len(target) > self.hparams.max_output_length:
            target = target[: self.hparams.max_output_length]
            att = [1] * len(self.hparams.max_output_length)
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
