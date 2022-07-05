import os
import sys
import argparse 
import pickle
import pandas as pd
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModel, T5EncoderModel, T5Tokenizer
from tqdm import tqdm

from knockknock import slack_sender
from slack import get_webhook_url, get_channel

def load_data(split):
   if split == "train":
      df = pd.read_csv(args.train_file)
   elif split == "dev":
      df = pd.read_csv(args.dev_file)
   elif split == "test":
      df = pd.read_csv(args.test_file)
   else:
      raise NotImplementedError('Check the split!')
   return df

def check_corpusId_emb(idx, df, save_dict):
   print(f'=== Checking.. {idx}')
   with open(os.path.join(args.save_path, f"{idx}_results.pickle"), 'rb') as f:
      corpusId_emb = pickle.load(f)['corpusId_emb_dict']
      found_idx = []
      for i, (_input, _output) in enumerate(zip(df['input'], df['output'])):
         corpus_id = corpus.index(_output)
         if corpus_id in corpusId_emb.keys():
               found_idx.append(i)
               emb_dict = corpusId_emb[corpus_id]
               output_tok = list(emb_dict.keys())
               output_emb = list(emb_dict.values())
               save_dict['input'].append(_input)
               save_dict['output'].append(_output)
               save_dict['output_tokid'].append(output_tok) 
   
      # remove found_idx
      df = df.drop(found_idx)
   return df, save_dict

def dump(fname, file):
   if not os.path.exists(args.save_path):
      os.makedirs(args.save_path)
   with open(os.path.join(args.save_path, fname), "wb") as f:
      pickle.dump(file, f)

def construct_dataset(split):
    df = load_data(split)
    df_inputNum = len(df['input'])
    save_dict = {'input': [], 'output': [], 'output_tokid': []}

    for i in range(args.filenum):
        df, save_dict = check_corpusId_emb(i, df, save_dict)

    assert len(save_dict['input']) == df_inputNum 
    return save_dict, f"gr_contextualized_{split}.pickle"

def bi_construct_dataset(split, first_only=False):
   df = load_data(split)
   save_dict = {'input': [], 'output': [], 'output_tokid': []}
   for _input, _output in zip(df["input"], df["output"]):
      corpus_id = corpus.index(_output)
      emb_dict = corpusId_emb_dict[corpus_id]
      output_tok = list(emb_dict.keys())
      output_emb = list(emb_dict.values())

      assert output_tok[-1] == 1
      if first_only:
         save_dict['input'].append(_input)
         save_dict['output'].append(_output)
         save_dict['output_tokid'].append([output_tok[0]])
      else:
         for _tok in output_tok[:-1]:
            save_dict['input'].append(_input)
            save_dict['output'].append(_output)
            save_dict['output_tokid'].append([_tok])
   return save_dict, f"bi_contextualized_first_only_{first_only}_{split}.pickle" 


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--corpus", default=None, required=True, type=str)
    parser.add_argument("--train_file", default=None, required=True, type=str)
    parser.add_argument("--dev_file", default=None, required=True, type=str)
    parser.add_argument("--test_file", default=None, required=True, type=str)
    parser.add_argument("--save_path", default=None, required=True, type=str)
    parser.add_argument("--emb_path", default=None, required=True, type=str)
    parser.add_argument("--filenum", default=None, required=True, type=int)
    parser.add_argument("--t5", action='store_true')
    args = parser.parse_args()

    corpus_file = pd.read_csv(args.corpus)
    corpus_file = corpus_file.fillna("")
    corpus = list(corpus_file['corpus'])
    corpus_num = len(corpus)

    if args.t5:
        print(f'## Loading T5EncoderModel')
        model = T5EncoderModel.from_pretrained(args.emb_path).cuda()
    else:
        model = AutoModel.from_pretrained(args.emb_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.emb_path)

    ### construct dataset
    train_dict, train_fname = construct_dataset('train')
    dump(train_fname, train_dict)
    dev_dict, dev_fname = construct_dataset('dev')
    dump(dev_fname, dev_dict)
    test_dict, test_fname = construct_dataset('test')
    dump(test_fname, test_dict)

    print("DONE!!")

