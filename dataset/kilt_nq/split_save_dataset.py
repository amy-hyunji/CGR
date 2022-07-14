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
   #print(f'=== Checking.. {idx}')
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
      if i == 0: continue
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

def construct_group():
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


def construct_group_prefix_tree():
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
      
   #{'tok_Idlist_dict': tok_Idlist_dict, 'tok_Id_dict': tok_Id_dict,  'tokId_corpus': tokId_corpus, 'corpusId_fileId_dict': corpusId_fileId_dict, 'corpusId_emb_dict': corpusId_emb_dict, 'corpusId_corpus_dict': corpusId_corpus_dict, 'corpusId_tokenList_dict': corpusId_tokenList_dict})

   # tok_Idlist_dict -> construct_group -> tokGroupId_tokIdList, tokId_tokGroupId
   # tok_Id_dict
   print(f'=== Construct Group')
   tok_Idlist_dict = {}; tok_Id_dict = {}; tokId_emb = {}
   for idx in tqdm(range(args.filenum)):
      with open(os.path.join(args.save_path, f"{idx}_results.pickle"), 'rb') as f:
         f_pickle = pickle.load(f)
         _tok_Idlist_dict = f_pickle['tok_Idlist_dict']
         _tok_Id_dict = f_pickle['tok_Id_dict']

         if 'tokId_emb' in f_pickle.keys():
            _tokId_emb = f_pickle['tokId_emb']
            for t_id, t_emb in _tokId_emb.items():
               tokId_emb[t_id] = t_emb
         else:
            _corpusId_emb_dict = f_pickle['corpusId_emb_dict']
            for _, emb_dict in _corpusId_emb_dict.items():
               for t_id, t_emb in emb_dict.items():
                  tokId_emb[t_id] = t_emb 

         for _tok, _Idlist in _tok_Idlist_dict.items():
            if _tok in tok_Idlist_dict.keys():
               tok_Idlist_dict[_tok] = list(set(tok_Idlist_dict[_tok]+_Idlist))
            else:
               tok_Idlist_dict[_tok] = _Idlist
         for _tok, _Id in _tok_Id_dict.items():
            tok_Id_dict[_tok] = _Id
   tokId_tokGroupId, tokGroupId_tokIdList = construct_group()
   dump("tokGroupId_tokIdList.pickle", tokGroupId_tokIdList)
   dump("tokId_tokGroupId.pickle", tokId_tokGroupId)
   dump("tokId_tokText.pickle", tok_Id_dict)
   dump("tokId_emb.pickle", tokId_emb)

   del tok_Idlist_dict
   del tokGroupId_tokIdList
   del tok_Id_dict 

   # corpusId_tokenList_dict -> construct_group_prefix_tree -> group_tree 
   # construct tokId_corpus, corpusId_fileId_dict, corpusId_emb_dict 
   print(f'=== Construct Trie')
   corpusId_tokenList_dict = {}; tokId_corpus = {}; corpusId_fileId_dict = {}; corpusId_emb_dict = {}
   for idx in tqdm(range(args.filenum)):
      if idx == 0: continue
      with open(os.path.join(args.save_path, f"{idx}_results.pickle"), 'rb') as f:
         f_pickle = pickle.load(f)
         _corpusId_tokenList_dict = f_pickle['corpusId_tokenList_dict']
         _tokId_corpus = f_pickle['tokId_corpus']
         _corpusId_fileId_dict = f_pickle['corpusId_fileId_dict']
         _corpusId_emb_dict = f_pickle['corpusId_emb_dict']
         for _corpusId, _tokenList in _corpusId_tokenList_dict.items():
            corpusId_tokenList_dict[_corpusId] = _tokenList
         for _tokId, _corpus in _tokId_corpus.items():
            tokId_corpus[_tokId] = _corpus 
         for _cId, _fId in _corpusId_fileId_dict.items():
            corpusId_fileId_dict[_cId] = _fId
         for _cId, _emb in _corpusId_emb_dict.items():
            corpusId_emb_dict[_cId] = _emb
   group_tree = construct_group_prefix_tree()
   dump("groupId_tree.pickle", group_tree)
   dump("tokId_corpus.pickle", tokId_corpus)
   dump("corpusId_fileId.pickle", corpusId_fileId_dict)
   dump("corpusId_emb.pickle", corpusId_emb_dict)

   del group_tree 
   del corpusId_tokenList_dict 
   del corpusId_emb_dict
   del corpusId_fileId_dict 
   del tokId_corpus 

   ### construct dataset
   print(f'=== Construct Train Dataset')
   train_dict, train_fname = construct_dataset('train')
   dump(train_fname, train_dict)
   print(f'=== Construct Dev Dataset')
   dev_dict, dev_fname = construct_dataset('dev')
   dump(dev_fname, dev_dict)
   print(f'=== Construct Test Dataset')
   test_dict, test_fname = construct_dataset('test')
   dump(test_fname, test_dict)

   print("DONE!!")

