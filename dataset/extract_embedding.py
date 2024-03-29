import os
import sys
import argparse 
import pickle
import pandas as pd
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModel, T5EncoderModel, T5Tokenizer
from tqdm import tqdm
import re
# from knockknock import slack_sender
# from slack import get_webhook_url, get_channel

def insert(d, k, v):
   if k not in d.keys():
      d[k] = set([v])
   else:
      d[k].add(v)

def lmap(f, x):
    return list(map(f, x))

def ids_to_text(generated_ids, tokenizer):
    gen_text = tokenizer.batch_decode([generated_ids], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return lmap(str.strip, gen_text)

def insert_tokId(nodeId, tokId):
    if tokId == -1:
        return
    if nodeId not in nodeId_tokIdList.keys():
        nodeId_tokIdList[nodeId] = [tokId]
    else:
        nodeId_tokIdList[nodeId].append(tokId)

def dump(fname, file):
   with open(os.path.join(args.save_path, fname), "wb") as f:
      pickle.dump(file, f)

def encode_sp(sen, model, tokenizer):
   _tok = tokenizer(sen, return_tensors='pt', add_special_tokens=False)
   _input_ids = _tok['input_ids'].cuda()
   _attention_mask = _tok["attention_mask"].cuda()
   _tok_decode = tokenizer.convert_ids_to_tokens(_input_ids[0])
   model_ret = model(input_ids=_input_ids, attention_mask=_attention_mask, return_dict=True)
   last_hidden_state = model_ret['last_hidden_state'][0]
   last_hidden_state = last_hidden_state.detach().cpu().numpy()
   _input_ids = _input_ids.detach().cpu().numpy()
   return _tok_decode, _input_ids, last_hidden_state

def construct_sp():

   tokId_emb = {} # {tokid: emb}
   tok_Idlist_dict = {} # {tok_text: [Idlist of the tok]}
   tok_Id_dict = {} # {Id: tok_text}

   # tokId = 0 -> <pad> token 
   _tok_decode, _input_ids, last_hidden_state = encode_sp("<pad>", model, tokenizer)
   assert len(_tok_decode) == 1
   tok_Idlist_dict[_tok_decode[0]] = [0]
   tok_Id_dict[0] = _tok_decode[0] 
   assert _input_ids[0][0] == 0
   tokId_emb[0] = last_hidden_state[0]

   # tokId = 1 -> </s> token
   _tok_decode, _input_ids, last_hidden_state = encode_sp("</s>", model, tokenizer)
   assert _tok_decode[0] == "</s>"
   assert len(_tok_decode) == 1
   tok_Idlist_dict[_tok_decode[0]] = [1]
   tok_Id_dict[1] = _tok_decode[0] 
   assert _input_ids[0][0] == 1
   tokId_emb[1] = last_hidden_state[0]
   return tok_Idlist_dict, tok_Id_dict, tokId_emb

def construct_corpus():
   corpusId_corpus_dict = {} # {corpusId: corpus} 
   corpusId_emb_dict = {} # {corpusId: {tok: {emb}}}
   tokId_corpus = {} # {tokid: corpusText}

   total_tok_num = 0; tokId = 2

   for corpusId in tqdm(range(corpus_num)):
      elem = corpus_file["corpus"][corpusId]
      _tok_decode, _input_ids, last_hidden_state = encode_sp(elem, model, tokenizer)
      
      _tok_dict = {}
      assert len(_input_ids[0])==len(last_hidden_state)==len(_tok_decode)
      total_tok_num += len(last_hidden_state)

      for tok_pos, (_text, _ids, _emb) in enumerate(zip(_tok_decode, _input_ids[0], last_hidden_state)):
         tok_Id_dict[tokId] = _text 
         if _text not in tok_Idlist_dict.keys():
            tok_Idlist_dict[_text] = [tokId]
         else:
            tok_Idlist_dict[_text].append(tokId)
         _tok_dict[tokId] = _emb
         tokId_corpus[tokId] = elem 
         tokId_emb[tokId] = _emb
         tokId += 1
         
         # Add EOS Token 
         if tok_pos == len(_tok_decode)-1:
            _tok_dict[1] = tokId_emb[1]

      corpusId_corpus_dict[corpusId] = elem
      corpusId_emb_dict[corpusId] = _tok_dict 

   print(f'total_tok_num: {total_tok_num}')
   print(f'tokId: {tokId}')
   return corpusId_corpus_dict, corpusId_emb_dict, tokId_corpus 

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

def construct_dataset(split,corpus2tokenList):
   df = load_data(split)
   save_dict = {'input': [], 'output': [], 'output_tokid': []}
   for _input, _output in zip(df['input'], df['output']):
      corpus_id = corpus.index(_output)
      emb_dict = corpusId_emb_dict[corpus_id]
      output_tok = list(emb_dict.keys())
      output_emb = list(emb_dict.values())
      save_dict['input'].append(_input)
      save_dict['output'].append(_output)
      save_dict['output_tokid'].append(output_tok)

   return save_dict, f"gr_contextualized_{split}.pickle"

def bi_construct_dataset(split,corpus2tokenList ,first_only=False):
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
   return save_dict, f"bi_contextualized_first_only_{first_only}.pickle" 


def construct_group():
   tokGroupId_tok_dict = {}
   tokId_tokGroupId = {}
   tokGroupId_tokIdList = {}
   tokGroupId = 2 ## assert tokGroupId 1 is </s> for generate()
   tokTextList = list(tok_Idlist_dict.keys())
   assert len(tokTextList) == len(set(tokTextList))
   for tokText, tokIdList in tok_Idlist_dict.items():
      if tokText == "</s>":
         print(f"Found </s> and set it to 1!!!")
         tokGroupId_tok_dict[1] = tokText 
         tokGroupId_tokIdList[1] = tokIdList  
         for tokId in tokIdList:
            assert tokId not in tokId_tokGroupId.keys()
            tokId_tokGroupId[tokId] = 1 
      elif tokText == "<pad>":
         print(f"Found <pad> and set it to 0!!!")
         tokGroupId_tok_dict[0] = tokText 
         tokGroupId_tokIdList[0] = tokIdList  
         for tokId in tokIdList:
            assert tokId not in tokId_tokGroupId.keys()
            tokId_tokGroupId[tokId] = 0 
      else:
         tokGroupId_tok_dict[tokGroupId] = tokText
         tokGroupId_tokIdList[tokGroupId] = tokIdList
         for tokId in tokIdList:
            assert tokId not in tokId_tokGroupId.keys()
            tokId_tokGroupId[tokId] = tokGroupId
         tokGroupId += 1
   return tokGroupId_tok_dict, tokId_tokGroupId, tokGroupId_tokIdList

def construct_node_prefix_tree():
   # find token number
   toknum = 0
   for cdict in corpusId_emb_dict.values():
      toknum += len(list(cdict.keys()))
      toknum += 2

   tree = [set() for i in range(toknum)]
   group_set = {}; token_set = {}
   inv_group_set = {}; inv_token_set = {}
   nodeid = 2

   for c_id, (corpusid, corpusdict) in enumerate(corpusId_emb_dict.items()):
      tokidlist = list(corpusdict.keys())
      tokgroupidlist = [tokId_tokGroupId[el] for el in tokidlist]
      tokgroupidlist = [0] + tokgroupidlist + [0]
      tokidlist = [0] + tokidlist + [0]

      cur_nid = 0
      for i in range(len(tokgroupidlist)-1):
         gid = tokgroupidlist[i]
         next_gid = tokgroupidlist[i+1]
         tokid = tokidlist[i]

         insert(group_set, cur_nid, gid)
         insert(token_set, cur_nid, tokid)
         insert(inv_group_set, gid, cur_nid)
         insert(inv_token_set, tokid, cur_nid)
         
         # find next nid
         next_nid = cur_nid
         for _next_nid in tree[cur_nid]:
               if next_gid in group_set[_next_nid]:
                  next_nid = _next_nid

         # failed to find next nid, create new nid
         if next_nid == cur_nid:
            tree[cur_nid].add(nodeid)
            group_set[nodeid] = set()
            next_nid = nodeid
            nodeid += 1
         cur_nid = next_nid
   return group_set, token_set, inv_group_set, inv_token_set, tree

def construct_group_prefix_tree():
   sys.setrecursionlimit(900000000)

   constrained_dict = {}
   for corpusId, corpusDict in corpusId_emb_dict.items():
      cur_dict = constrained_dict # cur_dict[-2]: the node number
      tokIdList = list(corpusDict.keys())
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
   parser.add_argument("--train_file", default=None, type=str)
   parser.add_argument("--dev_file", default=None, type=str)
   parser.add_argument("--test_file", default=None, type=str)
   parser.add_argument("--save_path", default=None, required=True, type=str)
   parser.add_argument("--emb_path", default=None, required=True, type=str)
   parser.add_argument("--t5", action='store_true')
   parser.add_argument("--bi", action='store_true')
   parser.add_argument("--first_only", action='store_true')
   args = parser.parse_args()

   # assert not os.path.exists(args.save_path), f'{args.save_path} already exists!! Check if it is correct!'

   if args.first_only and not args.bi: 
      assert False, f"First Only is only applied to bi-encoder for now!"

   if not ((args.train_file is None and args.dev_file is None and args.test_file is None) or (args.train_file is not None and args.dev_file is not None and args.test_file is not None)):
      assert False, f'Either ALL (train|dev|test) file is None or ALL file is NOT None!!'

   corpus_file = pd.read_csv(args.corpus)
   corpus = list(corpus_file['corpus'])
   corpus_num = len(corpus)

   if args.t5:
      print(f'## Loading T5EncoderModel')
      model = T5EncoderModel.from_pretrained(args.emb_path).cuda()
   else:
      model = AutoModel.from_pretrained(args.emb_path).cuda()
   tokenizer = AutoTokenizer.from_pretrained(args.emb_path)

   # add pad and </s>
   tok_Idlist_dict, tok_Id_dict, tokId_emb = construct_sp()
   # add the rest 
   corpusId_corpus_dict, corpusId_emb_dict, tokId_corpus = construct_corpus()

   # Grouping
   tokGroupId_tok_dict, tokId_tokGroupId, tokGroupId_tokIdList = construct_group()

   # construct corpus_tree
   group_tree = construct_group_prefix_tree()
   node_group_set, node_token_set, node_inv_group_set, node_inv_token_set, node_tree = construct_node_prefix_tree()
   node_sup_set = {'group_set': node_group_set, "token_set": node_token_set, "inv_group_set": node_inv_group_set, "inv_token_set": node_inv_token_set}

   if args.train_file is None:
      pass 
   else:
      corpusId_tokenList_dict = pickle.load(open(os.path.join(args.save_path, "corpusId_tokenList_dict.pickle"), "rb"))

      corpus2tokenList = {}
      for corpusId, tokenList in corpusId_tokenList_dict.items():
         context = corpus[corpusId]
         m = re.match("<title>(.*?)<context>",context)
         title = m.group(1).strip()
         corpus2tokenList[title] = tokenList
      if args.bi:
         train_dict, train_fname = bi_construct_dataset("train",corpus2tokenList ,first_only=args.first_only)
         dev_dict, dev_fname = bi_construct_dataset("dev",corpus2tokenList, first_only=args.first_only)
         test_dict, test_fname = bi_construct_dataset("test", corpus2tokenList,first_only=args.first_only)
      else:
         train_dict, train_fname = construct_dataset('train',corpus2tokenList)
         dev_dict, dev_fname = construct_dataset('dev',corpus2tokenList)
         test_dict, test_fname = construct_dataset('test',corpus2tokenList)

   """
   각 token은 하나씩 존재하고, GroupId를 가지고 있다. 
   해당 GroupId 안에는 여러 tokId가 존재하고 이는 각각 contextualized tokId로 연결
   -> GroupId: 이전에서의 tokenId와 비슷한 역할을 한다고 생각하면 된다.
   둘 다 0부터 시작하고 tokenId의 0은 <pad> 이고 groupId의 0은 <pad>, 1은 </s>
   + corpus tree는 GroupId로 만들어진다! 
   """
   os.makedirs(args.save_path, exist_ok=True)
   dump("tokId_emb.pickle", tokId_emb)
   dump("tokGroupId_tokIdList.pickle", tokGroupId_tokIdList)
   dump("tokId_tokGroupId.pickle", tokId_tokGroupId)
   dump("tokId_tokText.pickle", tok_Id_dict)
   dump("tokId_corpus.pickle", tokId_corpus)
   dump("groupId_tree.pickle", group_tree)
   dump("nodeId_tree.pickle", node_tree)
   dump("nodeId_sup_set.pickle", node_sup_set)
   if args.train_file is not None:
      dump(train_fname, train_dict)
      dump(dev_fname, dev_dict)
      dump(test_fname, test_dict)

   """ 
   dump("tokGroupId_tok.pickle", tokGroupId_tok_dict)
   dump("tokText_TokIdList.pickle", tok_Idlist_dict)
   dump("corpusId_corpus.pickle", corpusId_corpus_dict)
   dump("corpusId_emb.pickle", corpusId_emb_dict)
   """

   print("DONE!!")
