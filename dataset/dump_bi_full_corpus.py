import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np

from argparse import ArgumentParser
from transformers import BartTokenizer, BartModel, T5EncoderModel, T5Tokenizer
from tqdm import tqdm 
from knockknock import slack_sender
from slack import get_webhook_url, get_channel
from collections import defaultdict

def encode_list(title_list, context_list, _model, _tokenizer):

    if context_list is not None:       
        assert False, f"Context list: {context_list}" 
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
                padding="longest",
            )
    _input_ids = _tok['input_ids'].to(model.device)
    _attention_mask = _tok["attention_mask"].to(model.device)
    #encoder = model.get_encoder().eval()
    model_ret = model(input_ids=_input_ids, attention_mask=_attention_mask, return_dict=True)
    assert len(title_tok) == len(model_ret['last_hidden_state'])
    last_hidden_state = [state[:toklen].detach().cpu().numpy() for (state, toklen) in zip(model_ret['last_hidden_state'], title_tok)]
    _tok_decode = [tokenizer.convert_ids_to_tokens(_ids)[:toklen] for (_ids, toklen) in zip(_input_ids, title_tok)]
    _input_ids = _input_ids.detach().cpu().numpy()
    return _tok_decode, _input_ids, last_hidden_state   


def t5_construct_sp(_model, _tokenizer, emb_f):
    tokId_emb = {}; tokId2tokText = {}; tokText2tokIdList = defaultdict(list) 

    if args.dump_batch == 1:
        _tok_decode, _input_ids, last_hidden_state = encode_list(["<pad>"], None, _model, _tokenizer)
        tokText2tokIdList[_tok_decode[0][0]].append(0)
        tokId2tokText[0] = _tok_decode[0][0] 
        emb_f[0][:] = last_hidden_state[0][0]
        assert _input_ids[0][0] == 0

        _tok_decode, _input_ids, last_hidden_state = encode_list(["</s>"], None, _model, _tokenizer)
        tokText2tokIdList[_tok_decode[0][0]].append(1)
        tokId2tokText[1] = _tok_decode[0][0]
        emb_f[1][:] = last_hidden_state[0][0]
        assert _input_ids[0][0] == 1

        emb_f.flush()
    else:
        _tok_decode, _input_ids, last_hidden_state = encode_list(["<pad>", "</s>"], None, _model, _tokenizer)
        assert len(_tok_decode) == 2
        
        tokText2tokIdList[_tok_decode[0][0]].append(0)
        tokId2tokText[0] = _tok_decode[0][0] 
        assert _input_ids[0][0] == 0

        tokText2tokIdList[_tok_decode[1][0]].append(1)
        tokId2tokText[1] = _tok_decode[1][0]
        assert _input_ids[1][0] == 1

        emb_f[0][:] = last_hidden_state[0][0]
        emb_f[1][:] = last_hidden_state[1][0]
        emb_f.flush()

    return tokText2tokIdList, tokId2tokText


def t5_construct_corpus(_model, _tokenizer, _corpus, _context, emb_f):
    tokText2tokIdList, tokId2tokText = t5_construct_sp(_model, _tokenizer, emb_f)
    cur_tokId = 2; corpusId = 0
    tokId2corpus = {}
    corpusId_tokenList_dict = {} # for grouptree
    for i in tqdm(range(0, len(corpus), args.dump_batch)):
        iter_corpus = _corpus[i:i+args.dump_batch]
        tok_decode_list, _, last_hidden_state_list = encode_list(iter_corpus, None, _model, _tokenizer)
        for elem, tok_decode, last_hidden_state in zip(iter_corpus, tok_decode_list, last_hidden_state_list):
            
            assert len(tok_decode) == len(last_hidden_state)
            _tok_list = []
            for _tok, _last_hidden_state in zip(tok_decode, last_hidden_state):
                if _tok == "<pad>": 
                    print("is pad!")
                    break 
                tokId2tokText[cur_tokId] = _tok 
                tokText2tokIdList[_tok].append(cur_tokId)
                _tok_list.append(cur_tokId)
                tokId2corpus[cur_tokId] = elem
                emb_f[cur_tokId][:] = _last_hidden_state
                cur_tokId += 1

            _tok_list.append(1)
            corpusId_tokenList_dict[corpusId] = _tok_list
            corpusId += 1
    emb_f.flush()
    return tokId2corpus, tokText2tokIdList, tokId2tokText, corpusId_tokenList_dict 

def bart_construct_sp(_model, _tokenizer, emb_f):
    tokId_emb = {}; tokId2tokText = {}; tokText2tokIdList = defaultdict(list) 

    if args.dump_batch < 4:
        _tok_decode, _input_ids, last_hidden_state = encode_list(["<s>"], None, _model, _tokenizer)
        tokText2tokIdList[_tok_decode[0][0]].append(0)
        tokId2tokText[0] = _tok_decode[0][0] 
        emb_f[0][:] = last_hidden_state[0][0]
        assert _input_ids[0][0] == 0

        _tok_decode, _input_ids, last_hidden_state = encode_list(["<pad>"], None, _model, _tokenizer)
        tokText2tokIdList[_tok_decode[0][0]].append(1)
        tokId2tokText[1] = _tok_decode[0][0]
        emb_f[1][:] = last_hidden_state[0][0]
        assert _input_ids[0][0] == 1

        _tok_decode, _input_ids, last_hidden_state = encode_list(["</s>"], None, _model, _tokenizer)
        tokText2tokIdList[_tok_decode[0][0]].append(2)
        tokId2tokText[2] = _tok_decode[0][0]
        emb_f[2][:] = last_hidden_state[0][0]
        assert _input_ids[0][0] == 2

        _tok_decode, _input_ids, last_hidden_state = encode_list(["<unk>"], None, _model, _tokenizer)
        tokText2tokIdList[_tok_decode[0][0]].append(3)
        tokId2tokText[3] = _tok_decode[0][0]
        emb_f[3][:] = last_hidden_state[0][0]
        assert _input_ids[0][0] == 3 
        
        emb_f.flush()
    else:
        _tok_decode, _input_ids, last_hidden_state = encode_list(["<s>", "<pad>", "</s>", "<unk>"], None, _model, _tokenizer)
        assert len(_tok_decode) == 4
        
        tokText2tokIdList[_tok_decode[0][0]].append(0)
        tokId2tokText[0] = _tok_decode[0][0] 
        assert _input_ids[0][0] == 0

        tokText2tokIdList[_tok_decode[1][0]].append(1)
        tokId2tokText[1] = _tok_decode[1][0]
        assert _input_ids[1][0] == 1

        tokText2tokIdList[_tok_decode[0][0]].append(2)
        tokId2tokText[2] = _tok_decode[2][0] 
        assert _input_ids[2][0] == 2

        tokText2tokIdList[_tok_decode[1][0]].append(3)
        tokId2tokText[3] = _tok_decode[3][0]
        assert _input_ids[3][0] == 3

        emb_f[0][:] = last_hidden_state[0][0]
        emb_f[1][:] = last_hidden_state[1][0]
        emb_f[2][:] = last_hidden_state[2][0]
        emb_f[3][:] = last_hidden_state[3][0]

        emb_f.flush()

    return tokText2tokIdList, tokId2tokText

def bart_construct_corpus(_model, _tokenizer, _corpus, _context, emb_f):
    print("Construct Special Tokens!")
    tokText2tokIdList, tokId2tokText = bart_construct_sp(_model, _tokenizer, emb_f)
    cur_tokId = 4; corpusId = 0
    tokId2corpus = {}
    corpusId_tokenList_dict = {} # for grouptree
    for i in tqdm(range(0, len(corpus), args.dump_batch)):
        iter_corpus = _corpus[i:i+args.dump_batch]
        tok_decode_list, _, last_hidden_state_list = encode_list(iter_corpus, None, _model, _tokenizer)

        for elem, tok_decode, last_hidden_state in zip(iter_corpus, tok_decode_list, last_hidden_state_list):
            assert len(tok_decode) == len(last_hidden_state)
            _tok_list = []
            for _tok, _last_hidden_state in zip(tok_decode, last_hidden_state):
                if _tok == "<pad>": break 
                tokId2tokText[cur_tokId] = _tok 
                tokText2tokIdList[_tok].append(cur_tokId)
                _tok_list.append(cur_tokId)
                tokId2corpus[cur_tokId] = elem
                emb_f[cur_tokId][:] = _last_hidden_state
                cur_tokId += 1

            _tok_list.append(2)
            corpusId_tokenList_dict[corpusId] = _tok_list
            corpusId += 1
    emb_f.flush()
    return tokId2corpus, tokText2tokIdList, tokId2tokText, corpusId_tokenList_dict 

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

def bi_construct_dataset(split, corpus, emb_f):
    df = load_data(split)
    save_dict = {'input': [], 'output': [], 'output_tokid': []}
    for _input, _output in zip(df["input"], df["output"]):
        corpus_id = corpus.index(_output)
        output_tok = corpusId_tokenList_dict[corpus_id]
        output_emb = [emb_f[tok][:] for tok in output_tok]

        if args.t5: assert output_tok[-1] == 1
        if args.bart: assert output_tok[-1] == 2
        for _tok in output_tok[:-1]:
            save_dict['input'].append(_input)
            save_dict['output'].append(_output)
            save_dict['output_tokid'].append([_tok])
    return save_dict, f"bi_contextualized_{split}.pickle" 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--corpus", default=None, required=True, type=str)
    parser.add_argument("--train_file", default=None, required=True, type=str)
    parser.add_argument("--dev_file", default=None, required=True, type=str)
    parser.add_argument("--test_file", default=None, required=True, type=str)
    parser.add_argument("--save_path", default=None, required=True, type=str)
    parser.add_argument("--emb_path", default=None, required=True, type=str)
    parser.add_argument("--dump_batch", default=10, type=int)
    parser.add_argument("--t5", action='store_true')
    parser.add_argument("--bart", action='store_true')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
      
    corpus_file = pd.read_csv(args.corpus)
    corpus_file = corpus_file.fillna("")
    corpus = list(corpus_file['corpus'])
    print(f"### Loading Full Corpus")
    corpus_num = len(corpus)
    print(f"corpus_num: {corpus_num}")

    emb_f = os.path.join(args.save_path, f"tokId_emb.dat")
    if os.path.exists(emb_f): os.system(f"rm {emb_f}")
    emb_f = np.memmap(emb_f, dtype="float32", mode="w+", shape=(36909000, 1024))
    emb_f.flush()

    if args.t5:
        print(f'## Loading T5EncoderModel')
        model = T5EncoderModel.from_pretrained(args.emb_path).cuda()
        tokenizer = T5Tokenizer.from_pretrained(args.emb_path)
        tokId2corpus, tokText2tokIdList, tokId2tokText, corpusId_tokenList_dict = t5_construct_corpus(model, tokenizer, corpus, None, emb_f)
    elif args.bart:
        print(f'## Loading BartModel')
        model = BartModel.from_pretrained(args.emb_path).get_encoder().cuda()
        tokenizer = BartTokenizer.from_pretrained(args.emb_path)
        tokId2corpus, tokText2tokIdList, tokId2tokText, corpusId_tokenList_dict = bart_construct_corpus(model, tokenizer, corpus, None, emb_f)
    else:
        assert False

    #group_tree = construct_group_prefix_tree() 
    emb_f = np.memmap(os.path.join(args.save_path, "tokId_emb.dat"), dtype="float32", mode="r+", shape=(36909000, 1024))
    train_dict, train_fname = bi_construct_dataset("train", corpus, emb_f)
    dev_dict, dev_fname = bi_construct_dataset("dev", corpus, emb_f)
    test_dict, test_fname = bi_construct_dataset("test", corpus, emb_f)

    dump("tokId2corpus.pickle", tokId2corpus)
    dump("tokText2tokIdList.pickle", tokText2tokIdList)
    dump("tokId2tokText.pickle", tokId2tokText)
    dump("corpusId_tokenList_dict.pickle", corpusId_tokenList_dict)
    dump(train_fname, train_dict)
    dump(dev_fname, dev_dict)
    dump(test_fname, test_dict)   

    print("==== DONE ====")
