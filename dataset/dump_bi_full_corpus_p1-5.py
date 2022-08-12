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
                max_length=2000,
                #padding="max_length",
                truncation=True
            )
    _input_ids = _tok['input_ids'].to(model.device)
    _attention_mask = _tok["attention_mask"].to(model.device)
    encoder = model.get_encoder().eval()
    model_ret = encoder(input_ids=_input_ids, attention_mask=_attention_mask, return_dict=True)
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
    if args.idx == 0:
        tokText2tokIdList, tokId2tokText = t5_construct_sp(_model, _tokenizer, emb_f)
        return None, tokText2tokIdList, tokId2tokText, None
    else:
        tokId_emb= {}; tokId2tokText = {}; tokText2tokIdList = defaultdict(list) 
        cur_tokId = 0; corpusId = 0
        tokId2corpus = {}
        corpusId_tokenList_dict = {} # for grouptree
        for i in tqdm(range(0, len(corpus), args.dump_batch)):
            iter_corpus = _corpus[i:i+args.dump_batch]
            iter_context = _context[i:i+args.dump_batch]
            tok_decode_list, _, last_hidden_state_list = encode_list(iter_corpus, iter_context, _model, _tokenizer)

            for elem, tok_decode, last_hidden_state in zip(iter_corpus, tok_decode_list, last_hidden_state_list):
                assert len(tok_decode) == len(last_hidden_state)
                _tok_list = []
                for _tok, _last_hidden_state in zip(tok_decode, last_hidden_state):
                    if _tok == "<pad>": break 
                    tokId2tokText[cur_tokId] = _tok 
                    tokText2tokIdList[_tok].append(cur_tokId)
                    tokId_emb[cur_tokId] = _last_hidden_state 
                    _tok_list.append(cur_tokId)
                    tokId2corpus[cur_tokId] = elem
                    emb_f[cur_tokId][:] = _last_hidden_state
                    cur_tokId += 1

                _tok_list.append(1)
                corpusId_tokenList_dict[corpusId] = _tok_list
                corpusId += 1
        emb_f.flush()
        return tokId2corpus, tokText2tokIdList, tokId2tokText, corpusId_tokenList_dict 

def dump(fname, file):
    with open(os.path.join(args.save_path, fname), "wb") as f:
        pickle.dump(file, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--corpus", default=None, required=True, type=str)
    # parser.add_argument("--train_file", default=None, required=True, type=str)
    # parser.add_argument("--dev_file", default=None, required=True, type=str)
    # parser.add_argument("--test_file", default=None, required=True, type=str)
    parser.add_argument("--save_path", default=None, required=True, type=str)
    parser.add_argument("--emb_path", default=None, required=True, type=str)
    parser.add_argument("--dump_batch", default=10, type=int)
    parser.add_argument("--idx", default=-1, type=int)
    parser.add_argument("--t5", action='store_true')
    parser.add_argument("--bart", action='store_true')
    args = parser.parse_args()


    if args.idx == 0:
        toknum = 2
    elif args.idx == 1:
        toknum = 2780800 # 2780710
    elif args.idx == 2:
        toknum = 2899200 # 2899139
    elif args.idx == 3:
        toknum = 3054900 # 3054867
    elif args.idx == 4:
        toknum = 3121700 # 3121684
    elif args.idx == 5:
        toknum = 3160858 # 3160900
    elif args.idx == 6:
        toknum = 3203500 # 3203474
    elif args.idx == 7:
        toknum = 3303300 # 3303217 
    elif args.idx == 8:
        toknum = 3389700 # 3389672
    elif args.idx == 9:
        toknum = 3349100 # 3349042
    elif args.idx == 10:
        toknum = 3346600 # 3346522
    elif args.idx == 11:
        toknum = 2753200 # 2753104
    elif args.idx == 12:
        toknum = 2547600 # 2547571
    else:
        assert False

    print(f"### Starting idx: {args.idx}")
    print(f"### toknum: {toknum}")

    corpus_file = pd.read_csv(os.path.join(args.corpus, f"idx_{args.idx}.csv"))
    corpus_file = corpus_file.fillna("")
    corpus = list(corpus_file['corpus'])
    context = list(corpus_file['context'])
    assert len(corpus) == len(context)
    print(f"### Loading Full Corpus")
    corpus_num = len(corpus)
    print(f"corpus_num: {corpus_num}")

    args.save_path = os.path.join(args.save_path, f"idx_{args.idx}")
    print(f"Saving in {args.save_path}")
    os.makedirs(args.save_path, exist_ok=True)
    emb_f = os.path.join(args.save_path, f"tokId_emb_{args.idx}.dat")
    if os.path.exists(emb_f): os.system(f"rm {emb_f}")
    emb_f = np.memmap(emb_f, dtype="float32", mode="w+", shape=(toknum, 1024))
    emb_f.flush()

    if args.t5:
        print(f'## Loading T5EncoderModel')
        model = T5EncoderModel.from_pretrained(args.emb_path).cuda()
        tokenizer = T5Tokenizer.from_pretrained(args.emb_path)
        tokId2corpus, tokText2tokIdList, tokId2tokText, corpusId_tokenList_dict = t5_construct_corpus(model, tokenizer, corpus, context, emb_f)
    elif args.bart:
        assert False
        print(f'## Loading BartModel')
        model = BartModel.from_pretrained(args.emb_path).get_encoder().cuda()
        tokenizer = BartTokenizer.from_pretrained(args.emb_path)
        tok_Idlist_dict, tok_Id_dict, tokId_emb = bart_construct_sp()
        assert len(tokId_emb.keys()) == 4
        corpusId_corpus_dict, corpusId_emb_dict, tokId_corpus, corpusId_tokenList_dict = bart_construct_corpus()
    else:
        assert False

    #group_tree = construct_group_prefix_tree() 

    # train_dict, train_fname = bi_construct_dataset("train", first_only=args.first_only)
    # dev_dict, dev_fname = bi_construct_dataset("dev", first_only=args.first_only)
    # test_dict, test_fname = bi_construct_dataset("test", first_only=args.first_only)

    if args.idx == 0:
        dump("tokText2tokIdList.pickle", tokText2tokIdList)
        dump("tokId2tokText.pickle", tokId2tokText)
    else:
        dump("tokId2corpus.pickle", tokId2corpus)
        dump("tokText2tokIdList.pickle", tokText2tokIdList)
        dump("tokId2tokText.pickle", tokId2tokText)
        dump("corpusId_tokenList_dict.pickle", corpusId_tokenList_dict)
    # dump(train_fname, train_dict)
    # dump(dev_fname, dev_dict)
    # dump(test_fname, test_dict)   

    print(f"==== DONE idx: {args.idx} ====")