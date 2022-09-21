import os
import sys
import json
import faiss
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
from sklearn.cluster import KMeans

def encode_list(title_list, context_list, _model, _tokenizer):

    if context_list is not None:        
        context_list = [" ".join([_title, _sen]).strip() for (_title, _sen) in zip(title_list, context_list)]
        title_tok = [len(_tokenizer(_title, return_tensors='pt', add_special_tokens=False).input_ids[0]) for _title in title_list]
        #print("title_tok: ", title_tok)
    else:
        context_list = title_list
        title_tok = [len(_tokenizer(_title, return_tensors='pt', add_special_tokens=False).input_ids[0]) for _title in title_list]

    _tok = _tokenizer(
                context_list, 
                return_tensors='pt', 
                add_special_tokens=False, 
                max_length=args.max_length,
                #padding="max_length",
                truncation=True
            )
    _input_ids = _tok['input_ids'].to(_model.device)
    _attention_mask = _tok["attention_mask"].to(_model.device)
    encoder = _model.get_encoder().eval()
    model_ret = encoder(input_ids=_input_ids, attention_mask=_attention_mask, return_dict=True)
    assert len(title_tok) == len(model_ret['last_hidden_state'])
    last_hidden_state = [state[:toklen].detach().cpu().numpy() for (state, toklen) in zip(model_ret['last_hidden_state'], title_tok)]
    _tok_decode = [_tokenizer.convert_ids_to_tokens(_ids)[:toklen] for (_ids, toklen) in zip(_input_ids, title_tok)]
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

def t5_construct_corpus(_model, _tokenizer, _corpus, _context, emb_f, tokId2tokText, tokText2tokIdList, tokId2corpus, corpusId_tokenList_dict, cur_tokId, corpusId):
    if args.idx == 0:
        tokText2tokIdList, tokId2tokText = t5_construct_sp(_model, _tokenizer, emb_f)
        return None, tokText2tokIdList, tokId2tokText, None
    else:
        # tokId2tokText = {}; tokText2tokIdList = defaultdict(list) 
        # cur_tokId = 0; corpusId = 0
        # tokId2corpus = {}
        # corpusId_tokenList_dict = {} # for grouptree
        for i in tqdm(range(0, len(_corpus), args.dump_batch)):
            if i % 500000 == 0 and i != 0:                    
                temp_dump_emb(tokId2corpus, tokText2tokIdList, tokId2tokText, corpusId_tokenList_dict, cur_tokId, corpusId)
            iter_corpus = _corpus[i:i+args.dump_batch]
            iter_context = _context[i:i+args.dump_batch]
            if iter_corpus[0] == "" and iter_context[0] == "":
                print('pass')
                continue
            tok_decode_list, _, last_hidden_state_list = encode_list(iter_corpus, iter_context, _model, _tokenizer)

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

                _tok_list.append(1)
                corpusId_tokenList_dict[corpusId] = _tok_list
                corpusId += 1
        emb_f.flush()
        return tokId2corpus, tokText2tokIdList, tokId2tokText, corpusId_tokenList_dict 

def temp_dump_emb(tokId2corpus, tokText2tokIdList, tokId2tokText, corpusId_tokenList_dict, cur_tokId,  corpusId):
    if args.idx == 0:
        dump("tokText2tokIdList.pickle", tokText2tokIdList)
        dump("tokId2tokText.pickle", tokId2tokText)
    else:
        dump("tokId2corpus.pickle", tokId2corpus)
        dump("tokText2tokIdList.pickle", tokText2tokIdList)
        dump("tokId2tokText.pickle", tokId2tokText)
        dump("corpusId_tokenList_dict.pickle", corpusId_tokenList_dict)
        dump("ids.pickle", {"cur_tokId": cur_tokId, "corpusId": corpusId})

def dump(fname, file):
    with open(os.path.join(args.save_path, fname), "wb") as f:
        pickle.dump(file, f)

def get_toknum(idx):
    if idx == 0:
        toknum = 2
    elif idx == 1:
        toknum = 2780800 # 2780710
    elif idx == 2:
        toknum = 2899200 # 2899139
    elif idx == 3:
        toknum = 3054900 # 3054867
    elif idx == 4:
        toknum = 3121700 # 3121684
    elif idx == 5:
        toknum = 3160858 # 3160900
    elif idx == 6:
        toknum = 3203500 # 3203474
    elif idx == 7:
        toknum = 3303300 # 3303217 
    elif idx == 8:
        toknum = 3389700 # 3389672
    elif idx == 9:
        toknum = 3349100 # 3349042
    elif idx == 10:
        toknum = 3346600 # 3346522
    elif idx == 11:
        toknum = 2753200 # 2753104
    elif idx == 12:
        toknum = 2547600 # 2547571
    else:
        assert False
    return toknum

@slack_sender(webhook_url=get_webhook_url(), channel=get_channel())
def dump_each_idx(args):
    toknum = get_toknum(args.idx)
    print(f"### Starting idx: {args.idx}")
    print(f"### toknum: {toknum}")

    if args.idx != 0:
        corpus_file = pd.read_csv(os.path.join(args.corpus, f"idx_{args.idx}.csv"))
        corpus_file = corpus_file.fillna("")
        corpus = list(corpus_file['corpus'])
        context = list(corpus_file['context'])
        assert len(corpus) == len(context)
        print(f"### Loading Full Corpus")
        corpus_num = len(corpus)
        print(f"corpus_num: {corpus_num}")
    else:
        corpus = None
        context = None

    print(f"Saving in {args.save_path}")
    os.makedirs(args.save_path, exist_ok=True)
    emb_f = os.path.join(args.save_path, f"tokId_emb_{args.idx}.dat")
    if os.path.exists(emb_f): 
        emb_f = np.memmap(emb_f, dtype="float32", mode="readwrite", shape=(toknum, 1024))
        tokId2corpus = pickle.load(open(os.path.join(args.save_path, "tokText2corpus.pickle"), "rb"))
        tokText2tokIdList = pickle.load(open(os.path.join(args.save_path, "tokText2tokIdList.pickle"), "rb"))
        tokId2tokText = pickle.load(open(os.path.join(args.save_path, "tokId2tokText.pickle"), "rb"))
        corpusId_tokenList_dict = pickle.load(open(os.path.join(args.save_path, "corpusId_tokenList_dict.pickle"), "rb"))
        ids_dict = pickle.load(open(os.path.join(args.save_path, "ids.pickle"), "rb"))
        cur_tokId = ids_dict["cur_tokId"]
        corpusId = ids_dict["corpusId"]

    else:
        emb_f = np.memmap(emb_f, dtype="float32", mode="w+", shape=(toknum, 1024))
        emb_f.flush()
        tokId2corpus ={}; tokText2tokIdList = defaultdict(list); tokId2tokText={}; corpusId_tokenList_dict = {}
        cur_tokId = 0; corpusId = 0


    if args.t5:
        print(f'## Loading T5EncoderModel')
        model = T5EncoderModel.from_pretrained(args.emb_path).cuda()
        tokenizer = T5Tokenizer.from_pretrained(args.emb_path)
        tokId2corpus, tokText2tokIdList, tokId2tokText, corpusId_tokenList_dict = t5_construct_corpus(model, tokenizer, corpus, context, emb_f, tokId2tokText, tokText2tokIdList, tokId2corpus, corpusId_tokenList_dict, cur_tokId, corpusId)
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

    if args.idx == 0:
        dump("tokText2tokIdList.pickle", tokText2tokIdList)
        dump("tokId2tokText.pickle", tokId2tokText)
    else:
        dump("tokId2corpus.pickle", tokId2corpus)
        dump("tokText2tokIdList.pickle", tokText2tokIdList)
        dump("tokId2tokText.pickle", tokId2tokText)
        dump("corpusId_tokenList_dict.pickle", corpusId_tokenList_dict)

    print(f"==== DONE idx: {args.idx} ====")


def combine_all_idx(args):
    total_f = np.memmap(os.path.join(args.save_path, "tokId_emb.w_para.dat"), dtype="float32", mode="w+", shape=(45000000, 1024))
    path_list = [f"{args.save_path}/max_length_2000_idx_{i}" for i in range(args.idx_num)]
    #path_list = ["n_kilt_total_corpus.w_para.sub/max_length_2000_idx_0","n_kilt_total_corpus.w_para.sub/max_length_2000_idx_1/", "n_kilt_total_corpus.w_para.sub/idx_2", "n_kilt_total_corpus.w_para.sub/idx_3", "n_kilt_total_corpus.w_para.sub/max_length_2000_idx_4/", "n_kilt_total_corpus.w_para.sub/idx_5", "n_kilt_total_corpus.w_para.sub/idx_6", "n_kilt_total_corpus.w_para.sub/idx_7", "n_kilt_total_corpus.w_para.sub/idx_8", "n_kilt_total_corpus.w_para.sub/idx_9", "n_kilt_total_corpus.w_para.sub/idx_10", "n_kilt_total_corpus.w_para.sub/idx_11", "n_kilt_total_corpus.w_para.sub/max_length_2000_idx_12/"]#, "n_kilt_total_corpus.w_para.sub/max_length_2000_idx_13"]
    t_tokId = 0; t_corpusId = 0
    # tokId2corpus, tokText2tokIdList, tokId2tokText, corpusId_tokenList_dict
    total_corpus_list = []
    tokId2corpus = {}; tokText2tokIdList = defaultdict(list); tokId2tokText = {}; corpusId_tokenList_dict = {}; corpus_tokenList_dict = {}
    for i, _path in tqdm(enumerate(path_list)):
        print(f'File: {_path} // Start tokId: {t_tokId}')
        toknum = get_toknum(i)
        emb_f = np.memmap(os.path.join(_path, f"tokId_emb_{i}.dat"), dtype="float32", mode="readonly", shape=(toknum, 1024))
        if i == 0:
            _tokText2tokIdList = pickle.load(open(os.path.join(_path, "tokText2tokIdList.pickle"), "rb"))
            _tokId2tokText = pickle.load(open(os.path.join(_path, "tokId2tokText.pickle"), "rb"))

            for _tokId, _tokText in _tokId2tokText.items():
                tokId2tokText[_tokId] = _tokText 
                tokText2tokIdList[_tokText] = _tokText2tokIdList[_tokText]
                total_f[t_tokId][:] = emb_f[_tokId][:]
                t_tokId += 1
                if _tokId == 0: assert _tokText == "<pad>"
                if _tokId == 1: assert _tokText == "</s>"
        else:
            if i == 1: 
                assert t_tokId == 2
            _tokId2corpus = pickle.load(open(os.path.join(_path, "tokId2corpus.pickle"), "rb"))
            _tokText2tokIdList = pickle.load(open(os.path.join(_path, "tokText2tokIdList.pickle"), "rb")) 
            _tokId2tokText = pickle.load(open(os.path.join(_path, "tokId2tokText.pickle"), "rb")) 
            _corpusId_tokenList_dict = pickle.load(open(os.path.join(_path, "corpusId_tokenList_dict.pickle"), "rb")) 

            for c_corpusId, c_tokenList in _corpusId_tokenList_dict.items():
                t_tokenList = []
                t_corpus = _tokId2corpus[c_tokenList[0]]
                total_corpus_list.append(t_corpus)
                for c_tokId in c_tokenList[:-1]:
                    c_text = _tokId2tokText[c_tokId]
                    assert _tokId2corpus[c_tokId] == t_corpus, f"tokId: {c_tokId}\ncorpusId: {c_corpusId}\none is {t_corpus}\nthe other is {_tokId2corpus[c_tokId]}"
                    #c_corpus = _tokId2corpus[c_tokId]
                    tokText2tokIdList[c_text].append(t_tokId)
                    tokId2tokText[t_tokId] = c_text
                    total_f[t_tokId][:] = emb_f[c_tokId][:]
                    tokId2corpus[t_tokId] = t_corpus
                    t_tokenList.append(t_tokId)
                    t_tokId += 1
                t_tokenList.append(1)
                corpusId_tokenList_dict[t_corpusId] = t_tokenList
                corpus_tokenList_dict[total_corpus_list[-1]] = t_tokenList
                t_corpusId += 1

    total_f.flush()
    print(f'Total # of corpus: {len(set(total_corpus_list))}')
    
    # with open('total_corpus_list.json', 'w') as f:
    #     json.dump(list(set(total_corpus_list)), f)
    # sys.exit()

    dump("tokId2corpus.pickle", tokId2corpus)
    dump("tokText2tokIdList.pickle", tokText2tokIdList)
    dump("tokId2tokText.pickle", tokId2tokText)
    dump("corpusId_tokenList_dict.pickle", corpusId_tokenList_dict)
    dump("corpus_tokenList_dict.pickle", corpus_tokenList_dict)
    return tokId2corpus, tokText2tokIdList, tokId2tokText, corpusId_tokenList_dict, corpus_tokenList_dict

#@slack_sender(webhook_url=get_webhook_url(), channel=get_channel())
def t5_construct_group(tokText2tokIdList):
    tokGroupId2tokText = {}
    tokId2tokGroupId = {}
    tokGroupId2tokIdList = {}
    tokGroupId = 2 ## assert tokGroupId 1 is </s> for generate()
    tokTextList = list(tokText2tokIdList.keys())
    assert len(tokTextList) == len(set(tokTextList))

    for tokText, tokIdList in tokText2tokIdList.items():
        if tokText == "</s>":
            print(f"Found </s> and set it to 1!!!")
            tokGroupId2tokText[1] = tokText 
            tokGroupId2tokIdList[1] = tokIdList  
            for tokId in tokIdList:
                assert tokId not in tokId2tokGroupId.keys()
                tokId2tokGroupId[tokId] = 1 
        elif tokText == "<pad>":
            print(f"Found <pad> and set it to 0!!!")
            tokGroupId2tokText[0] = tokText 
            tokGroupId2tokIdList[0] = tokIdList  
            for tokId in tokIdList:
                assert tokId not in tokId2tokGroupId.keys()
                tokId2tokGroupId[tokId] = 0 
        else:
            tokGroupId2tokText[tokGroupId] = tokText
            tokGroupId2tokIdList[tokGroupId] = tokIdList
            for tokId in tokIdList:
                assert tokId not in tokId2tokGroupId.keys(), f"{tokId} exists!"
                #if tokId in tokId2tokGroupId: assert tokId2tokGroupId[tokId] == tokGroupId
                tokId2tokGroupId[tokId] = tokGroupId
            tokGroupId += 1
    return tokGroupId2tokText, tokId2tokGroupId, tokGroupId2tokIdList

def t5_construct_group_prefix_tree(corpusId_tokenList_dict):
    sys.setrecursionlimit(900000000)
    constrained_dict = {}

    for corpusId, tokIdList in corpusId_tokenList_dict.items():
        cur_dict = constrained_dict 
        tokGroupIdList = [tokId2tokGroupId[el] for el in tokIdList]
        tokGroupIdList = [0]+tokGroupIdList
        
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

def do_light_cluster(model, tokGroupId2tokIdList, clusterId, c_tokGroupId, clusterId2clusterEmb, tokId2tokGroupId):
    no_cluster = 0
    total_cluster = 0
    tokText_emb = {}
    
    # clusterId2clusterEmb = np.memmap(cluster_path, dtype="float32", mode="w+", shape=(37000000,1024)) 
    # clusterId = 0 # 0,1 is for <pad> and </s>

    start_idx = True
    for tokGroupId, tokIdList in tqdm(tokGroupId2tokIdList.items()):

        if tokGroupId < c_tokGroupId:
            continue

        if tokGroupId % 500000 == 0 and not start_idx:
            temp_dump_light_cluster(clusterId, tokGroupId, tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId)
            clusterId2clusterEmb.flush()

        text = tokId2tokText[tokIdList[0]]
        emb_list = [emb_f[_id][:] for _id in tokIdList]
        prev = False
        start_idx = False
        if len(emb_list) > args.cluster_num:
            prev = True
            # reduce the number of embedings to cluster_num by kmeans 
            df = pd.DataFrame(emb_list)
            if args.cluster_method == "k-means":
                # model = FaissKMeans(n_clusters=args.cluster_num)
                # centers = model.fit(np.array(emb_list), np.array(tokIdList))
                # predicts = np.array(model.predict(np.array(emb_list)))
                model = KMeans(n_clusters=args.cluster_num, algorithm='auto', max_iter=100, n_init=3)
                model.fit(df)
                predicts = np.array(model.predict(df))
                centers = model.cluster_centers_ # [15, 768]
            elif args.cluster_method == "gmm":
                gmm = GaussianMixture(n_components=args.cluster_num, random_state=0)
                gmm.fit(df)
                predicts = np.array(gmm.predict(df))
                label_list = [[] for _ in range(args.cluster_num)]
                for label, emb in zip(predicts, emb_list):
                    label_list[label].append(emb)
                centers = np.array([np.array(el).mean(axis=0) for el in label_list])
            else:
                assert False

        if not prev:
            # use the embedding 
            predicts = np.array([i for i in range(len(tokIdList))]) 
            centers = np.array(emb_list)

        clusterIdDict = {}
        for i, center in enumerate(centers):
            clusterIdDict[i] = clusterId 
            clusterId2tokText[clusterId] = text 
            clusterId2clusterEmb[clusterId][:] = center 
            clusterId += 1

        for _predict, _id in zip(predicts, tokIdList):
            _group = tokId2tokGroupId[_id]
            _clusterId = clusterIdDict[_predict]
            tokGroupId2clusterIdList[_group].append(_clusterId)
            clusterId2tokGroupId[_clusterId] = _group 
            tokId2clusterId[_id] = _clusterId 
            tokText2clusterIdList[text].append(_clusterId)
        
        if text == "<pad>" or text == "</s>":
            assert len(tokIdList) == 1 and len(tokText2clusterIdList[text]) == 1
        if args.t5:
            if clusterId == 1: assert tokId2clusterId[0] == 0
            if clusterId == 2: assert tokId2clusterId[1] == 1
        if args.bart:
            if clusterId == 1: assert tokId2clusterId[0] == 0
            if clusterId == 1: assert tokId2clusterId[0] == 0
    print(f'no_cluster: {no_cluster}\ttotal_cluster: {total_cluster}')
    clusterId2clusterEmb.flush()
    del clusterId2clusterEmb
    return tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId, cluster_path 



#@slack_sender(webhook_url=get_webhook_url(), channel=get_channel())
def do_cluster(model, tokGroupId2tokIdList, clusterId, c_tokGroupId, clusterId2clusterEmb, tokId2tokGroupId):
    no_cluster = 0
    total_cluster = 0
    tokText_emb = {}
    
    # clusterId2clusterEmb = np.memmap(cluster_path, dtype="float32", mode="w+", shape=(37000000,1024)) 
    # clusterId = 0 # 0,1 is for <pad> and </s>

    start_idx = True
    for tokGroupId, tokIdList in tqdm(tokGroupId2tokIdList.items()):

        if tokGroupId < c_tokGroupId:
            continue

        if tokGroupId % 500000 == 0 and not start_idx:
            temp_dump_cluster(clusterId, tokGroupId, tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId)
            clusterId2clusterEmb.flush()

        text = tokId2tokText[tokIdList[0]]
        prev = False
        start_idx = False
        emb_list = [emb_f[_id][:] for _id in tokIdList]
        if len(tokIdList) > args.cluster_num:
            prev = True
            # reduce the number of embedings to cluster_num by kmeans 
            df = pd.DataFrame(emb_list)
            if args.cluster_method == "k-means":
                # model = FaissKMeans(n_clusters=args.cluster_num)
                # centers = model.fit(np.array(emb_list), np.array(tokIdList))
                # predicts = np.array(model.predict(np.array(emb_list)))
                model = KMeans(n_clusters=args.cluster_num, algorithm='auto')
                model.fit(df)
                predicts = np.array(model.predict(df))
                centers = model.cluster_centers_ # [15, 768]
            elif args.cluster_method == "gmm":
                gmm = GaussianMixture(n_components=args.cluster_num, random_state=0)
                gmm.fit(df)
                predicts = np.array(gmm.predict(df))
                label_list = [[] for _ in range(args.cluster_num)]
                for label, emb in zip(predicts, emb_list):
                    label_list[label].append(emb)
                centers = np.array([np.array(el).mean(axis=0) for el in label_list])
            elif args.cluster_method == "dbscan":
                raise NotImplementedError('Bug Exists!')
                db = DBSCAN(eps=2.5, min_samples=5).fit(df)
                # db.labels_ : 클러스터 레이블값 : label이 -1이면 noise
                # db.core_sample_indices_ : 없는 값들은 outlier 
                #print(f'len(emb_list): {len(emb_list)}')
                labels = db.labels_
                print(f'# of labels: {set(labels)}')
                not_outlier = db.core_sample_indices_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters == 0:
                    print('# of cluster: 0 !!')
                prev = False
                no_cluster += 1
                total_cluster += 1
                #print(f'# of clusters: {n_clusters}')
                label_list = [[] for _ in range(n_clusters)]
                assert len(labels) == len(emb_list)
                for i, (label, emb) in enumerate(zip(labels, emb_list)):
                    if i in not_outlier:
                        label_list[label].append(emb)
                centers = np.array([np.array(el).mean(axis=0) for el in label_list])
                print(f"length of label_list: {len(label_list)}")
                print(f"shape of centers: {centers.shape}")

        if not prev:
            # use the embedding 
            predicts = np.array([i for i in range(len(tokIdList))]) 
            centers = np.array(emb_list)

        clusterIdDict = {}
        for i, center in enumerate(centers):
            clusterIdDict[i] = clusterId 
            clusterId2tokText[clusterId] = text 
            clusterId2clusterEmb[clusterId][:] = center 
            clusterId += 1

        for _predict, _id in zip(predicts, tokIdList):
            _group = tokId2tokGroupId[_id]
            _clusterId = clusterIdDict[_predict]
            tokGroupId2clusterIdList[_group].append(_clusterId)
            clusterId2tokGroupId[_clusterId] = _group 
            tokId2clusterId[_id] = _clusterId 
            tokText2clusterIdList[text].append(_clusterId)
        
        if text == "<pad>" or text == "</s>":
            assert len(tokIdList) == 1 and len(tokText2clusterIdList[text]) == 1
        if args.t5:
            if clusterId == 1: assert tokId2clusterId[0] == 0
            if clusterId == 2: assert tokId2clusterId[1] == 1
        if args.bart:
            if clusterId == 1: assert tokId2clusterId[0] == 0
            if clusterId == 1: assert tokId2clusterId[0] == 0
    print(f'no_cluster: {no_cluster}\ttotal_cluster: {total_cluster}')
    clusterId2clusterEmb.flush()
    del clusterId2clusterEmb
    return tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId, cluster_path 

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


def bi_construct_dataset(split, corpus2tokenList, emb_f):
    df = load_data(split)
    save_dict = {'input': [], 'output': [], 'output_tokid': []}
    data_len = len(df)
    for i in tqdm(range(data_len)):
        _input = df['input'][i]
        _output = df['output'][i]
        output_tok = corpus2tokenList[_output]
        output_emb = [emb_f[tok][:] for tok in output_tok]

        #if args.t5: assert output_tok[-1] == 1
        #if args.bart: assert output_tok[-1] == 2
        for _tok in output_tok[:-1]:
            save_dict['input'].append(_input)
            save_dict['output'].append(_output)
            save_dict['output_tokid'].append([_tok])
    return save_dict, f"bi_{args.data_name}_contextualized_{split}.pickle" 

def gr_construct_dataset(split, corpus2tokenList, emb_f):
    df = load_data(split)
    save_dict = {'input': [], 'output': [], 'output_tokid': []}
    data_len = len(df)
    for i in tqdm(range(data_len)):
        _input = df['input'][i]
        _output = df['output'][i]
        output_tok = corpus2tokenList[_output]
        output_emb = [emb_f[tok][:] for tok in output_tok]
        
        if args.t5: assert output_tok[-1] == 1
        if args.bart: assert output_tok[-1] == 2
        
        save_dict['input'].append(_input)
        save_dict['output'].append(_output)
        save_dict['output_tokid'].append(output_tok)

    return save_dict, f"gr_{args.data_name}_contextualized_{split}.pickle"

def construct_dataset(_dict, tokId2clusterId, split, _type):
    ret_dict = {'input': [], 'output': [], 'output_tokid': []}
    for _input, _output, _output_tokid in zip(_dict['input'], _dict['output'], _dict['output_tokid']):
        ret_dict['input'].append(_input)
        ret_dict['output'].append(_output)
        ret_dict['output_tokid'].append([tokId2clusterId[el] for el in _output_tokid])        
    return ret_dict, f"{_type}_{args.data_name}_{split}_cluster_{args.cluster_num}.pickle"

def temp_dump_light_cluster(clusterId, tokGroupId, tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId): 
    dump(f"light_tokGroupId2clusterIdList_{args.cluster_num}.pickle", tokGroupId2clusterIdList)
    dump(f"light_clusterId2tokGroupId_{args.cluster_num}.pickle", clusterId2tokGroupId)
    dump(f"light_clusterId2tokText_{args.cluster_num}.pickle", clusterId2tokText)
    dump(f"light_tokText2clusterIdList_{args.cluster_num}.pickle", tokText2clusterIdList)
    dump(f"light_tokId2clusterId_{args.cluster_num}.pickle", tokId2clusterId)
    dump(f'light_cur_info_{args.cluster_num}.pickle', {'tokGroupId': tokGroupId, 'clusterId': clusterId})



def temp_dump_cluster(clusterId, tokGroupId, tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId): 
    dump(f"tokGroupId2clusterIdList_{args.cluster_num}.pickle", tokGroupId2clusterIdList)
    dump(f"clusterId2tokGroupId_{args.cluster_num}.pickle", clusterId2tokGroupId)
    dump(f"clusterId2tokText_{args.cluster_num}.pickle", clusterId2tokText)
    dump(f"tokText2clusterIdList_{args.cluster_num}.pickle", tokText2clusterIdList)
    dump(f"tokId2clusterId_{args.cluster_num}.pickle", tokId2clusterId)
    dump(f'cur_info_{args.cluster_num}.pickle', {'tokGroupId': tokGroupId, 'clusterId': clusterId})


# @slack_sender(webhook_url=get_webhook_url(), channel=get_channel())
# def do_cluster(model, tokGroupId2tokIdList):
#     no_cluster = 0
#     total_cluster = 0
#     tokText_emb = {}

#     tokText2clusterIdList = defaultdict(list)
#     tokId2clusterId = {}
#     tokGroupId2clusterIdList = defaultdict(list)
#     clusterId2tokGroupId = {}
#     clusterId2tokText = {}
#     clusterId2clusterEmb = {}
#     clusterId = 0 # 0,1 is for <pad> and </s>


#     for tokGroupId, tokIdList in tqdm(tokGroupId2tokIdList.items()):
#         text = tokId2tokText[tokIdList[0]]
#         emb_list = [emb_f[_id][:] for _id in tokIdList]
#         prev = False

#         if len(emb_list) > args.cluster_num:
#             prev = True
#             # reduce the number of embedings to cluster_num by kmeans 
#             df = pd.DataFrame(emb_list)
#             if args.cluster_method == "k-means":
#                 model = FaissKMeans(n_clusters=args.cluster_num)
#                 centers = model.fit(np.array(emb_list), np.array(tokIdList))
#                 predicts = np.array(model.predict(np.array(emb_list)))
#                 #model = KMeans(n_clusters=args.cluster_num, algorithm='auto')
#                 #model.fit(df)
#                 #predicts = np.array(model.predict(df))
#                 #centers = model.cluster_centers_ # [15, 768]
#             elif args.cluster_method == "gmm":
#                 gmm = GaussianMixture(n_components=args.cluster_num, random_state=0)
#                 gmm.fit(df)
#                 predicts = np.array(gmm.predict(df))
#                 label_list = [[] for _ in range(args.cluster_num)]
#                 for label, emb in zip(predicts, emb_list):
#                     label_list[label].append(emb)
#                 centers = np.array([np.array(el).mean(axis=0) for el in label_list])
#             elif args.cluster_method == "dbscan":
#                 raise NotImplementedError('Bug Exists!')
#                 db = DBSCAN(eps=2.5, min_samples=5).fit(df)
#                 # db.labels_ : 클러스터 레이블값 : label이 -1이면 noise
#                 # db.core_sample_indices_ : 없는 값들은 outlier 
#                 #print(f'len(emb_list): {len(emb_list)}')
#                 labels = db.labels_
#                 print(f'# of labels: {set(labels)}')
#                 not_outlier = db.core_sample_indices_
#                 n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#                 if n_clusters == 0:
#                     print('# of cluster: 0 !!')
#                 prev = False
#                 no_cluster += 1
#                 total_cluster += 1
#                 #print(f'# of clusters: {n_clusters}')
#                 label_list = [[] for _ in range(n_clusters)]
#                 assert len(labels) == len(emb_list)
#                 for i, (label, emb) in enumerate(zip(labels, emb_list)):
#                     if i in not_outlier:
#                         label_list[label].append(emb)
#                 centers = np.array([np.array(el).mean(axis=0) for el in label_list])
#                 print(f"length of label_list: {len(label_list)}")
#                 print(f"shape of centers: {centers.shape}")

#         if not prev:
#             # use the embedding 
#             predicts = np.array([i for i in range(len(tokIdList))]) 
#             centers = np.array(emb_list)

#         clusterIdDict = {}
#         for i, center in enumerate(centers):
#             clusterIdDict[i] = clusterId 
#             clusterId2tokText[clusterId] = text 
#             clusterId2clusterEmb[clusterId] = center 
#             clusterId += 1

#         for _predict, _id in zip(predicts, tokIdList):
#             _group = tokId2tokGroupId[_id]
#             _clusterId = clusterIdDict[_predict]
#             tokGroupId2clusterIdList[_group].append(_clusterId)
#             clusterId2tokGroupId[_clusterId] = _group 
#             tokId2clusterId[_id] = _clusterId 
#             tokText2clusterIdList[text].append(_clusterId)
        
#         if text == "<pad>" or text == "</s>":
#             assert len(tokIdList) == 1 and len(tokText2clusterIdList[text]) == 1
#         if args.t5:
#             if clusterId == 1: assert tokId2clusterId[0] == 0
#             if clusterId == 2: assert tokId2clusterId[1] == 1
#         if args.bart:
#             if clusterId == 1: assert tokId2clusterId[0] == 0
#             if clusterId == 1: assert tokId2clusterId[0] == 0
#     print(f'no_cluster: {no_cluster}\ttotal_cluster: {total_cluster}')

#     return tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId, clusterId2clusterEmb

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--corpus", default=None, required=True, type=str)
    parser.add_argument("--data_name", default=None, required=True, type=str)
    parser.add_argument("--train_file", default=None, required=True, type=str)
    parser.add_argument("--dev_file", default=None, required=True, type=str)
    parser.add_argument("--test_file", default=None, required=True, type=str)
    parser.add_argument("--save_path", default=None, required=True, type=str)
    parser.add_argument("--emb_path", default=None, required=True, type=str)
    parser.add_argument("--cluster_method", default="k-means", type=str)
    parser.add_argument("--dump_batch", default=10, type=int)
    parser.add_argument("--cluster_num", default=5, type=int)
    parser.add_argument("--idx", default=-1, type=int)
    parser.add_argument("--idx_num", default=-1, type=int)
    parser.add_argument("--max_length", default=2000, type=int)
    parser.add_argument("--t5", action='store_true')
    parser.add_argument("--bart", action='store_true')
    parser.add_argument("--action", required=True, type=str)
    args = parser.parse_args()


    if args.action == "dump":
        args.save_path = os.path.join(args.save_path, f"max_length_{args.max_length}_idx_{args.idx}")
        dump_each_idx(args)
    elif args.action == "combine":
        tokId2corpus, tokText2tokIdList, tokId2tokText, corpusId_tokenList_dict, corpus_tokenList_dict = combine_all_idx(args)
    elif args.action == "group":
        tokText2tokIdList = pickle.load(open(os.path.join(args.save_path, "tokText2tokIdList.pickle"), 'rb'))
        corpusId_tokenList_dict = pickle.load(open(os.path.join(args.save_path, "corpusId_tokenList_dict.pickle"), 'rb'))
        tokGroupId2tokText, tokId2tokGroupId, tokGroupId2tokIdList = t5_construct_group(tokText2tokIdList)
        assert 2 in tokId2tokGroupId
        group_trie = t5_construct_group_prefix_tree(corpusId_tokenList_dict)

        dump("tokGroupId2tokText.pickle", tokGroupId2tokText)
        dump("tokId2tokGroupId.pickle", tokId2tokGroupId)
        dump("tokGroupId2tokIdList.pickle", tokGroupId2tokIdList)
        dump("group_trie.pickle", group_trie)

    elif args.action == "dataset":
        print(f"Action: Construct DataSet {args.data_name}!")

        corpus_file = pd.read_csv("n_kilt_total_corpus.csv")
        corpus_file = corpus_file.fillna("")
        corpus = list(corpus_file['corpus'])

        emb_f = np.memmap(os.path.join(args.save_path, "tokId_emb.w_para.dat"), dtype="float32", mode="readonly", shape=(45000000, 1024))
        corpus_tokenList_dict = pickle.load(open(os.path.join(args.save_path, "corpus_tokenList_dict.pickle"), "rb"))

        corpus2tokenList = {}
        for corpus, tokenList in corpus_tokenList_dict.items():
            corpus2tokenList[corpus] = tokenList

        train_dict, train_fname = bi_construct_dataset("train", corpus2tokenList, emb_f)
        dev_dict, dev_fname = bi_construct_dataset("dev", corpus2tokenList, emb_f)
        test_dict, test_fname = bi_construct_dataset("test", corpus2tokenList, emb_f)
        dump(train_fname, train_dict)
        dump(dev_fname, dev_dict)
        dump(test_fname, test_dict)   

        train_dict, train_fname = gr_construct_dataset("train", corpus2tokenList, emb_f)
        dev_dict, dev_fname = gr_construct_dataset("dev", corpus2tokenList, emb_f)
        test_dict, test_fname = gr_construct_dataset("test", corpus2tokenList, emb_f)
        dump(train_fname, train_dict)
        dump(dev_fname, dev_dict)
        dump(test_fname, test_dict)   

    elif args.action == "cluster":   
        print(f'!!!DO Cluster!!!')
        cluster_path = os.path.join(args.save_path, f"clusterId_emb_{args.cluster_num}.dat")

        if f"cur_info_{args.cluster_num}.pickle" in os.listdir(args.save_path):
            tokGroupId2clusterIdList = pickle.load(open(os.path.join(args.save_path, f'tokGroupId2clusterIdList_{args.cluster_num}.pickle'), "rb"))
            clusterId2tokGroupId = pickle.load(open(os.path.join(args.save_path, f'clusterId2tokGroupId_{args.cluster_num}.pickle'), "rb"))
            clusterId2tokText = pickle.load(open(os.path.join(args.save_path, f'clusterId2tokText_{args.cluster_num}.pickle'), "rb"))
            tokText2clusterIdList = pickle.load(open(os.path.join(args.save_path, f'tokText2clusterIdList_{args.cluster_num}.pickle'), "rb"))
            tokId2clusterId = pickle.load(open(os.path.join(args.save_path, f'tokId2clusterId_{args.cluster_num}.pickle'), "rb"))
            cur_info = pickle.load(open(os.path.join(args.save_path, f'cur_info_{args.cluster_num}.pickle'), "rb"))
            c_tokGroupId = cur_info['tokGroupId']
            clusterId = cur_info['clusterId']
            clusterId2clusterEmb = np.memmap(cluster_path, dtype="float32", mode="readwrite", shape=(45000000,1024)) 
        else:
            tokGroupId2clusterIdList = defaultdict(list)
            clusterId2tokGroupId = {}
            clusterId2tokText = {}
            tokText2clusterIdList = defaultdict(list)
            tokId2clusterId = {}
            clusterId = 0
            c_tokGroupId = 0
            clusterId2clusterEmb = np.memmap(cluster_path, dtype="float32", mode="w+", shape=(45000000,1024)) 
        clusterId2clusterEmb.flush()

        emb_f = np.memmap(os.path.join(args.save_path, "tokId_emb.w_para.dat"), dtype="float32", mode="readonly", shape=(45000000, 1024))
        tokGroupId2tokIdList = pickle.load(open(os.path.join(args.save_path, "tokGroupId2tokIdList.pickle"), 'rb'))
        tokId2tokText = pickle.load(open(os.path.join(args.save_path, "tokId2tokText.pickle"), 'rb'))
        tokId2tokGroupId = pickle.load(open(os.path.join(args.save_path, "tokId2tokGroupId.pickle"), 'rb'))

        model = T5EncoderModel.from_pretrained(args.emb_path).cuda()

        tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId, cluster_path = do_cluster(model, tokGroupId2tokIdList, clusterId, c_tokGroupId, clusterId2clusterEmb, tokId2tokGroupId)
        
        #clusterIdList2corpusId = get_clusterIdList2corpusId(corpusId_tokenList_dict, tokId2clusterId)


        dump(f"tokGroupId2clusterIdList_{args.cluster_num}.pickle", tokGroupId2clusterIdList)
        dump(f"clusterId2tokGroupId_{args.cluster_num}.pickle", clusterId2tokGroupId)
        dump(f"clusterId2tokText_{args.cluster_num}.pickle", clusterId2tokText)
        dump(f"tokText2clusterIdList_{args.cluster_num}.pickle", tokText2clusterIdList)
        dump(f"tokId2clusterId_{args.cluster_num}.pickle", tokId2clusterId)

    elif args.action == "light_cluster":   
        print(f'!!!DO Light Cluster!!!')
        cluster_path = os.path.join(args.save_path, f"light_clusterId_emb_{args.cluster_num}.dat")

        if f"light_cur_info_{args.cluster_num}.pickle" in os.listdir(args.save_path):
            tokGroupId2clusterIdList = pickle.load(open(os.path.join(args.save_path, f'light_tokGroupId2clusterIdList_{args.cluster_num}.pickle'), "rb"))
            clusterId2tokGroupId = pickle.load(open(os.path.join(args.save_path, f'light_clusterId2tokGroupId_{args.cluster_num}.pickle'), "rb"))
            clusterId2tokText = pickle.load(open(os.path.join(args.save_path, f'light_clusterId2tokText_{args.cluster_num}.pickle'), "rb"))
            tokText2clusterIdList = pickle.load(open(os.path.join(args.save_path, f'light_tokText2clusterIdList_{args.cluster_num}.pickle'), "rb"))
            tokId2clusterId = pickle.load(open(os.path.join(args.save_path, f'light_tokId2clusterId_{args.cluster_num}.pickle'), "rb"))
            cur_info = pickle.load(open(os.path.join(args.save_path, f'light_cur_info_{args.cluster_num}.pickle'), "rb"))
            c_tokGroupId = cur_info['tokGroupId']
            clusterId = cur_info['clusterId']
            clusterId2clusterEmb = np.memmap(cluster_path, dtype="float32", mode="readwrite", shape=(45000000,1024)) 
        else:
            tokGroupId2clusterIdList = defaultdict(list)
            clusterId2tokGroupId = {}
            clusterId2tokText = {}
            tokText2clusterIdList = defaultdict(list)
            tokId2clusterId = {}
            clusterId = 0
            c_tokGroupId = 0
            clusterId2clusterEmb = np.memmap(cluster_path, dtype="float32", mode="w+", shape=(45000000,1024)) 
        clusterId2clusterEmb.flush()

        emb_f = np.memmap(os.path.join(args.save_path, "tokId_emb.w_para.dat"), dtype="float32", mode="readonly", shape=(45000000, 1024))
        tokGroupId2tokIdList = pickle.load(open(os.path.join(args.save_path, "tokGroupId2tokIdList.pickle"), 'rb'))
        tokId2tokText = pickle.load(open(os.path.join(args.save_path, "tokId2tokText.pickle"), 'rb'))
        tokId2tokGroupId = pickle.load(open(os.path.join(args.save_path, "tokId2tokGroupId.pickle"), 'rb'))

        model = T5EncoderModel.from_pretrained(args.emb_path).cuda()

        tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId, cluster_path = do_light_cluster(model, tokGroupId2tokIdList, clusterId, c_tokGroupId, clusterId2clusterEmb, tokId2tokGroupId)
        
        #clusterIdList2corpusId = get_clusterIdList2corpusId(corpusId_tokenList_dict, tokId2clusterId)


        dump(f"light_tokGroupId2clusterIdList_{args.cluster_num}.pickle", tokGroupId2clusterIdList)
        dump(f"light_clusterId2tokGroupId_{args.cluster_num}.pickle", clusterId2tokGroupId)
        dump(f"light_clusterId2tokText_{args.cluster_num}.pickle", clusterId2tokText)
        dump(f"light_tokText2clusterIdList_{args.cluster_num}.pickle", tokText2clusterIdList)
        dump(f"light_tokId2clusterId_{args.cluster_num}.pickle", tokId2clusterId)




    elif args.action == "cluster_dataset":
        print('Dumping full ver.')
        cluster_path = os.path.join(args.save_path, f"clusterId_emb_{args.cluster_num}.dat")
        clusterId2clusterEmb = np.memmap(cluster_path, dtype="float32", mode="readonly", shape=(45000000, 1024))
        tokId2clusterId = pickle.load(open(os.path.join(args.save_path, f'tokId2clusterId_{args.cluster_num}.pickle'), "rb"))

        train_dict = pickle.load(open(os.path.join(args.save_path, f"gr_{args.data_name}_contextualized_train.pickle"), 'rb'))
        dev_dict = pickle.load(open(os.path.join(args.save_path, f"gr_{args.data_name}_contextualized_dev.pickle"), 'rb'))
        test_dict = pickle.load(open(os.path.join(args.save_path, f"gr_{args.data_name}_contextualized_test.pickle"), 'rb'))

        train_dict, train_fname = construct_dataset(train_dict, tokId2clusterId, "train", "gr")
        dev_dict, dev_fname = construct_dataset(dev_dict, tokId2clusterId, "dev", "gr")
        test_dict, test_fname = construct_dataset(test_dict, tokId2clusterId, "test", "gr")
        dump(train_fname, train_dict)
        dump(dev_fname, dev_dict)
        dump(test_fname, test_dict)   


        train_dict = pickle.load(open(os.path.join(args.save_path, f"bi_{args.data_name}_contextualized_train.pickle"), 'rb'))
        dev_dict = pickle.load(open(os.path.join(args.save_path, f"bi_{args.data_name}_contextualized_dev.pickle"), 'rb'))
        test_dict = pickle.load(open(os.path.join(args.save_path, f"bi_{args.data_name}_contextualized_test.pickle"), 'rb'))

        train_dict, train_fname = construct_dataset(train_dict, tokId2clusterId, "train", "bi")
        dev_dict, dev_fname = construct_dataset(dev_dict, tokId2clusterId, "dev", "bi")
        test_dict, test_fname = construct_dataset(test_dict, tokId2clusterId, "test", "bi")
        dump(train_fname, train_dict)
        dump(dev_fname, dev_dict)
        dump(test_fname, test_dict) 

        """
        print('Dumping light ver.')
        cluster_path = os.path.join(args.save_path, f"light_clusterId_emb_{args.cluster_num}.dat")
        clusterId2clusterEmb = np.memmap(cluster_path, dtype="float32", mode="readonly", shape=(45000000, 1024))
        tokId2clusterId = pickle.load(open(os.path.join(args.save_path, f'light_tokId2clusterId_{args.cluster_num}.pickle'), "rb"))

        train_dict = pickle.load(open(os.path.join(args.save_path, f"gr_{args.data_name}_contextualized_train.pickle"), 'rb'))
        dev_dict = pickle.load(open(os.path.join(args.save_path, f"gr_{args.data_name}_contextualized_dev.pickle"), 'rb'))
        test_dict = pickle.load(open(os.path.join(args.save_path, f"gr_{args.data_name}_contextualized_test.pickle"), 'rb'))

        train_dict, train_fname = construct_dataset(train_dict, tokId2clusterId, "train", "light_gr")
        dev_dict, dev_fname = construct_dataset(dev_dict, tokId2clusterId, "dev", "light_gr")
        test_dict, test_fname = construct_dataset(test_dict, tokId2clusterId, "test", "light_gr")
        dump(train_fname, train_dict)
        dump(dev_fname, dev_dict)
        dump(test_fname, test_dict)   
      

        train_dict = pickle.load(open(os.path.join(args.save_path, f"bi_{args.data_name}_contextualized_train.pickle"), 'rb'))
        dev_dict = pickle.load(open(os.path.join(args.save_path, f"bi_{args.data_name}_contextualized_dev.pickle"), 'rb'))
        test_dict = pickle.load(open(os.path.join(args.save_path, f"bi_{args.data_name}_contextualized_test.pickle"), 'rb'))

        train_dict, train_fname = construct_dataset(train_dict, tokId2clusterId, "train", "light_bi")
        dev_dict, dev_fname = construct_dataset(dev_dict, tokId2clusterId, "dev", "light_bi")
        test_dict, test_fname = construct_dataset(test_dict, tokId2clusterId, "test", "light_bi")
        dump(train_fname, train_dict)
        dump(dev_fname, dev_dict)
        dump(test_fname, test_dict)  
        """

    print("==== DONE ====")
