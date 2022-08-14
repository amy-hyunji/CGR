import os
import sys
import faiss
import argparse
import pickle
import pandas as pd
import numpy as np

from argparse import ArgumentParser
from transformers import BartTokenizer, BartModel, T5EncoderModel, T5Tokenizer
from tqdm import tqdm 
from knockknock import slack_sender
#from slack import get_webhook_url, get_channel
from collections import defaultdict
from sklearn.cluster import KMeans


class FaissKMeans:
    def __init__(self, n_clusters=10, n_init=100, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X, y):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init,
                                   gpu=True
                                   )
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]
        return self.cluster_centers_

    def predict(self, X):
        ret = self.kmeans.index.search(X.astype(np.float32), 1)[1]
        return [elem[0] for elem in ret]

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

#@slack_sender(webhook_url=get_webhook_url(), channel=get_channel())
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

def bart_construct_group(tokText2tokIdList):
    tokGroupId2tokText = {}
    tokId2tokGroupId = {}
    tokGroupId2tokIdList = {}
    tokGroupId = 4 ## assert tokGroupId 1 is </s> for generate()
    tokTextList = list(tokText2tokIdList.keys())
    assert len(tokTextList) == len(set(tokTextList))

    for tokText, tokIdList in tokText2tokIdList.items():
        if tokText == "<s>":
            print(f"Found <s> and set it to 0!!!")
            tokGroupId2tokText[0] = tokText 
            tokGroupId2tokIdList[0] = tokIdList  
            for tokId in tokIdList:
                assert tokId not in tokId2tokGroupId.keys()
                tokId2tokGroupId[tokId] = 0 
        elif tokText == "<pad>":
            print(f"Found <pad> and set it to 1!!!")
            tokGroupId2tokText[1] = tokText 
            tokGroupId2tokIdList[1] = tokIdList  
            for tokId in tokIdList:
                assert tokId not in tokId2tokGroupId.keys()
                tokId2tokGroupId[tokId] = 1
        elif tokText == "</s>":
            print(f"Found </s> and set it to 2!!!")
            tokGroupId2tokText[2] = tokText 
            tokGroupId2tokIdList[2] = tokIdList  
            for tokId in tokIdList:
                assert tokId not in tokId2tokGroupId.keys()
                tokId2tokGroupId[tokId] = 2
        elif tokText == "<unk>":
            print(f"Found <unk> and set it to 3!!!")
            tokGroupId2tokText[3] = tokText 
            tokGroupId2tokIdList[3] = tokIdList  
            for tokId in tokIdList:
                assert tokId not in tokId2tokGroupId.keys()
                tokId2tokGroupId[tokId] = 3
        else:
            tokGroupId2tokText[tokGroupId] = tokText
            tokGroupId2tokIdList[tokGroupId] = tokIdList
            for tokId in tokIdList:
                assert tokId not in tokId2tokGroupId.keys()
                tokId2tokGroupId[tokId] = tokGroupId
            tokGroupId += 1
    return tokGroupId2tokText, tokId2tokGroupId, tokGroupId2tokIdList

def bart_construct_group_prefix_tree(corpusId_tokenList_dict):
    sys.setrecursionlimit(900000000)
    constrained_dict = {}

    for corpusId, tokIdList in corpusId_tokenList_dict.items():
        cur_dict = constrained_dict 
        tokGroupIdList = [tokId2tokGroupId[el] for el in tokIdList]
        tokGroupIdList = [2, 0]+tokGroupIdList
        
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
                assert tokId not in tokId2tokGroupId.keys()
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

        if args.t5: assert output_tok[-1] == 1
        if args.bart: assert output_tok[-1] == 2
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

def dump(fname, file):
    with open(os.path.join(args.save_path, fname), "wb") as f:
        pickle.dump(file, f)

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

def temp_dump_light_cluster(clusterId, tokGroupId, tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId): 
    dump(f"light_tokGroupId2clusterIdList_{arg.cluster_num}.pickle", tokGroupId2clusterIdList)
    dump(f"light_clusterId2tokGroupId_{args.cluster_num}.pickle", clusterId2tokGroupId)
    dump(f"light_clusterId2tokText_{args.cluster_num}.pickle", clusterId2tokText)
    dump(f"light_tokText2clusterIdList_{args.cluster_num}.pickle", tokText2clusterIdList)
    dump(f"light_tokId2clusterId_{args.cluster_num}.pickle", tokId2clusterId)
    dump(f'light_cur_info_{args.cluster_num}.pickle', {'tokGroupId': tokGroupId, 'clusterId': clusterId})



def temp_dump_cluster(clusterId, tokGroupId, tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId): 
    dump(f"tokGroupId2clusterIdList_{arg.cluster_num}.pickle", tokGroupId2clusterIdList)
    dump(f"clusterId2tokGroupId_{args.cluster_num}.pickle", clusterId2tokGroupId)
    dump(f"clusterId2tokText_{args.cluster_num}.pickle", clusterId2tokText)
    dump(f"tokText2clusterIdList_{args.cluster_num}.pickle", tokText2clusterIdList)
    dump(f"tokId2clusterId_{args.cluster_num}.pickle", tokId2clusterId)
    dump(f'cur_info_{args.cluster_num}.pickle', {'tokGroupId': tokGroupId, 'clusterId': clusterId})

def construct_dataset(_dict, tokId2clusterId, split):
    ret_dict = {'input': [], 'output': [], 'output_tokid': []}
    for _input, _output, _output_tokid in zip(_dict['input'], _dict['output'], _dict['output_tokid']):
        ret_dict['input'].append(_input)
        ret_dict['output'].append(_output)
        ret_dict['output_tokid'].append([tokId2clusterId[el] for el in _output_tokid])        
    return ret_dict, f"gr_{args.data_name}_{split}_cluster_{args.cluster_num}.pickle"

def get_clusterIdList2corpusId(corpusId_tokenList_dict, tokId2clusterId):
   groupList2corpusId = {}
   for corpusId, tokenList in corpusId_tokenList_dict.items():
      groupList = [tokId2clusterId[el] for el in tokenList]
      groupList = tuple(groupList)
      groupList2corpusId[groupList] = corpusId
   return groupList2corpusId

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--corpus", default=None, required=True, type=str)
    parser.add_argument("--data_name", default=None, required=True, type=str)
    parser.add_argument("--train_file", default=None, required=True, type=str)
    parser.add_argument("--dev_file", default=None, required=True, type=str)
    parser.add_argument("--test_file", default=None, required=True, type=str)
    parser.add_argument("--save_path", default=None, required=True, type=str)
    parser.add_argument("--emb_path", default=None, required=True, type=str)
    parser.add_argument("--action", default=None, required=True, type=str)
    parser.add_argument("--cluster_method", default="k-means", type=str)
    parser.add_argument("--dump_batch", default=10, type=int)
    parser.add_argument("--cluster_num", default=5, type=int)
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

    if args.action == "dump":
        print("Action: Dump Embedding!")
        emb_f = os.path.join(args.save_path, f"tokId_emb.dat")
        if os.path.exists(emb_f): os.system(f"rm {emb_f}")
        emb_f = np.memmap(emb_f, dtype="float32", mode="w+", shape=(37000000, 1024))
        emb_f.flush()

        if args.t5:
            print(f'## Loading T5EncoderModel')
            model = T5EncoderModel.from_pretrained(args.emb_path).cuda()
            tokenizer = T5Tokenizer.from_pretrained(args.emb_path)
            tokId2corpus, tokText2tokIdList, tokId2tokText, corpusId_tokenList_dict = t5_construct_corpus(model, tokenizer, corpus, None, emb_f)
            tokGroupId2tokText, tokId2tokGroupId, tokGroupId2tokIdList = t5_construct_group(tokText2tokIdList)
            group_trie = t5_construct_group_prefix_tree(corpusId_tokenList_dict)
        elif args.bart:
            print(f'## Loading BartModel')
            model = BartModel.from_pretrained(args.emb_path).get_encoder().cuda()
            tokenizer = BartTokenizer.from_pretrained(args.emb_path)
            tokId2corpus, tokText2tokIdList, tokId2tokText, corpusId_tokenList_dict = bart_construct_corpus(model, tokenizer, corpus, None, emb_f)
            tokGroupId2tokText, tokId2tokGroupId, tokGroupId2tokIdList = bart_construct_group(tokText2tokIdList)
            group_trie = bart_construct_group_prefix_tree(corpusId_tokenList_dict) 
        else:
            assert False

        dump("tokId2corpus.pickle", tokId2corpus)
        dump("tokText2tokIdList.pickle", tokText2tokIdList)
        dump("tokId2tokText.pickle", tokId2tokText)
        dump("corpusId_tokenList_dict.pickle", corpusId_tokenList_dict)
        dump("tokGroupId2tokText.pickle", tokGroupId2tokText)
        dump("tokId2tokGroupId.pickle", tokId2tokGroupId)
        dump("tokGroupId2tokIdList.pickle", tokGroupId2tokIdList)
        dump("group_trie.pickle", group_trie)
        del emb_f
        print('******** DONE DUMPING!!! *********')

    elif args.action == "dataset":
        print(f"Action: Construct DataSet {args.data_name}!")
        emb_f = np.memmap(os.path.join(args.save_path, "tokId_emb.dat"), dtype="float32", mode="readonly", shape=(37000000, 1024))
        corpusId_tokenList_dict = pickle.load(open(os.path.join(args.save_path, "corpusId_tokenList_dict.pickle"), "rb"))

        corpus2tokenList = {}
        for corpusId, tokenList in corpusId_tokenList_dict.items():
            corpus2tokenList[corpus[corpusId]] = tokenList

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

        print(f'Dump Dataset for {args.data_name}!')

    elif args.action == "cluster":
        print("Action: Do Cluster!")
        cluster_path = os.path.join(args.save_path, f"clusterId_emb_{args.cluster_num}.dat")
        model = T5EncoderModel.from_pretrained(args.emb_path).cuda()

        if f"cur_info_{args.cluster_num}.pickle" in os.listdir(args.save_path):
            tokGroupId2clusterIdList = pickle.load(open(os.path.join(args.save_path, f'tokGroupId2clusterIdList_{args.cluster_num}.pickle'), "rb"))
            clusterId2tokGroupId = pickle.load(open(os.path.join(args.save_path, f'clusterId2tokGroupId_{args.cluster_num}.pickle'), "rb"))
            clusterId2tokText = pickle.load(open(os.path.join(args.save_path, f'clusterId2tokText_{args.cluster_num}.pickle'), "rb"))
            tokText2clusterIdList = pickle.load(open(os.path.join(args.save_path, f'tokText2clusterIdList_{args.cluster_num}.pickle'), "rb"))
            tokId2clusterId = pickle.load(open(os.path.join(args.save_path, f'tokId2clusterId_{args.cluster_num}.pickle'), "rb"))
            cur_info = pickle.load(open(os.path.join(args.save_path, f'cur_info_{args.cluster_num}.pickle'), "rb"))
            c_tokGroupId = cur_info['tokGroupId']
            clusterId = cur_info['clusterId']
            clusterId2clusterEmb = np.memmap(cluster_path, dtype="float32", mode="readwrite", shape=(37000000,1024)) 
        else:
            tokGroupId2clusterIdList = defaultdict(list)
            clusterId2tokGroupId = {}
            clusterId2tokText = {}
            tokText2clusterIdList = defaultdict(list)
            tokId2clusterId = {}
            clusterId = 0
            c_tokGroupId = 0
            clusterId2clusterEmb = np.memmap(cluster_path, dtype="float32", mode="w+", shape=(37000000,1024)) 
        clusterId2clusterEmb.flush()

        emb_f = np.memmap(os.path.join(args.save_path, "tokId_emb.dat"), dtype="float32", mode="readonly", shape=(37000000, 1024))
        tokGroupId2tokIdList = pickle.load(open(os.path.join(args.save_path, "tokGroupId2tokIdList.pickle"), 'rb'))
        tokId2tokText = pickle.load(open(os.path.join(args.save_path, "tokId2tokText.pickle"), 'rb'))
        tokId2tokGroupId = pickle.load(open(os.path.join(args.save_path, "tokId2tokGroupId.pickle"), 'rb'))
        assert tokId2tokGroupId[0] == 0

        tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId, cluster_path = do_cluster(model, tokGroupId2tokIdList, clusterId, c_tokGroupId, clusterId2clusterEmb, tokId2tokGroupId)

        #clusterIdList2corpusId = get_clusterIdList2corpusId(corpusId_tokenList_dict, tokId2clusterId)
        dump(f"tokGroupId2clusterIdList_{arg.cluster_num}.pickle", tokGroupId2clusterIdList)
        dump(f"clusterId2tokGroupId_{args.cluster_num}.pickle", clusterId2tokGroupId)
        dump(f"clusterId2tokText_{args.cluster_num}.pickle", clusterId2tokText)
        dump(f"tokText2clusterIdList_{args.cluster_num}.pickle", tokText2clusterIdList)
        dump(f"tokId2clusterId_{args.cluster_num}.pickle", tokId2clusterId)

    elif args.action == "light_cluster":
        print("Action: Do Light Cluster!")
        cluster_path = os.path.join(args.save_path, f"clusterId_emb_{args.cluster_num}.dat")
        model = T5EncoderModel.from_pretrained(args.emb_path).cuda()

        if f"light_cur_info_{args.cluster_num}.pickle" in os.listdir(args.save_path):
            tokGroupId2clusterIdList = pickle.load(open(os.path.join(args.save_path, f'light_tokGroupId2clusterIdList_{args.cluster_num}.pickle'), "rb"))
            clusterId2tokGroupId = pickle.load(open(os.path.join(args.save_path, f'light_clusterId2tokGroupId_{args.cluster_num}.pickle'), "rb"))
            clusterId2tokText = pickle.load(open(os.path.join(args.save_path, f'light_clusterId2tokText_{args.cluster_num}.pickle'), "rb"))
            tokText2clusterIdList = pickle.load(open(os.path.join(args.save_path, f'light_tokText2clusterIdList_{args.cluster_num}.pickle'), "rb"))
            tokId2clusterId = pickle.load(open(os.path.join(args.save_path, f'light_tokId2clusterId_{args.cluster_num}.pickle'), "rb"))
            cur_info = pickle.load(open(os.path.join(args.save_path, f'light_cur_info_{args.cluster_num}.pickle'), "rb"))
            c_tokGroupId = cur_info['tokGroupId']
            clusterId = cur_info['clusterId']
            clusterId2clusterEmb = np.memmap(cluster_path, dtype="float32", mode="readwrite", shape=(37000000,1024)) 
        else:
            tokGroupId2clusterIdList = defaultdict(list)
            clusterId2tokGroupId = {}
            clusterId2tokText = {}
            tokText2clusterIdList = defaultdict(list)
            tokId2clusterId = {}
            clusterId = 0
            c_tokGroupId = 0
            clusterId2clusterEmb = np.memmap(cluster_path, dtype="float32", mode="w+", shape=(37000000,1024)) 
        clusterId2clusterEmb.flush()

        emb_f = np.memmap(os.path.join(args.save_path, "tokId_emb.dat"), dtype="float32", mode="readonly", shape=(37000000, 1024))
        tokGroupId2tokIdList = pickle.load(open(os.path.join(args.save_path, "tokGroupId2tokIdList.pickle"), 'rb'))
        tokId2tokText = pickle.load(open(os.path.join(args.save_path, "tokId2tokText.pickle"), 'rb'))
        tokId2tokGroupId = pickle.load(open(os.path.join(args.save_path, "tokId2tokGroupId.pickle"), 'rb'))
        assert tokId2tokGroupId[0] == 0

        tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId, cluster_path = do_light_cluster(model, tokGroupId2tokIdList, clusterId, c_tokGroupId, clusterId2clusterEmb, tokId2tokGroupId)

        #clusterIdList2corpusId = get_clusterIdList2corpusId(corpusId_tokenList_dict, tokId2clusterId)
        dump(f"light_tokGroupId2clusterIdList_{arg.cluster_num}.pickle", tokGroupId2clusterIdList)
        dump(f"light_clusterId2tokGroupId_{args.cluster_num}.pickle", clusterId2tokGroupId)
        dump(f"light_clusterId2tokText_{args.cluster_num}.pickle", clusterId2tokText)
        dump(f"light_tokText2clusterIdList_{args.cluster_num}.pickle", tokText2clusterIdList)
        dump(f"light_tokId2clusterId_{args.cluster_num}.pickle", tokId2clusterId)

    elif args.action == "cluster_dataset":
        print("Action: Dump Cluster Embedding!")
        train_dict = pickle.load(open(os.path.join(args.save_path, "gr_contextualized_train.pickle"), 'rb'))
        dev_dict = pickle.load(open(os.path.join(args.save_path, "gr_contextualized_dev.pickle"), 'rb'))
        test_dict = pickle.load(open(os.path.join(args.save_path, "gr_contextualized_test.pickle"), 'rb'))
        
        #clusterId2clusterEmb = np.memmap(cluster_path, dtype="float32", mode="readonly", shape=(37000000, 1024))
        tokId2clusterId = pickle.load(open(os.path.join(args.save_path, "tokId2clusterId.pickle")), "rb")

        train_dict, train_fname = construct_dataset(train_dict, tokId2clusterId, "train")
        dev_dict, dev_fname = construct_dataset(dev_dict, tokId2clusterId, "dev")
        test_dict, test_fname = construct_dataset(test_dict, tokId2clusterId, "test")
        dump(train_fname, train_dict)
        dump(dev_fname, dev_dict)
        dump(test_fname, test_dict)   

    else:
        assert False
    print("==== DONE ====")
