import sys
import pickle 
import pandas as pd
import re 
import string
import json

from tqdm import tqdm
from transformers import T5Tokenizer, BartTokenizer

def lmap(f, x):
    return list(map(f, x))

def ids_to_text(generated_ids, tokenizer):
    gen_text = tokenizer.batch_decode([generated_ids], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return lmap(str.strip, gen_text)

### Parameters to change ###
corpusId_emb = pickle.load(open("./dataset/kilt_nq/corpusId_emb.pickle", "rb"))
tokId_tokGroupId = pickle.load(open("./dataset/kilt_nq/tokId_tokGroupId.pickle", "rb"))
output_path = "./dataset/kilt_nq/nq_toy_prefix_tree.pickle"
############################

sys.setrecursionlimit(900000000)

constrained_dict = {}
corpusId_tokGroupList = {}
for corpusId, corpusDict in corpusId_emb.items():
    cur_dict = constrained_dict
    tokIdList = list(corpusDict.keys())
    tokGroupIdList = [tokId_tokGroupId[el] for el in tokIdList]
    tokGroupIdList = [-1] + tokGroupIdList + [1]
    corpusId_tokGroupList[corpusId] = tokGroupIdList
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
            if prev in cur_dict.keys(): pass
            else: cur_dict[prev] = {}
            cur_dict = cur_dict[prev]

with open(output_path, "wb") as f:
    pickle.dump(constrained_dict, f)
print(f'Saved tree in {output_path}')

with open("corpusId_tokGroupList.pickle", "wb") as f:
    pickle.dump(corpusId_tokGroupList, f)