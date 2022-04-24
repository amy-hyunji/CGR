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

def insert_tokId(nodeId, tokId):
    if tokId == -1:
        return
    if nodeId not in nodeId_tokIdList.keys():
        nodeId_tokIdList[nodeId] = [tokId]
    else:
        nodeId_tokIdList[nodeId].append(tokId)

### Parameters to change ###
corpusId_emb = pickle.load(open("./dataset/kilt_nq/corpusId_emb.pickle", "rb"))
tokId_tokGroupId = pickle.load(open("./dataset/kilt_nq/tokId_tokGroupId.pickle", "rb"))
corpusId_tokGroupList_path = "corpusId_tokGroupList_fixed.pickle"
output_path = "./dataset/kilt_nq/nq_toy_prefix_tree_fixed.pickle"
nodeId_tokIdList_path = "./dataset/kilt_nq/nodeId_tokIdList.pickle"
############################

sys.setrecursionlimit(900000000)

node_id = 1
nodeId_tokIdList = {}
# ex)
# constrained_dict[-2]: dummy node id
# constrained_dict[-1][-2]: root node id, nodeId_tokIdList[constrained_dict[-1][-2]] contains tokIds of the child of root
# constrained_dict[-1][266][-2]: nodeId_tokIdList[constrained_dict[-1][266][-2]] contains tokIds of the child of 266(=groupId)


constrained_dict = {-2:0} # save node_id with key -2, from 0 to total node num
corpusId_tokGroupList = {}
for corpusId, corpusDict in corpusId_emb.items():
    cur_dict = constrained_dict # cur_dict[-2]: the node number
    tokIdList = list(corpusDict.keys())
    tokGroupIdList = [tokId_tokGroupId[el] for el in tokIdList]
    tokGroupIdList = [-1] + tokGroupIdList + [1]
    corpusId_tokGroupList[corpusId] = tokGroupIdList
    
    tokIdList = [-1] + tokIdList + [1]
    assert len(tokIdList) == len(tokGroupIdList)
    for i in range(len(tokGroupIdList)-1):
        prev = tokGroupIdList[i]
        cur = tokGroupIdList[i+1]
        
        prev_tokId = tokIdList[i]
        cur_tokId = tokIdList[i+1]
        
        insert_tokId(cur_dict[-2], prev_tokId)
        if i == len(tokGroupIdList)-2:
            if prev in cur_dict.keys():
                if cur not in cur_dict[prev].keys():
                    insert_tokId(cur_dict[prev][-2], cur_tokId)
                    cur_dict[prev][cur] = {-2:node_id}
                    node_id += 1
            else:
                cur_dict[prev] = {-2:node_id}
                node_id += 1
                insert_tokId(cur_dict[prev][-2], cur_tokId)
                cur_dict[prev][cur] = {-2:node_id}
                node_id += 1
        else:
            if prev in cur_dict.keys():
                pass
            else:
                cur_dict[prev] = {-2:node_id}
                node_id += 1
            cur_dict = cur_dict[prev]

with open(output_path, "wb") as f:
    pickle.dump(constrained_dict, f)
print(f'Saved tree in {output_path}')

with open(corpusId_tokGroupList_path, "wb") as f:
    pickle.dump(corpusId_tokGroupList, f)

with open(nodeId_tokIdList_path, "wb") as f:
    pickle.dump(nodeId_tokIdList, f)
