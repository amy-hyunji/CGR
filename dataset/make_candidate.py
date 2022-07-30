import os
import pickle

from tqdm import tqdm


def dump(fname, file):
    with open(os.path.join(dataset, fname), "wb") as f:
        pickle.dump(file, f)

def _get_groupId_from_tokId(tokId):
        return tokId2groupId[tokId]

def _get_tokIdList_from_groupIdList(groupIdList):
    tokIdList = []
    for groupId in groupIdList:
        tokIdList.extend(groupId2tokId[groupId])
    return list(set(tokIdList))

def _get_from_trie(input_ids, trie_dict):
    if len(input_ids) == 0:
        possible_GroupList = list(trie_dict.keys())
        tokIdList = _get_tokIdList_from_groupIdList(possible_GroupList)
        return tokIdList
    else:
        curGroupId = _get_groupId_from_tokId(input_ids[0])
        if curGroupId in list(trie_dict.keys()):
            return _get_from_trie(input_ids[1:], trie_dict[curGroupId]) 
        else:
            return []

dataset = "t5-base-p1-5"
groupId_tree = "groupId_tree.pickle"
train_path = "gr_contextualized_train.pickle"
tokId_tokGroupId_path = "tokId_tokGroupId.pickle"
tokGroupId_tokIdList_path = "tokGroupId_tokIdList.pickle"

group_trie = pickle.load(open(os.path.join(dataset, groupId_tree), "rb"))
df_train = pickle.load(open(os.path.join(dataset, train_path), "rb"))
tokId2groupId = pickle.load(open(os.path.join(dataset, tokId_tokGroupId_path), "rb"))
groupId2tokId = pickle.load(open(os.path.join(dataset, tokGroupId_tokIdList_path), "rb"))

tokId2candidate = {}
for output_tokIds in tqdm(df_train["output_tokid"]):
    output_tokIds = [0] + output_tokIds

    for i in range(1,len(output_tokIds)):
        cur_tokId = output_tokIds[:i][-1]
        if cur_tokId in tokId2candidate.keys():
            continue
        candidate = _get_from_trie(output_tokIds[:i], group_trie)
        tokId2candidate[cur_tokId] = candidate

dump("tokId2candidate.pickle", tokId2candidate)