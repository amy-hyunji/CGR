import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd 
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
from argparse import ArgumentParser
from collections import defaultdict
from sklearn.mixture import GaussianMixture


def construct_count_subtree():
    # corpus2clusterIdList
    corpus2clusterIdList = {}
    for tokId, corpus in tokId_corpus.items():
        if corpus in corpus2clusterIdList.keys():
            corpus2clusterIdList[corpus].append(tokId2clusterId[tokId])
        else:
            corpus2clusterIdList[corpus] = [tokId2clusterId[tokId]]
    
    # construct cluster_tree and count_subtree
    sys.setrecursionlimit(900000000)

    count_subtree_dict = {}
    constrained_dict = {}

    for corpus, tokIdList in corpus2clusterIdList.items():
        cur_dict = constrained_dict
        cur_subtree_dict = count_subtree_dict

        tokIdList = [0] + tokIdList + [1]
        tokGroupIdList = [tokId_tokGroupId[el] for el in tokIdList]
        #tokIdList = [0] + tokGroupIdList
        tokIdList = tokGroupIdList
        print("tokIdList")
        print(tokIdList)

        for i in range(len(tokIdList)-1):
            prev = tokIdList[i]
            cur = tokIdList[i+1]

            if i == len(tokIdList)-2:
                if prev in cur_dict.keys():
                    cur_subtree_dict[prev][-1] += 1
                    if cur not in cur_dict[prev].keys():
                        cur_dict[prev][cur] = {}
                        cur_subtree_dict[prev][cur] = {-1:1}
                else:
                    cur_dict[prev] = {cur: {}}
                    cur_subtree_dict[prev] = {-1:1, cur:{-1:1}}
            else:
                if prev in cur_dict.keys():
                    cur_subtree_dict[prev][-1] += 1
                    pass
                else:
                    cur_dict[prev] = {}
                    cur_subtree_dict[prev] = {-1:1}
                cur_dict = cur_dict[prev]
                cur_subtree_dict = cur_subtree_dict[prev]

    return constrained_dict, count_subtree_dict



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--cluster_num', default=5, type=int)
    parser.add_argument('--basedir', default=None, required=True, type=str)
    parser.add_argument('--cluster_method', default="k-means", type=str)
    args = parser.parse_args()


    tokId_corpus = pickle.load(open(os.path.join(args.basedir, f'tokId_corpus.pickle'), 'rb'))
    tokId2clusterId = pickle.load(open(os.path.join(args.basedir, f'{args.cluster_method}_tokId_clusterId_{args.cluster_num}.pickle'), 'rb'))
    tokId_tokGroupId = pickle.load(open(os.path.join(args.basedir, f"{args.cluster_method}_clusterId_tokGroupId_{args.cluster_num}.pickle"), 'rb'))


    cur_dir = "/mnt/nfs/jaeyoung/contextualized_GENRE/dataset/kilt_nq"
    cluster_tree, count_subtree = construct_count_subtree()
    with open(os.path.join(cur_dir, f"{args.cluster_method}_clusterId_count_subtree_{args.cluster_num}.pickle"), 'wb') as f:
        pickle.dump(count_subtree, f)
    with open(os.path.join(cur_dir, f"{args.cluster_method}_clusterId_cluster_tree_{args.cluster_num}.pickle"), 'wb') as f:
        pickle.dump(cluster_tree, f)