import os
import pickle
import argparse
import numpy as np
import pandas as pd 
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
from argparse import ArgumentParser
from collections import defaultdict
from sklearn.mixture import GaussianMixture

def construct_dataset(_dict, tokId2clusterId):
   ret_dict = {'input': [], 'output': [], 'output_tokid': []}
   for _input, _output, _output_tokid in zip(_dict['input'], _dict['output'], _dict['output_tokid']):
      ret_dict['input'].append(_input)
      ret_dict['output'].append(_output)
      ret_dict['output_tokid'].append([tokId2clusterId[el] for el in _output_tokid]) 
   return ret_dict

def construct_prefix_tree(corpus2clusterIdList):
   sys.setrecursionlimit(900000000)

   constrained_dict = {}
   for corpus, tokIdList in corpus2clusterIdList.items():
      cur_dict = constrained_dict # cur_dict[-2]: the node number
      tokIdList = [0] + tokIdList
      
      for i in range(len(tokIdList)-1):
         prev = tokIdList[i]
         cur = tokIdList[i+1]
         
         if i == len(tokIdList)-2:
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

def do_total_cluster():
   total_embs = list(tokId_embs.values())
   total_Id = list(tokId_embs.keys())
   if args.cluster_method == "k-means":
      df = pd.DataFrame(total_embs)
      print('+++ Do Clustering!!')
      model = KMeans(n_clusters=args.cluster_num, algorithm='auto')
      model.fit(df)
      print('+++ Done Clustering!!')
      predicts = np.array(model.predict(df))
      centers = model.cluster_centers_
   else:
      assert False

   tokId2clusterId = {}
   assert len(predicts) == len(total_Id)
   for predict, tokid in zip(predicts, total_Id):
      tokId2clusterId[tokid] = predict 
   
   # construct clusterId_tree
   corpus2clusterIdList = {}
   for tokId, corpus in tokId_corpus.items():
      if corpus in corpus2tokIdList.keys():
         corpus2tokIdList[corpus].append(tokId2clusterId[tokId])
      else:
         corpus2tokIdList[corpus] = [tokId2clusterId[tokId]]

   cluster_tree = construct_prefix_tree(corpus2clusterIdList)

   return tokId2clusterId, cluster_tree

def do_cluster():
   no_cluster = 0
   total_cluster = 0
   tokText_emb = {}
   for id, text in tokId_tokText.items():
       if text in tokText_emb.keys():
           tokText_emb[text].append([id, tokId_embs[id]])
       else:
           tokText_emb[text] = [[id, tokId_embs[id]]]
   print(f'keys in tokText_emb: {len(tokText_emb.keys())}')
   tokText2clusterIdList = defaultdict(list)
   tokId2clusterId = {}
   tokGroupId2clusterIdList = defaultdict(list)
   clusterId2tokGroupId = {}
   clusterId2tokText = {}
   clusterId2clusterEmb = {}
   clusterId = 0 # 0,1 is for <pad> and </s>
   for text, val in tqdm(tokText_emb.items()):
      #print(f'Working on ID: {id} and Text: {text}')
      #print(f'# of embeddings: {len(tokText_emb[text])}')
      id_list = [_val[0] for _val in val]
      emb_list = [_val[1] for _val in val]

      prev = False

      if len(val) > args.cluster_num:
         prev = True
         # reduce the number of embedings to cluster_num by kmeans 
         df = pd.DataFrame(emb_list)
         if args.cluster_method == "k-means":
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
         predicts = np.array([i for i in range(len(id_list))]) 
         centers = np.array(emb_list)

      #assert centers.shape[0] <= args.cluster_num 
      #assert len(predicts) == len(id_list) 


      clusterIdDict = {}
      for i, center in enumerate(centers):
         clusterIdDict[i] = clusterId 
         clusterId2tokText[clusterId] = text 
         clusterId2clusterEmb[clusterId] = center 
         clusterId += 1

      for _predict, _id in zip(predicts, id_list):
         _group = tokId_tokGroupId[_id]
         _clusterId = clusterIdDict[_predict]
         tokGroupId2clusterIdList[_group].append(_clusterId)
         clusterId2tokGroupId[_clusterId] = _group 
         tokId2clusterId[_id] = _clusterId 
         tokText2clusterIdList[text].append(_clusterId)
      
      if text == "<pad>" or text == "</s>":
         assert len(id_list) == 1 and len(tokText2clusterIdList[text]) == 1
      if clusterId == 1: assert tokId2clusterId[0] == 0
      if clusterId == 2: assert tokId2clusterId[1] == 1
   print(f'no_cluster: {no_cluster}\ttotal_cluster: {total_cluster}')

   return tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId, clusterId2clusterEmb

if __name__ == "__main__":
   parser = ArgumentParser()
   parser.add_argument('--cluster_num', default=15, type=int)
   parser.add_argument('--do_total', action='store_true')
   parser.add_argument('--cluster_method', default="k-means", type=str)
   parser.add_argument('--basedir', default=None, required=True, type=str)
   args = parser.parse_args()

   tokId_embs = pickle.load(open(os.path.join(args.basedir,'tokId_emb.pickle'), 'rb'))
   tokId_corpus = pickle.load(open(os.path.join(args.basedir,'tokId_corpus.pickle'), 'rb'))
   tokId_tokText = pickle.load(open(os.path.join(args.basedir, 'tokId_tokText.pickle'), 'rb'))
   tokId_tokGroupId = pickle.load(open(os.path.join(args.basedir, "tokId_tokGroupId.pickle"), 'rb'))
   tokGroupId_tokIdList = pickle.load(open(os.path.join(args.basedir, "tokGroupId_tokIdList.pickle"), "rb"))

   if "gr_contextualized_train.pickle" in os.listdir(args.basedir):
      train = pickle.load(open(os.path.join(args.basedir, "gr_contextualized_train.pickle"), 'rb'))
      dev = pickle.load(open(os.path.join(args.basedir, "gr_contextualized_dev.pickle"), 'rb'))
      test = pickle.load(open(os.path.join(args.basedir, "gr_contextualized_test.pickle"), 'rb'))
   elif "contextualized_train.pickle" in os.listdir(args.basedir):
      train = pickle.load(open(os.path.join(args.basedir, "contextualized_train.pickle"), 'rb'))
      dev = pickle.load(open(os.path.join(args.basedir, "contextualized_dev.pickle"), 'rb'))
      test = pickle.load(open(os.path.join(args.basedir, "contextualized_test.pickle"), 'rb'))
   else:
      assert False

   if args.do_total:
      tokId2clusterId, cluster_tree = do_total_cluster()
   else:
      tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId, clusterId2clusterEmb = do_cluster() 

   print('Done clustering!!')

   train = construct_dataset(train, tokId2clusterId)
   dev = construct_dataset(dev, tokId2clusterId)
   test = construct_dataset(test, tokId2clusterId)

   if args.do_total:
      with open(os.path.join(args.basedir, f"total_{args.cluster_method}_cluster_train_{args.cluster_num}.pickle"), 'wb') as f: 
         pickle.dump(train, f)
      with open(os.path.join(args.basedir, f"total_{args.cluster_method}_cluster_dev_{args.cluster_num}.pickle"), 'wb') as f: 
         pickle.dump(dev, f)
      with open(os.path.join(args.basedir, f"total_{args.cluster_method}_cluster_test_{args.cluster_num}.pickle"), 'wb') as f: 
         pickle.dump(test, f)
      with open(os.path.join(args.basedir, f"total_{args.cluster_method}_tokId_clusterId_{args.cluster_num}.pickle"), 'wb') as f:
         pickle.dump(tokId2clusterId, f)
      with open(os.path.join(args.basedir, f"total_cluster_tree.pickle"), 'wb') as f:
         pickle.dump(cluster_tree, f)

   else:
      with open(os.path.join(args.basedir, f"{args.cluster_method}_cluster_train_{args.cluster_num}.pickle"), 'wb') as f: 
         pickle.dump(train, f)
      with open(os.path.join(args.basedir, f"{args.cluster_method}_cluster_dev_{args.cluster_num}.pickle"), 'wb') as f: 
         pickle.dump(dev, f)
      with open(os.path.join(args.basedir, f"{args.cluster_method}_cluster_test_{args.cluster_num}.pickle"), 'wb') as f: 
         pickle.dump(test, f)
      with open(os.path.join(args.basedir, f"{args.cluster_method}_tokGroupId_clusterIdList_{args.cluster_num}.pickle"), 'wb') as f:
         pickle.dump(tokGroupId2clusterIdList, f)
      with open(os.path.join(args.basedir, f"{args.cluster_method}_clusterId_tokGroupId_{args.cluster_num}.pickle"), 'wb') as f:
         pickle.dump(clusterId2tokGroupId, f)
      with open(os.path.join(args.basedir, f"{args.cluster_method}_clusterId_tokText_{args.cluster_num}.pickle"), 'wb') as f:
         pickle.dump(clusterId2tokText, f)
      with open(os.path.join(args.basedir, f"{args.cluster_method}_tokText_clusterIdList_{args.cluster_num}.pickle"), 'wb') as f:
         pickle.dump(tokText2clusterIdList, f)
      with open(os.path.join(args.basedir, f"{args.cluster_method}_tokId_clusterId_{args.cluster_num}.pickle"), 'wb') as f:
         pickle.dump(tokId2clusterId, f)
      with open(os.path.join(args.basedir, f"{args.cluster_method}_clusterId_clusterEmb_{args.cluster_num}.pickle"), 'wb') as f:
         pickle.dump(clusterId2clusterEmb, f)
