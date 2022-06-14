import os
import pickle
import argparse
import numpy as np
import pandas as pd 
from tqdm import tqdm
from sklearn.cluster import KMeans
from argparse import ArgumentParser
from collections import defaultdict

def construct_dataset(_dict, tokId2clusterId):
   ret_dict = {'input': [], 'output': [], 'output_tokid': []}
   for _input, _output, _output_tokid in zip(_dict['input'], _dict['output'], _dict['output_tokid']):
      ret_dict['input'].append(_input)
      ret_dict['output'].append(_output)
      ret_dict['output_tokid'].append([tokId2clusterId[el] for el in _output_tokid]) 
   return ret_dict

def do_cluster():
   tokText_emb = {}
   for id, text in tokId_tokText.items():
       if text in tokText_emb.keys():
           tokText_emb[text].append([id, tokId_emb[id]])
       else:
           tokText_emb[text] = [[id, tokId_emb[id]]]
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

       if len(val) > args.cluster_num:
           # reduce the number of embedings to cluster_num by kmeans 
           df = pd.DataFrame(emb_list)
           model = KMeans(n_clusters=15, algorithm='auto')
           model.fit(df)
           predicts = np.array(model.predict(df))
           centers = model.cluster_centers_ # [15, 768]
       else:
           # use the embedding 
           predicts = np.array([i for i in range(len(id_list))]) 
           centers = np.array(emb_list)

       assert centers.shape[0] <= 15
       assert len(predicts) == len(id_list) 

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

   return tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId, clusterId2clusterEmb

if __name__ == "__main__":
   parser = ArgumentParser()
   parser.add_argument('--cluster_num', default=15, type=int)
   parser.add_argument('--basedir', default=None, required=True, type=str)
   args = parser.parse_args()

   tokId_emb = pickle.load(open(os.path.join(args.basedir,'tokId_emb.pickle'), 'rb'))
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

   tokGroupId2clusterIdList, clusterId2tokGroupId, clusterId2tokText, tokText2clusterIdList, tokId2clusterId, clusterId2clusterEmb = do_cluster() 
   print('Done clustering!!')

   train = construct_dataset(train, tokId2clusterId)
   dev = construct_dataset(dev, tokId2clusterId)
   test = construct_dataset(test, tokId2clusterId)

   with open(os.path.join(args.basedir, f"cluster_train_{args.cluster_num}.pickle"), 'wb') as f: 
      pickle.dump(train, f)
   with open(os.path.join(args.basedir, f"cluster_dev_{args.cluster_num}.pickle"), 'wb') as f: 
      pickle.dump(dev, f)
   with open(os.path.join(args.basedir, f"cluster_test_{args.cluster_num}.pickle"), 'wb') as f: 
      pickle.dump(test, f)

   with open(os.path.join(args.basedir, f"tokGroupId_clusterIdList_{args.cluster_num}.pickle"), 'wb') as f:
      pickle.dump(tokGroupId2clusterIdList, f)
   with open(os.path.join(args.basedir, f"clusterId_tokGroupId_{args.cluster_num}.pickle"), 'wb') as f:
      pickle.dump(clusterId2tokGroupId, f)
   with open(os.path.join(args.basedir, f"clusterId_tokText_{args.cluster_num}.pickle"), 'wb') as f:
      pickle.dump(clusterId2tokText, f)
   with open(os.path.join(args.basedir, f"tokText_clusterIdList_{args.cluster_num}.pickle"), 'wb') as f:
       pickle.dump(tokText2clusterIdList, f)
   with open(os.path.join(args.basedir, f"tokId_clusterId_{args.cluster_num}.pickle"), 'wb') as f:
       pickle.dump(tokId2clusterId, f)
   with open(os.path.join(args.basedir, f"clusterId_clusterEmb_{args.cluster_num}.pickle"), 'wb') as f:
       pickle.dump(clusterId2clusterEmb, f)
