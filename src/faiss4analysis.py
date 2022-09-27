import os
import sys
import faiss
import torch
import pickle
import numpy as np
from tqdm import tqm
from collections import defaultdict 
from transformers import T5Model, T5Tokenizer 

d = 1024
k = 5
scale = 100

clusterId_emb_path = "../dataset/pipeline_bi.total.epoch47/pipeline_bi.total.epoch47/clusterId_emb_5.pickle"
clusterId_emb = pickle.load(open(clusterId_emb_path, "rb"))
print(f"Success on opening clusterId_emb in {clusterId_emb_path}")

#clusterId2tokText = pickle.load(open("../dataset/pipeline_bi.total.epoch47/pipeline_bi.total.epoch47/clusterId2tokGroupId_5.pickle", "rb"))

xb = np.array(list(clusterId_emb.values())).astype("float32")

index = faiss.IndexFlatIP(d)
index.add(xb)
print("Done adding index!")

xq = xb[:100]

D, I = index.search(xq, k)

save_dict = defaultdict(list)

clusterId = 0
for ind in I:
   save_dict[clusterId] = ind 
   if clusterId == 0:
      print(f"ind of {clusterId} : {ind}")
   clusterId += 1

