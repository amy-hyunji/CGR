import pickle
import numpy as np  

cluster_num = 117508

dat_path = "./total_nq/t5_large.nq_full/clusterId_emb_5.dat"
pickle_path = "./total_nq/t5_large.nq_full/clusterId_emb_5.pickle"

dat_f = np.memmap(dat_path, shape=(37000000,1024), mode='readonly', dtype="float32")

clusterId2Emb = {}
for i in range(cluster_num):
    clusterId2Emb[i] = dat_f[i][:]

with open(pickle_path, "wb") as f:
    pickle.dump(clusterId2Emb, f)