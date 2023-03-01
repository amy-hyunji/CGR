import os
import pickle
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
"""
save_path = "wikitext2.train.t5-base"
dim=768
token_cnt = 2753500
"""

"""
save_path = "nq.answer.t5-base"
dim=768
token_cnt = 358100 
"""


def dat2pik(args):
    save_path = args.save_path
    dim = args.dim
    cluster = args.cluster
    token_cnt = args.token_cnt
    memmap_size = args.memmap_size
    if args.token:
       if args.w_vd:
          assert False
       elif args.only_vd:
          assert False
       else:
          total_f = np.memmap(os.path.join(save_path, f"tokId_emb.w_para.dat"), dtype="float32", mode="readonly", shape=(memmap_size, dim))
    else:
       if args.w_vd:
          total_f = np.memmap(os.path.join(save_path, f"w_vd.clusterId_emb_{cluster}.dat"), dtype="float32", mode="readonly", shape=(memmap_size, dim))
       elif args.only_vd:
          total_f = np.memmap(os.path.join(save_path, f"only_vd.clusterId_emb_{cluster}.dat"), dtype="float32", mode="readonly", shape=(memmap_size, dim))
       else:
          total_f = np.memmap(os.path.join(save_path, f"clusterId_emb_{cluster}.dat"), dtype="float32", mode="readonly", shape=(memmap_size, dim))
    #total_f = np.memmap(save_path, dtype="float32", mode="readonly", shape=(memmap_size, dim))

    total_dict = {}

    # check if token_cnt is correct
    if total_f[token_cnt].tolist().count(0) != dim:
        assert False, f"Check token_cnt {total_f.shape}"

    for i in tqdm(range(token_cnt)):
        # tens = total_f[i]
        total_dict[i] = total_f[i]

    if args.token:
       if args.w_vd:
           assert False
       elif args.only_vd:
           assert False
       else:
           with open(os.path.join(save_path, "tokId_emb.pickle"), "wb") as f:
              pickle.dump(total_dict, f)
    else:
       if args.w_vd:
           with open(os.path.join(save_path, f"w_vd.clusterId_emb_{cluster}.pickle"), "wb") as f:
              pickle.dump(total_dict, f)
       elif args.only_vd:
           with open(os.path.join(save_path, f"only_vd.clusterId_emb_{cluster}.pickle"), "wb") as f:
              pickle.dump(total_dict, f)
       else:
           with open(os.path.join(save_path, f"clusterId_emb_{cluster}.pickle"), "wb") as f:
              pickle.dump(total_dict, f)
    
#    with open(f"/data/project/rw/contextualized_GENRE/dataset/bi.t5_large.hotpot.epoch8/{args.token_cnt}_dummy.pickle", "wb") as f:
#        pickle.dump(total_dict, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--dim", default=1024, type=int)
    parser.add_argument("--cluster", default=5, type=int)
    parser.add_argument("--token_cnt", required=True, type=int) #
    parser.add_argument('--memmap_size',default=37000000,type=int)
    parser.add_argument('--token', action="store_true")
    parser.add_argument('--w_vd', action="store_true")
    parser.add_argument('--only_vd', action="store_true")
    args = parser.parse_args()
## 615399
    dat2pik(args)
