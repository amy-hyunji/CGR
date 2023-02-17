import time
import torch 
import faiss
import pickle
import numpy as np
from argparse import ArgumentParser
# import faiss.contrib.torch_utils

def get_quantizer(args, metric):
   dimension = args.dimension
   quantizer_threshold = args.quantizer_threshold
   if dimension > quantizer_threshold: #768,2048
      qtype = faiss.ScalarQuantizer.QT_8bit
      if args.gpu:
         quantizer = faiss.GpuIndexIVFScalarQuantizer(dimension, qtype, metric)
      else:
         quantizer = faiss.IndexScalarQuantizer(dimension, qtype, metric)
      return quantizer, True 
   else:
      if args.gpu:
         quantizer = faiss.GpuIndexFlatIP(dimension)
      else:
         quantizer = faiss.IndexFlatIP(dimension)
      return quantizer, False

def main(args):
   metric = faiss.METRIC_INNER_PRODUCT
   
   if args.load_name.endswith(".dat"):
      print("Load.. dat")
      dstore_keys = np.memmap(args.load_name, dtype="float32", shape=(args.memmap_size,args.dimension), mode="readonly")
   elif args.load_name.endswith(".pickle"):
      print("Load.. pickle")
      f = pickle.load(open(args.load_name, "rb"))
      dstore_keys = np.array([el for el in f]) 
   else:
      assert False

   if args.exhaustive_search:
      quantizer, requires_training = get_quantizer(args, metric)
      index = faiss.IndexIDMap(quantizer)
      to_train = index
   else:
      quantizer, _ = get_quantizer(args, metric)
      requires_training = True 
      if args.product_quantization:
         if args.gpu:
            print("using GpuIndexIVFPQ")
            index = faiss.GpuIndexIVFPQ(
                  quantizer, args.dimension, args.ncentroids, code_size, 8
               )
         else:
            print("using IndexIVFPQ")
            index = faiss.IndexIVFPQ(
                  quantizer, args.dimension, args.ncentroids, code_size, 8
               )
      else:
         if args.gpu:
            print("using GpuIndexIVFFlat")
            index = faiss.GpuIndexIVFFlat(
                  quantizer, args.dimension, args.ncentroids, metric
               )
         else:
            print("using IndexIVFFlat")
            index = faiss.IndexIVFFlat( ## 현재 사용되는 index 
                  quantizer, args.dimension, args.ncentroids, metric
               )

      index.nprobe = args.probe 
      to_train = index

   if requires_training:
      # print(dstore_keys[32001].shape)

      reduced_total_num_vals = 0
      zeros = np.zeros(1024,dtype=np.float32)
      
      for i in range(args.memmap_size):
         if np.array_equal(zeros, dstore_keys[i]):
            # print(i,dstore_keys[i])
            reduced_total_num_vals = i
            break
      print(reduced_total_num_vals)

      print("Start Training ...")
      start = time.time()
      # input = torch.from_numpy(dstore_keys.astype(np.float32))
      input = dstore_keys.astype(np.float32)

      to_train.train(input)
      print(f"Training took {time.time()-start} s")

   start = 0
   start_time = time.time()

      
   num_keys_to_add_at_a_time = 100 
   while start < reduced_total_num_vals:
      end = min(reduced_total_num_vals, start+num_keys_to_add_at_a_time)
      to_add = dstore_keys[start:end].copy()
      # items = torch.tensor(to_add.astype(np.float32))
      # ids = torch.arange(start, end, dtype=torch.int64)
      items = to_add.astype(np.float32)
      ids = np.arange(start,end,dtype=np.int64)
      index.add_with_ids(items, ids)

      start += num_keys_to_add_at_a_time

      if start % 1000000 == 0:
         print(f"Added {index.ntotal} tokens so far")
         #gc.collect() # 이거 뭐지? 

   print(f"Adding total {index.ntotal} keys")
   print(f"Adding took {time.time()-start_time} s")

   start = time.time()
   faiss.write_index(index, f"{args.index_name}")
   print(f"Writing index took {time.time()-start} s")


if __name__ == "__main__":
   parser = ArgumentParser()
   parser.add_argument("--load_name",default="/data/project/rw/contextualized_GENRE/dataset/bi.t5_large.hotpot.epoch8/clusterId_emb_5.dat",type=str)
   parser.add_argument('--exhaustive_search', default=False, action='store_true')
   parser.add_argument('--product_quantization', default=False, action='store_true')
   parser.add_argument('--gpu', default=False, action='store_true')
   parser.add_argument('--quantizer_threshold',default=2048,type=int)
   parser.add_argument("--dimension",default=1024,type=int)
   parser.add_argument('--total_num_vals',default=32000000,type=int)
   parser.add_argument('--probe',default=32,type=int)
   parser.add_argument('--ncentroids',default=4096,type=int)
   parser.add_argument('--memmap_size',default=37000000,type=int)


   args = parser.parse_args()
   args.index_name = f"/data/project/rw/contextualized_GENRE/dataset/bi.t5_large.hotpot.epoch8/{args.total_num_vals}_clusterId_emb.ES.{args.exhaustive_search}.PQ.{args.product_quantization}.GPU.{args.gpu}.{args.quantizer_threshold}"

   main(args)




