import json
import pickle
from collections import defaultdict 

dataset = "fever"

unique_dev = {"input": [], "output": [], "output_tokid": []}
input2output = defaultdict(list) 
prev_input = []

dev_file = f"./total_nq/t5_large.nq_full/{dataset}/gr_{dataset}_dev_cluster_5.pickle"

dev_file = pickle.load(open(dev_file, "rb"))

for _input, _output, _output_tokid in zip(dev_file["input"], dev_file["output"], dev_file["output_tokid"]):
   input2output[_input].append(_output)
   if _input in prev_input:
      continue
   else:
      unique_dev["input"].append(_input) 
      unique_dev["output"].append(_output)
      unique_dev["output_tokid"].append(_output_tokid)

with open(f"total_nq/t5_large.nq_full/{dataset}/unique.gr_{dataset}_dev_cluster_5.pickle", "wb") as f:
   pickle.dump(unique_dev, f)

with open(f"total_nq/t5_large.nq_full/{dataset}/dev_input2output.json", "w") as f:
   json.dump(input2output, f)
