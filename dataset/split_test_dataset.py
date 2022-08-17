import pickle
import math

dataset = "nq"

dev = pickle.load(open(f"{dataset}_dev.pickle", "rb"))
unique_dev = {"input": [], "output": [], "output_tokid": []}

for _input, _output, _output_tokid in zip(dev["input"], dev["output"], dev["output_tokid"]):
   if _input not in unique_dev["input"]:
      unique_dev["input"].append(_input)
      unique_dev["output"].append(_output)
      unique_dev["output_tokid"].append(_output_tokid)

unique_len = len(unique_dev["input"])
per_len = math.ceil(unique_len/10)

print(f"# of dev: {len(dev['input'])}")
print(f"# of unique dev: {unique_len}")

dump_len = 0
for i in range(10):
   sub_dev = {}
   sub_dev["input"] = unique_dev["input"][i*per_len:(i+1)*per_len] 
   sub_dev["output"] = unique_dev["output"][i*per_len:(i+1)*per_len] 
   sub_dev["output_tokid"] = unique_dev["output_tokid"][i*per_len:(i+1)*per_len] 
   dump_len += len(sub_dev["input"])
   with open(f"{dataset}_dev_sub{i}.pickle", "wb") as f:
      pickle.dump(sub_dev, f)

assert dump_len == unique_len
