import pandas as pd
import pickle

df = pd.read_csv("nfcorpus_dump/test.csv")

_pickle = {"input": [], "output": [], "output_tokid": []}

for _input, _output in zip(df["input"], df["output"]):
    _pickle["input"].append(_input)
    _pickle["output"].append(_output)
    _pickle["output_tokid"].append([0])


with open("nfcorpus_dump/test.pickle", "wb") as f:
    pickle.dump(_pickle, f)
