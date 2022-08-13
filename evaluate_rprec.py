# input, gt, gt_tok, pred, pred_tok, em, recall

import json
import argparse
import jsonlines
import pandas as pd

from collections import defaultdict

def get_unique_text(pred_text, corpus):
    for elem in corpus:
        if elem.startswith(pred_text)
            return elem
    return pred_text

parser = argparse.ArgumentParser()
parser.add_argument("--pred", type=str, required=True)
parser.add_argument("--gt", type=str, required=True)
parser.add_argument("--corpus", type=str, required=True, default="./dataset/n_kilt_total_corpus.csv")
args = parser.parse_args()

corpus = pd.read_csv(args.corpus)
corpus = corpus.fillna("")
corpus = list(corpus["corpus"]) 

#pred_path = "./outputs/gr-nq_toy_bi_p1-5_full_ckpt/groupId_mbs_False_result_beam5.json"
pred_file = json.load(open(args.pred))
pred_base = "/".join(pred_path.split("/")[:-1])
gt_file = args.gt  #"dataset/total_nq/nq-dev-kilt.jsonl"

print(f"pred file: {pred_path}")
print(f"gt file: {gt_file}")

input2pred = defaultdict(list) 
for _input, _gt, _pred_list, _pred_tok_list in zip(pred_file["input"], pred_file["gt"], pred_file["pred"], pred_file["pred_tok"]):
    for _pred, _pred_tok in zip(_pred_list, _pred_tok_list):
        if 1 not in _pred_tok and _pred in _gt:
            _pred = get_unique_text(_pred)
        input2pred[_input].append(_pred)

pred_list = []

with jsonlines.open(gt_file) as f:
    for i, line in enumerate(f.iter()):
        _id = line["id"]
        _input = line["input"]
        assert _input in input2pred.keys(), f"Missing .. {_input}"
        _pred_list = input2pred[_input]
        _provenance_list = []
        for pred in _pred_list:
            _provenance_list.append({"title": pred})
        pred_list.append({"id": _id, "input": _input, "output": [{"provenance": _provenance_list}]})

with open(os.path.join(pred_base, "pred.jsonl"), mode="w") as f:
    for i in pred_list:
        f.write(json.dumps(i) + "\n")
    
print(f"Done saving in ... {os.path.join(pred_base, 'pred.jsonl')}")
