import pandas as pd
import jsonlines
import csv
import sys 

data = "trec-covid"
dev_path = f"{data}/qrels/test.tsv"
corpus_path = f"{data}/corpus.jsonl"
queries_path = f"{data}/queries.jsonl"

corpus = {}
with jsonlines.open(corpus_path) as f:
   for line in f.iter():
      corpus[line["_id"]] = line["title"]

queries = {}
with jsonlines.open(queries_path) as f:
   for line in f.iter():
      queries[int(line["_id"])] = line["text"]

dev_df = {"input": [], "output": []}
dev = pd.read_csv(dev_path, delimiter="\t", keep_default_na=False)
for i, (index, row) in enumerate(dev.iterrows()):
   corpus_id = row["corpus-id"]
   query_id = row["query-id"]
   _corpus = corpus[corpus_id] 
   _query = queries[query_id]
   dev_df["input"].append(_query)
   dev_df["output"].append(_corpus)

dev_df = pd.DataFrame(dev_df)
dev_df.to_csv(f"{data}/test.csv")
