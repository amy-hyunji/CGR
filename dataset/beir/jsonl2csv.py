import jsonlines
import pandas as pd

data = "scidocs"
jsonl_path = f"{data}/corpus.jsonl"
save_path = f"{data}/corpus.csv"
df = {"corpus": [], "context": []}

with jsonlines.open(jsonl_path) as f:
   for i, line in enumerate(f.iter()):
      df["corpus"].append(line["title"])
      df["context"].append(line["text"])

df = pd.DataFrame(df)
df.to_csv(save_path)
