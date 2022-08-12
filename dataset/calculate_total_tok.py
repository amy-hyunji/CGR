import pandas as pd

from transformers import T5Tokenizer, BartTokenizer
from tqdm import tqdm

###########################
model = "t5-large"
corpus_path = "n_kilt_total_corpus.w_para.sub/idx_13.csv"
###########################
print(f"corpus_path: {corpus_path}")

if "t5" in model:
   tok = T5Tokenizer.from_pretrained(model)
elif "bart" in model:
   tok = BartTokenizer.from_pretrained(model)
else:
   assert False 

total_len = 0
df = pd.read_csv(corpus_path)
df = df.fillna("")
corpus = list(df["corpus"])
for i in tqdm(range(len(corpus))):
   tok_ret = tok(corpus[i], add_special_tokens=False)["input_ids"]
   total_len += len(tok_ret) 
   if i % 100000 == 0: print(f"[{i}]: {total_len}")

print(f"## total number of tokens: {total_len}")
