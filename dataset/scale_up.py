import random
import pandas as pd 

#dataset = "nq"
dataset = "wow"
scale = 20 


total_corpus = list(pd.read_csv('kilt_total_corpus.csv')['corpus'])
if dataset == "nq":
   prev_corpus = list(pd.read_csv('kilt_nq/nq_toy_corpus.csv')['corpus'])
   output_path = f'kilt_nq/scale_{scale}_kilt_corpus.csv'
else:
   prev_corpus = list(pd.read_csv(f"kilt_{dataset}/scale_1_kilt_corpus.csv")["corpus"])
   output_path = f'kilt_{dataset}/scale_{scale}_kilt_corpus.csv'

prev_len = len(prev_corpus)

random.shuffle(total_corpus)
additional_corpus = []

idx = 0
while len(additional_corpus) < prev_len*(scale-1):
    sen = total_corpus[idx]
    idx += 1
    if idx % 100 == 0: print(f'=== {idx}/{prev_len*(scale-1)}')
    if sen in prev_corpus:
        continue
    else:
        additional_corpus.append(sen)

print(f'# of items in prev_corpus: {len(prev_corpus)} -> {len(set(prev_corpus))}')
print(f'# of items in additional corpus: {len(additional_corpus)} -> {len(set(additional_corpus))}')
print(f'# of items in new corpus: {len(prev_corpus+additional_corpus)} -> {len(set(prev_corpus+additional_corpus))}')
assert len(set(prev_corpus+additional_corpus)) == len(set(prev_corpus))*scale, f"new corpus: {len(set(prev_corpus+additional_corpus))} || prev_corpus: {len(set(prev_corpus))}"

df = pd.DataFrame({'corpus': prev_corpus+additional_corpus})
df.to_csv(output_path)


        
