import os
import json
import pickle
import pandas as pd

##### Parameters to Change #####
corpus = list(pd.read_csv('./nq_toy_corpus.csv')['corpus'])
save_path = "t5-base-emb"
corpusId_emb = pickle.load(open(os.path.join(save_path, 'corpusId_emb.pickle'), 'rb'))
split = "test"
############################################

df = pd.read_csv(f'./nq_toy_{split}.csv')
save_file = os.path.join(save_path, f"contextualized_{split}.pickle")

save_dict = {'input': [], 'output': [], 'output_tokid': []}
for _input, _output in zip(df['input'], df['output']):
   corpus_id = corpus.index(_output)
   emb_dict = corpusId_emb[corpus_id]
   output_tok = list(emb_dict.keys())
   output_emb = list(emb_dict.values())

   assert output_tok[-1] == 1
   for _tok in output_tok[:-1]:
      save_dict['input'].append(_input)
      save_dict['output'].append(_output)
      save_dict['output_tokid'].append([_tok])

with open(save_file, "wb") as f:
    pickle.dump(save_dict, f)
