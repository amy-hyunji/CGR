import json
import pickle
import pandas as pd

##### Parameters to Change #####
corpus = list(pd.read_csv('./nq_toy_corpus.csv')['corpus'])
corpusId_emb = pickle.load(open('corpusId_emb.pickle', 'rb'))
df = pd.read_csv('./nq_toy_dev.csv')
save_file ="contextualized_nq_toy_dev.pickle"
############################################

save_dict = {'input': [], 'output': [], 'output_tokid': [], 'output_tokemb': []}
for _input, _output in zip(df['input'], df['output']):
   corpus_id = corpus.index(_output)
   emb_dict = corpusId_emb[corpus_id]
   output_tok = list(emb_dict.keys())
   output_emb = list(emb_dict.values())
   save_dict['input'].append(_input)
   save_dict['output'].append(_output)
   save_dict['output_tokid'].append(output_tok)
   save_dict['output_tokemb'].append(output_emb)

with open(save_file, "wb") as f:
    pickle.dump(save_dict, f)
