import os
import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModel, T5EncoderModel, T5Tokenizer
from tqdm import tqdm

#model = AutoModel.from_pretrained("bert-base-uncased").cuda()
#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = T5EncoderModel.from_pretrained('t5-base').cuda()
tokenizer = T5Tokenizer.from_pretrained('t5-base')

corpus_file = pd.read_csv("/mnt/entailment/toy_GENRE/dataset/kilt_nq/nq_toy_corpus.csv")
corpus_num = len(corpus_file["corpus"])
corpusId_corpus_dict = {} # {corpusId: corpus}
corpusId_emb_dict= {} # {corpusId: {tok: {emb}}}
tokId_corpus = {} # {tokid: [corpusId, tok_pos]}
tokId_emb = {} # {tokid: emb}

# tokId = 0 -> <pad> token 
_tok = tokenizer("<pad>", return_tensors='pt', add_special_tokens=False)
_input_ids = _tok['input_ids'].cuda()
_attention_mask = _tok['attention_mask'].cuda()
model_ret = model(input_ids=_input_ids, attention_mask=_attention_mask, return_dict=True)
last_hidden_state = model_ret['last_hidden_state'][0]
last_hidden_state = last_hidden_state.detach().cpu().numpy()
_input_ids = _input_ids.detach().cpu().numpy()
assert len(_input_ids[0] == len(last_hidden_state))
assert _input_ids[0][0] == 0
tokId_emb[0] = last_hidden_state[0]


total_tok_num = 0
tokId = 1

for corpusId in tqdm(range(corpus_num)):
   elem = corpus_file["corpus"][corpusId]
   _tok = tokenizer(elem, return_tensors="pt")
   
   _input_ids = _tok["input_ids"].cuda()
   _attention_mask = _tok["attention_mask"].cuda()

   model_ret = model(input_ids=_input_ids, attention_mask=_attention_mask, return_dict=True)
   last_hidden_state = model_ret['last_hidden_state'][0] #  [seq_len, 768]
   last_hidden_state = last_hidden_state.detach().cpu().numpy()
   _input_ids = _input_ids.detach().cpu().numpy()

   _tok_dict = {}
   assert len(_input_ids[0])==len(last_hidden_state)
   total_tok_num += len(last_hidden_state)

   for tok_pos, (_ids, _emb) in enumerate(zip(_input_ids[0], last_hidden_state)):
      _tok_dict[tokId] = _emb
      tokId_corpus[tokId] = [corpusId, tok_pos]
      tokId_emb[tokId] = _emb
      tokId += 1

   corpusId_corpus_dict[corpusId] = elem
   corpusId_emb_dict[corpusId] = _tok_dict 

with open("corpusId_corpus.pickle", "wb") as f:
   pickle.dump(corpusId_corpus_dict, f)

with open("corpusId_emb.pickle", "wb") as f:
   pickle.dump(corpusId_emb_dict, f)

with open('tokId_emb.pickle', 'wb') as f:
   pickle.dump(tokId_emb, f)

with open('tokId_corpus.pickle', 'wb') as f:
   pickle.dump(tokId_corpus, f)

print(f'total_tok_num: {total_tok_num}')
print(f'tokId: {tokId}')
print("DONE!!")
