import os
import sys
import faiss
import torch
import pickle

import numpy as np

from tqdm import tqdm
from collections import defaultdict
from transformers import T5Model, T5Tokenizer

def _get_embedding(model, batch):
    query_input_ids = batch['input_ids'].to(model.device)
    query_attention_mask = batch['attention_mask'].to(model.device)
    if query_input_ids.shape[0] == 1:
        d_input = torch.tensor([[tok.pad_token_id]]).to(model.device)
    else:
        d_input = torch.cat([tok.pad_token_id]*(query_input_ids.shape[0]), 0).to(model.device)

    query_outputs = model(
        input_ids=query_input_ids,
        attention_mask=query_attention_mask,
        decoder_input_ids=d_input,
    )
    query_output = query_outputs.last_hidden_state[:, 0].squeeze()
    return query_output

d = 768 # dim of vector
k = 5 # number of cluster
scale = 100

filepath = f"../dataset/kilt_nq/bi-scale{scale}-nq_toy_p1-5"
tokId_emb = pickle.load(open(os.path.join(filepath, "tokId_emb.pickle"), "rb"))
print(f'Success on opening tokId_emb!')

xb = np.array(list(tokId_emb.values())).astype('float32')
print(f'Shape of database: {xb.shape}') #[database, 768]

index = faiss.IndexFlatIP(d)
index.add(xb)

print(f'Done adding index!')


testpath = os.path.join(filepath, "bi_contextualized_first_only_False_test.pickle") 
testfile = pickle.load(open(testpath, "rb"))
query_list = list(set(testfile["input"])) 


modelpath = "../outputs/bi_nq_toy_p1-5/best_tfmr_143"
model = T5Model.from_pretrained(modelpath).cuda()
tok = T5Tokenizer.from_pretrained(modelpath)

q_list = []
for query in tqdm(query_list):
    query = tok(query, return_tensors="pt") 
    q_list.append(_get_embedding(model, query).detach().cpu().numpy().astype('float32'))
xq = np.array(q_list)
print(f"Shape of query: {xq.shape}") #[10000,768]


"""
D -> query와 가까운 애의 거리
I -> query와 가까운 애의 index
"""
D, I = index.search(xq, k) 


# get answer_dict
answer_dict = defaultdict(list)

for _input, _output_tokid in zip(testfile['input'], testfile['output_tokid']):
    assert len(_output_tokid) == 1
    answer_dict[_input].append(_output_tokid[0])


save_dict = {'input': [], 'answer': [], 'predict': [], 'em': [], 'recall': []}
assert len(I) == len(query_list)
for ind, q in zip(I, query_list):
    save_dict['input'].append(ind)
    save_dict['predict'].append(ind)
    _ans = answer_dict[q]
    save_dict['answer'].append(_ans)
    if ind[0] in _ans:
        save_dict['em'].append(100)
    else:
        save_dict['em'].append(0)
    
    s_ind = set(ind); s_ans = set(_ans)
    if len(s_ind.intersection(s_ans)) > 0:
        save_dict['recall'].append(100)
    else:
        save_dict['recall'].append(0)

with open(os.path.join(modelpath, f"{scale}_result.pickle"), 'wb') as f:
    pickle.dump(save_dict, f)

print(f'DONE saving in {os.path.join(modelpath, f"{scale}_result.pickle")}')
print(f'EM: {np.array(save_dict["em"]).mean()}')
print(f'RECALL: {np.array(save_dict["recall"]).mean()}')
