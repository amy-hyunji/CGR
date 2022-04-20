import copy
import pickle
import torch
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration

_config = T5Config.from_pretrained("t5-base")
_config.update({"contextualized_emb_num": 243245})
_config.update({"contextualized_file": "./dataset/kilt_nq/tokId_emb.pickle"})
print("="*50)
print("Loading Tokenizer")
tok = T5Tokenizer.from_pretrained("t5-base")
_dict = pickle.load(open('./dataset/kilt_nq/tokId_emb.pickle', 'rb'))
inputs = tok.batch_encode_plus(["I like to go", "I hate going"], max_length=5, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
#_labels = tok.batch_encode_plus(["to school! I like school", "to work. I hate to work"], max_length=10, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
print(f"input: {inputs.shape}")
# [0, .., contextualized_emb_num-1]
_labels = [[3, 2, 1, 1, 10, 15], [2, 1, 5, 2, 8, 10]]
decoder_input_embed = []
for elem in _labels:
   el_list = []
   for el in elem:
      el_list.append(_dict[el])
   decoder_input_embed.append(el_list)
decoder_input_embed = torch.tensor(decoder_input_embed)

_labels = torch.tensor(_labels)
labels = copy.deepcopy(_labels)
labels[labels[:, :]==tok.pad_token_id] = -100

print(f"output: {labels.shape}")
print(f"labels: {labels}")
print("="*50)
print(f"Loading Model!")
model = T5ForConditionalGeneration.from_pretrained("t5-base", config=_config)
print("="*50)
outputs = model(inputs, decoder_input_ids=None, labels=labels, decoder_inputs_embeds=decoder_input_embed)
print(outputs.loss)
