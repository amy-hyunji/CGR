import os
import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModel, T5EncoderModel, T5Tokenizer
from tqdm import tqdm

#model = AutoModel.from_pretrained("bert-base-uncased").cuda()
#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = T5EncoderModel.from_pretrained('t5-base').cuda()
tokenizer = T5Tokenizer.from_pretrained('t5-base')

### Change to corpus file you want to use ###
corpus_file = pd.read_csv("./nq_toy_corpus.csv") 
#############################################


corpus_num = len(corpus_file["corpus"])
corpusId_corpus_dict = {} # {corpusId: corpus}
corpusId_emb_dict= {} # {corpusId: {tok: {emb}}}
tokId_corpus = {} # {tokid: [corpusId, tok_pos]}
tokId_emb = {} # {tokid: emb}
tok_Idlist_dict = {} # {tok_text: [Idlist of the tok]}
tok_Id_dict = {} # {Id: tok_text}

# tokId = 0 -> <pad> token 
_tok = tokenizer("<pad>", return_tensors='pt', add_special_tokens=False)
_input_ids = _tok['input_ids'].cuda()
_tok_decode = tokenizer.convert_ids_to_tokens(_input_ids[0])
assert len(_tok_decode) == 1
tok_Idlist_dict[_tok_decode[0]] = [0]
tok_Id_dict[0] = _tok_decode[0] 
_attention_mask = _tok['attention_mask'].cuda()
model_ret = model(input_ids=_input_ids, attention_mask=_attention_mask, return_dict=True)
last_hidden_state = model_ret['last_hidden_state'][0]
last_hidden_state = last_hidden_state.detach().cpu().numpy()
_input_ids = _input_ids.detach().cpu().numpy()
assert len(_input_ids[0] == len(last_hidden_state))
assert _input_ids[0][0] == 0
tokId_emb[0] = last_hidden_state[0]

# tokId = 1 -> </s> token
_tok = tokenizer("</s>", return_tensors='pt', add_special_tokens=False)
_input_ids = _tok['input_ids'].cuda()
_tok_decode = tokenizer.convert_ids_to_tokens(_input_ids[0])
assert _tok_decode[0] == "</s>"
assert len(_tok_decode) == 1
tok_Idlist_dict[_tok_decode[0]] = [1]
tok_Id_dict[1] = _tok_decode[0] 
_attention_mask = _tok['attention_mask'].cuda()
model_ret = model(input_ids=_input_ids, attention_mask=_attention_mask, return_dict=True)
last_hidden_state = model_ret['last_hidden_state'][0]
last_hidden_state = last_hidden_state.detach().cpu().numpy()
_input_ids = _input_ids.detach().cpu().numpy()
assert len(_input_ids[0] == len(last_hidden_state))
assert _input_ids[0][0] == 1
tokId_emb[1] = last_hidden_state[0]

total_tok_num = 0
tokId = 2

for corpusId in tqdm(range(corpus_num)):
   elem = corpus_file["corpus"][corpusId]
   _tok = tokenizer(elem, return_tensors="pt", add_special_tokens=False)
   if corpusId == 0: print(_tok)

   _input_ids = _tok["input_ids"].cuda()
   _attention_mask = _tok["attention_mask"].cuda()

   _tok_decode = tokenizer.convert_ids_to_tokens(_input_ids[0])
   model_ret = model(input_ids=_input_ids, attention_mask=_attention_mask, return_dict=True)
   last_hidden_state = model_ret['last_hidden_state'][0] #  [seq_len, 768]
   last_hidden_state = last_hidden_state.detach().cpu().numpy()
   _input_ids = _input_ids.detach().cpu().numpy()

   _tok_dict = {}
   assert len(_input_ids[0])==len(last_hidden_state)==len(_tok_decode)
   total_tok_num += len(last_hidden_state)

   for tok_pos, (_text, _ids, _emb) in enumerate(zip(_tok_decode, _input_ids[0], last_hidden_state)):
      tok_Id_dict[tokId] = _text 
      if _text not in tok_Idlist_dict.keys():
          tok_Idlist_dict[_text] = [tokId]
      else:
          tok_Idlist_dict[_text].append(tokId)
      _tok_dict[tokId] = _emb
      tokId_corpus[tokId] = [corpusId, tok_pos]
      tokId_emb[tokId] = _emb
      tokId += 1

   corpusId_corpus_dict[corpusId] = elem
   corpusId_emb_dict[corpusId] = _tok_dict 


print(f'total_tok_num: {total_tok_num}')
print(f'tokId: {tokId}')

tokGroupId_tok_dict = {}
tokId_tokGroupId = {}
tokGroupId_tokIdList = {}
tokGroupId = 2 ## assert tokGroupId 1 is </s> for generate()
tokTextList = list(tok_Idlist_dict.keys())
assert len(tokTextList) == len(set(tokTextList))
for tokText, tokIdList in tok_Idlist_dict.items():
   if tokText == "</s>":
       print(f"Found </s> and set it to 1!!!")
       tokGroupId_tok_dict[1] = tokText 
       tokGroupId_tokIdList[1] = tokIdList  
       for tokId in tokIdList:
           assert tokId not in tokId_tokGroupId.keys()
           tokId_tokGroupId[tokId] = 1 
   elif tokText == "<pad>":
       print(f"Found <pad> and set it to 0!!!")
       tokGroupId_tok_dict[0] = tokText 
       tokGroupId_tokIdList[0] = tokIdList  
       for tokId in tokIdList:
           assert tokId not in tokId_tokGroupId.keys()
           tokId_tokGroupId[tokId] = 0 
   else:
       tokGroupId_tok_dict[tokGroupId] = tokText
       tokGroupId_tokIdList[tokGroupId] = tokIdList
       for tokId in tokIdList:
          assert tokId not in tokId_tokGroupId.keys()
          tokId_tokGroupId[tokId] = tokGroupId
       tokGroupId += 1

"""
각 token은 하나씩 존재하고, GroupId를 가지고 있다. 
해당 GroupId 안에는 여러 tokId가 존재하고 이는 각각 contextualized tokId로 연결
-> GroupId: 이전에서의 tokenId와 비슷한 역할을 한다고 생각하면 된다.
둘 다 0부터 시작하고 tokenId의 0은 <pad> 이고 groupId의 0은 <pad>, 1은 </s>
+ corpus tree는 GroupId로 만들어진다! 
"""

with open("tokGroupId_tok.pickle", "wb") as f:
   pickle.dump(tokGroupId_tok_dict, f)

with open("tokId_tokGroupId.pickle", "wb") as f:
   pickle.dump(tokId_tokGroupId, f)

with open("tokGroupId_tokIdList.pickle", "wb") as f:
   pickle.dump(tokGroupId_tokIdList, f)

with open("tokText_TokIdList.pickle", "wb") as f:
   pickle.dump(tok_Idlist_dict, f)

with open("tokId_tokText.pickle", "wb") as f:
   pickle.dump(tok_Id_dict, f)

with open("corpusId_corpus.pickle", "wb") as f:
   pickle.dump(corpusId_corpus_dict, f)

with open("corpusId_emb.pickle", "wb") as f:
   pickle.dump(corpusId_emb_dict, f)

with open('tokId_emb.pickle', 'wb') as f:
   pickle.dump(tokId_emb, f)

with open('tokId_corpus.pickle', 'wb') as f:
   pickle.dump(tokId_corpus, f)

print("DONE!!")
