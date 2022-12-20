import re
import pickle
from tqdm import tqdm
from collections import defaultdict
from transformers import T5Tokenizer

def construct_input2elist(data_dict):
   text2id = {}
   input2elist = defaultdict(list)
   for _input, _ret, _ret_tokid, _ans, _ans_tokid in zip(data_dict["input"], data_dict["retrieval"], data_dict["retrieval_tokid"], data_dict["answer"], data_dict["answer_tokid"]):
      if _ret in _input:
         if _ret in text2id:
            assert text2id[_ret] == _ret_tokid
         else:
            text2id[_ret] = _ret_tokid
            input2elist[_input].append(_ret)
      if _ans in _input:
         if len(_ans_tokid) == 0:
            continue 
         else:
            if _ans in text2id:
               assert text2id[_ans] == _ans_tokid
            else:
               text2id[_ans] = _ans_tokid
               input2elist[_input].append(_ans)
   return text2id, input2elist

def remove_dup(elist, q):
   u_elist = []
   for o, o_e in enumerate(elist):
      dup_exists=False
      for i, i_e in enumerate(elist):
         if o == i: continue 
         else:
            if o_e in i_e:
               dup_exists = True 
               break 
      if not dup_exists and o_e in q:
         u_elist.append(o_e)
   return u_elist

def remove_space(s):
   while s.startswith(" "):
      s = s[1:]
   while s.endswith(" "):
      s = s[:-1]
   return s

def do_tok(qlist, text2id, tok):
   t_idlist = []
   for q in qlist:
      if "[Es]" not in q and "[Ee]" not in q:
         # is entity
         idlist = text2id[q]
         if idlist[-1] == 1:
            idlist = idlist[:-1]
      else:
         idlist = tok(q, add_special_tokens=False)["input_ids"]
      t_idlist.extend(idlist)
   t_idlist.append(1)
   return t_idlist

def add_input_tokid(data_dict, tok):
   text2id, input2elist = construct_input2elist(data_dict)
   print(f"** Done Constructing input2elist!")

   exists=0; not_exists=0
   data_dict["input_tokid"] = []
   for q_idx, q in tqdm(enumerate(data_dict["input"])):
      if len(input2elist[q]) != 0:
         print(f"[Prev] elist: {input2elist[q]}")
      elist = remove_dup(input2elist[q], q)
      if len(elist) != 0:
         print(f"[New] elist: {elist}")
      if len(elist) == 0:
         data_dict["input_tokid"].append([])
         not_exists+=1
      else:
         assert "**::" not in q
         print(f"[First] {q}")
         for e in elist:
            if e not in q: continue
            assert e in q
            q = q.replace(e, f"[Es]**:: {e} **::[Ee]")
            print(f"[add {e}] {q}")

         _qlist = q.split("**::")
         qlist = []
         for _q in _qlist:
            if _q == "": continue 
            else:
               qlist.append(remove_space(_q))
         q_ids = do_tok(qlist, text2id, tok)
         print(f"[original tok]: {tok(q)['input_ids']}")
         print(f"[q-id]: {q_ids}")
         print("\n\n")
         data_dict["input_tokid"].append(q_ids)    
         exists+=1
   assert len(data_dict["input_tokid"])==len(data_dict["input"])
   print(f"## Entity Exists: {exists}\tNot Exists: {not_exists}")
   return data_dict

if __name__ == "__main__":
   # input, retrieval, retrieval_tokid, answer, answer_tokid
   
   tok = T5Tokenizer.from_pretrained("t5-base")
   sp_tokens = ["[Title]", "[Answer]", "[Es]", "[Ee]"]
   tok.add_tokens(sp_tokens, special_tokens=True)

   train = pickle.load(open("tqa/gr_tqa_train_cluster_5_ans.w_entity.pickle", "rb"))
   n_train = add_input_tokid(train, tok)
   with open("tqa/gr_tqa_train_cluster_5_ans.w_entity.enc.pickle", "wb") as f:
      pickle.dump(n_train, f)

   dev = pickle.load(open("tqa/gr_tqa_dev_cluster_5_ans.w_entity.pickle", "rb"))
   n_dev = add_input_tokid(dev, tok)
   with open("tqa/gr_tqa_dev_cluster_5_ans.w_entity.enc.pickle", "wb") as f:
      pickle.dump(n_dev, f)

