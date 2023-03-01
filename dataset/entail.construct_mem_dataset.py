import pickle
import pandas as pd
from collections import defaultdict

def remove_space(sen):
   while sen.startswith(" "): sen = sen[1:]
   while sen.endswith(" "): sen = sen[:-1]
   return sen

def get_input(sen):
   sen = sen.split("[BECAUSE]")[0]
   sen = remove_space(sen)
   return sen 

def get_output(sen):
   if "[Es]" not in sen: return None
   sen_list = sen.split("[Es]")
   es_list = []
   for sen in sen_list:
      if "[Ee]" not in sen: continue
      sen = sen.split("[Ee]")[0]
      sen = remove_space(sen)
      es_list.append(sen)
   return es_list

def construct_mem_wo_q(corpus, corpus_tokenList, token_level):
   if not token_level:
      tokId2clusterId = pickle.load(open(f"{basedir}/tokId2clusterId_5.pickle", "rb"))

   df = {"input": [], "output": [], "output_tokid": []}
   for elem in corpus:
      df["input"].append(elem)
      df["output"].append(elem)
      if token_level:
         df["output_tokid"].append(corpus_tokenList[elem])
      else:
         df["output_tokid"].append([tokId2clusterId[el] for el in corpus_tokenList[elem]])
   return df


if __name__ == "__main__":

   with_q = False 
   task = 1
   token_level = True 

   if task == 1:
      basedir = "entailment"
   elif task == 2:
      basedir = "entailment_task2"
   elif task == 3:
      basedir = "entailment_task3"
   else:
      assert False

   corpus_tokenList = pickle.load(open(f"{basedir}/corpus_tokenList_dict.pickle", "rb"))
   
   if with_q:
      train = pd.read_csv(f"{basedir}/iter_train.csv")
      input2output = defaultdict(list) 

      if token_level:
         if "token_mem.pickle" in os.listdir(basedir):
            mem = pickle.load(open(f"{basedir}/token_mem.pickle", "rb")) 
         else:
            corpus = list(pd.read_csv(f"{basedir}/idx_1.csv")["corpus"])
            mem = construct_mem_wo_q(corpus, corpus_tokenList, token_level)
      else:
         if "mem.pickle" in os.listdir(basedir):
            mem = pickle.load(open(f"{basedir}/mem.pickle", "rb")) 
         else:
            corpus = list(pd.read_csv(f"{basedir}/idx_1.csv")["corpus"])
            mem = construct_mem_wo_q(corpus, corpus_tokenList, token_level)
         tokId2clusterId = pickle.load(open(f"{basedir}/tokId2clusterId_5.pickle", "rb"))

      for _input, _output in zip(train["input"], train["output"]):
         _input = get_input(_input)
         _output_list = get_output(_output)
         if _output_list is None: continue
         for _output in _output_list:
            input2output[_input].append(_output)

      df = {"input": [], "output": [], "output_tokid": []}
      for _input, _output_list in input2output.items():
         for _output in _output_list:
            df["input"].append(_input)
            df["output"].append(_output)
            if token_level:
               df["output_tokid"].append(corpus_tokenList[_output])
            else:
               df["output_tokid"].append([tokId2clusterId[el] for el in corpus_tokenList[_output]])

      print(f"# of Q dataset: {len(df['input'])}")
      df["input"].extend(mem["input"])
      df["output"].extend(mem["output"])
      df["output_tokid"].extend(mem["output_tokid"])

      print(f"# of Q+MEM dataset: {len(df['input'])}")

   else:
      corpus = list(pd.read_csv(f"{basedir}/idx_1.csv")["corpus"])
      df = construct_mem_wo_q(corpus, corpus_tokenList, token_level)
      
   with open(f"{basedir}/mem.query_{with_q}.token_level_{token_level}.pickle", "wb") as f:
      pickle.dump(df, f)

