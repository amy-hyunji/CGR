import pickle
import jsonlines

w_test_df = {"input": [], "output": []}
w_test_dict = {"input": [], "output": [], "output_tokid": []}
test_file = "total_wow/wow-test_without_answers-kilt.jsonl"
pickle_path = "total_wow/wow_test.pickle"
csv_path = "total_wow/wow_test.csv"

with jsonlines.open(test_file) as f:
   for i, line in enumerate(f.iter()):
      w_test_dict["input"].append(line["input"]) 
      w_test_dict["output"].append(line["input"]) 
      w_test_dict["output_tokid"].append([0]) 

          

with open(save_path, "wb") as f:
   pickle.dump(w_test_dict, f)
