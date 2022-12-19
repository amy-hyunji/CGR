import pickle
import pandas as pd

from collections import defaultdict

def clm2t5(clm_f):

    clm_f = pickle.load(open(clm_f, "rb"))

    t5_cbqa = {"input": [], "output": []} 
    t5_ret_ans = {"input": [], "output": []} 

    input2ans = defaultdict(list) 
    input2retans = defaultdict(list) 
    for _input, _retrieval, _answer in zip(clm_f["input"], clm_f["retrieval"], clm_f["answer"]):
        if _answer not in input2ans[_input]:
            input2ans[_input].append(_answer)
        ret_ans = f"{_retrieval} [Answer] {_answer}"
        if ret_ans not in input2retans[_input]:
            input2retans[_input].append(ret_ans)

    for _input, _answer in input2ans.items():
        for _ans in _answer:
            t5_cbqa["input"].append(_input)
            t5_cbqa["output"].append(_ans)

    for _input, _ret_ans in input2retans.items():
        for _ans in _ret_ans:
            t5_ret_ans["input"].append(_input)
            t5_ret_ans["output"].append(_ans)

    t5_cbqa = pd.DataFrame(t5_cbqa)
    t5_ret_ans = pd.DataFrame(t5_ret_ans)
    return t5_cbqa, t5_ret_ans

if __name__ == "__main__":

    train_clm_f = "tqa/gr_tqa_train_cluster_5_ans.pickle"
    dev_clm_f = "tqa/gr_tqa_dev_cluster_5_ans.pickle"

    t5_cbqa, t5_ret_ans = clm2t5(train_clm_f) 
    t5_cbqa.to_csv("t5.gr_tqa_train.cbqa.csv")
    t5_ret_ans.to_csv("tqa/t5.gr_tqa_train.ret_ans.csv")
    t5_cbqa, t5_ret_ans = clm2t5(dev_clm_f) 
    t5_cbqa.to_csv("t5.gr_tqa_dev.cbqa.csv")
    t5_ret_ans.to_csv("tqa/t5.gr_tqa_dev.ret_ans.csv")
