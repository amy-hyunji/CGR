import json
import pickle
import jsonlines

def input2ansret(kilt_f):

    input2ans_ret = {}

    with jsonlines.open(kilt_f) as f:
        for line in f.iter():
            _input = line["input"]
            output_list = line["output"]
            for out_dict in output_list:
                if "answer" not in out_dict or "provenance" not in out_dict:
                    continue
                else:
                    _answer = out_dict["answer"]
                    _ret_list = []
                    prov_list = out_dict["provenance"]
                    for prov in prov_list:
                        if "title" in prov:
                           _ret_list.append(prov["title"])
                    _ret_list = list(set(_ret_list))
                    if len(_ret_list) == 0:
                        continue 
                    else:
                        if _input in input2ans_ret:
                            input2ans_ret[_input].append([_answer, _ret_list])
                        else:
                            input2ans_ret[_input] = [[_answer, _ret_list]]
    return input2ans_ret

def dataset_w_entity(kilt_f, cgr_f, corpus2tokList, tokId2clusterId):
    input2ans_ret = input2ansret(kilt_f)
    cgr_f = pickle.load(open(cgr_f, "rb"))
    
    ret_dict = {"input": [], "retrieval": [], "retrieval_tokid": [], "answer": [], 'answer_tokid': []}
   
    is_entity = 0; not_entity = 0
    for input, ans_ret_list in input2ans_ret.items():
        for ans, retlist in ans_ret_list:
            for ret in retlist:
                if ret in corpus2tokList:
                    ret_dict["input"].append(input)
                    ret_dict["retrieval"].append(ret)
                    ret_dict["retrieval_tokid"].append(get_tokList(ret, corpus2tokList, tokId2clusterId))
                    if ans in corpus2tokList:
                        is_entity += 1
                        ret_dict["answer"].append(ans)
                        ret_dict["answer_tokid"].append(get_tokList(ans, corpus2tokList, tokId2clusterId))
                    else:
                        not_entity += 1
                        ret_dict["answer"].append(ans)
                        ret_dict["answer_tokid"].append([])

    print(f'Is Entity: {is_entity}\t Not Entity: {not_entity}')
    print(f"# of input2ans_ret => {len(set(list(input2ans_ret.keys())))}")
    print(f"# of cgr dataset => {len(set(cgr_f['input']))}")
    assert len(set(list(input2ans_ret.keys()))) == len(list(set(cgr_f["input"])))
    return ret_dict



def add2cgr(kilt_f, cgr_f, corpus2tokList, tokId2clusterId):
    input2ans_ret = input2ansret(kilt_f)
    cgr_f = pickle.load(open(cgr_f, "rb"))
    
    ret_dict = {"input": [], "retrieval": [], "retrieval_tokid": [], "answer": []}
    
    for input, ans_ret_list in input2ans_ret.items():
        for ans, retlist in ans_ret_list:
            for ret in retlist:
                if ret in corpus2tokList:
                    ret_dict["input"].append(input)
                    ret_dict["retrieval"].append(ret)
                    ret_dict["retrieval_tokid"].append(get_tokList(ret, corpus2tokList, tokId2clusterId))
                    ret_dict["answer"].append(ans)

    print(f"# of input2ans_ret => {len(set(list(input2ans_ret.keys())))}")
    print(f"# of cgr dataset => {len(set(cgr_f['input']))}")
    assert len(set(list(input2ans_ret.keys()))) == len(list(set(cgr_f["input"])))
    return ret_dict

def get_tokList(ret, corpus2tokList, tokId2clusterId):
    tokList = corpus2tokList[ret]
    clusterList = [tokId2clusterId[el] for el in tokList]
    return clusterList

if __name__ == "__main__":
    
    check_entity = True

    kilt_train = "../../total_trivia/trivia-train-kilt.jsonl"
    kilt_dev = "../../total_trivia/trivia-dev-kilt.jsonl"

    cgr_train = "./tqa/gr_tqa_train_cluster_5.pickle"
    cgr_dev = "./tqa/gr_tqa_dev_cluster_5.pickle"
   
    if check_entity:
        clm_train = "./tqa/gr_tqa_train_cluster_5_ans.w_entity.pickle"
        clm_dev = "./tqa/gr_tqa_dev_cluster_5_ans.w_entity.pickle"
    else:
        clm_train = "./tqa/gr_tqa_train_cluster_5_ans.pickle"
        clm_dev = "./tqa/gr_tqa_dev_cluster_5_ans.pickle"
    print(f'Start Saving\nTrain Dataset: {clm_train}\nDev Dataset: {clm_dev}')


    corpus2tokList = pickle.load(open("corpus_tokenList_dict.pickle", "rb")) 
    tokId2clusterId = pickle.load(open("tokId2clusterId_5.pickle", "rb")) 
    
    if check_entity:
        train_dict = dataset_w_entity(kilt_train, cgr_train, corpus2tokList, tokId2clusterId)
    else:
        train_dict = add2cgr(kilt_train, cgr_train, corpus2tokList, tokId2clusterId)
    with open(clm_train, "wb") as f:
        pickle.dump(train_dict, f)

    if check_entity:
        dev_dict = dataset_w_entity(kilt_dev, cgr_dev, corpus2tokList, tokId2clusterId)
    else:
        dev_dict = add2cgr(kilt_dev, cgr_dev, corpus2tokList, tokId2clusterId)
    with open(clm_dev, "wb") as f:
        pickle.dump(dev_dict, f)

    print(f'Done Saving\nTrain Dataset: {clm_train}\nDev Dataset: {clm_dev}')
