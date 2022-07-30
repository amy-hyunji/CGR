# contextualized_GENRE
GENRE with contextualized embedding

### Env
```pip install -r requirements.txt```

### Train
`python train.py --config config/nq_toy.json`

***Notice***
If you do not want to connect slack, remove *@slack_sender* on top of *main* function in train.py

### Files to look at in *transformers*
1. `src/transformers/models/t5/modeling_t5.py`: T5ConditionalGeneration, T5Stack
2. `src/transformers/generation_utils.py`: def generate()

### Config ###
dir: `src/config/full/gr/`
`nq_toy_p1-5_cand.json`: train 때, test와 동일하게 다음에 올 수 있는 토큰들로만 loss 계산
`nq_toy_p1-5_dc.json`: ground truth label이 아닌 decoder output의 logit으로 soft label을 사용
`nq_toy_p1-5_negloss.json`: 정답 label이 속한 groupId 내에 토큰들로 neg loss 계산, 원래 cross entropy loss에 추가
`nq_toy_p1-5_sft_inner_all.json`: 정답 label이 속한 groupId 내에 토큰들로 soft label 사용
`nq_toy_p1-5_total_cluster.json`: scale1에서 전체 토큰을 k-means으로 32,128개 cluster 만든 뒤 cluster center를 토큰 임베딩으로 사용
`nq_toy_p1-5_train_vocab_count_subtree.json`: decoder vocab을 train하는 setting에서 `subtree의 title 개수 >= num_beam` constraint를 사용
`nq_toy_scale10_beam5_bi_p1-5_full_p1-5_ckpt_cluster_5_train.json`: scale10, beam5, subtree constraint 사용o
`nq_toy_scale10_bi_p1-5_full_p1-5_ckpt_cluster_5_train.json`: scale10, beam20, subtree constraint 사용o
`nq_toy_scale10_org_bi_p1-5_full_p1-5_ckpt_cluster_5_train.json`:  scale10, beam20, subtree constraint 사용x
`nq_toy_scale20_beam5_bi_p1-5_full_p1-5_ckpt_cluster_5_train.json`: scale20, beam5, subtree constraint 사용o
`nq_toy_scale20_bi_p1-5_full_p1-5_ckpt_cluster_5_train.json`: scale20, beam20, subtree constraint 사용o
`nq_toy_scale20_org_bi_p1-5_full_p1-5_ckpt_cluster_5_train.json`: scale20, beam20, subtree constraint 사용x
`scale3_beam5_nq_toy_bi_p1-5_full_p1-5_ckpt_cluster_5_train.json`: scale3, beam5, subtree constraint 사용o
`scale3_nq_toy_bi_p1-5_full_p1-5_ckpt_cluster_5_train.json`: scale3, beam20, subtree constraint 사용o
`scale3_org_nq_toy_bi_p1-5_full_p1-5_ckpt_cluster_5_train.json`: scale3, beam20, subtree constraint 사용x

### dataset ###
`emb_analysis.ipynb`: ["Enc-Dec", "Enc-mean", "Enc-first"] 비교
`make_candidate.py`: `nq_toy_p1-5_cand.json` 사용할때 필요한 `tokId2candidate.pickle`를 만들기
`make_count_subtree.sh, make_count_subtree.py`: `cluster_5`(groupId마다 n_cluster=5)인 세팅에서 subtree에 나오는 title 개수 미리 계산, `f"{args.cluster_method}_clusterId_count_subtree_{args.cluster_num}.pickle"` 꼴로 저장, 각 노드마다 [-1]에 subtree에 있는 title 개수 저장

### transformers ###
`loss_transformers`: `modeling_t5.py`에 실험했던 loss들 구현
`total_cluster_transformers`: `modeling_t5.py`의 def _shift_right(), `generation_utils.py`의 def beam_search()에 pad_token_id(13382), eos_token_id(21) 지정
`subtree_transformers`: `generation_utils.py`의 def beam_search(), `generation_beam_search.py`: class BeamSearchScorer() - def process()