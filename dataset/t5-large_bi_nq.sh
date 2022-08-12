python dump_bi_full_corpus.py --corpus n_kilt_total_corpus.csv --train_file total_nq/nq_train.csv --dev_file total_nq/nq_dev.csv --test_file total_nq/nq_dev.csv --save_path total_nq/t5_large.nq_full --emb_path t5-large --t5 --dump_batch 100

#CUDA_VISIBLE_DEVICES=0 python dump_bi_full_corpus.py --corpus kilt_fever/scale_1_kilt_corpus.csv --train_file kilt_fever/fever_toy_train.csv --dev_file kilt_fever/fever_toy_dev.csv --test_file kilt_fever/fever_toy_test.csv --save_path kilt_fever/test --emb_path t5-large --t5 --dump_batch 100


#python cluster_embs.py --cluster_num 5 --basedir kilt_trex/gr-scale20-trex_toy_base_p1-5_ckpt
