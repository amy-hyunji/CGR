CUDA_VISIBLE_DEVICES=0 python dump_bi_full_corpus.py --corpus kilt_total_corpus.csv --train_file total_nq/nq_train.csv --dev_file total_nq/nq_dev.csv --test_file total_nq/nq_dev.csv --save_path total_nq/t5_large.nq_full --emb_path t5-large --t5 --dump_batch 100

#CUDA_VISIBLE_DEVICES=2 python split_save_dataset.py --corpus kilt_trex/scale_20_kilt_corpus_p1-5.csv --train_file kilt_trex/trex_toy_train.csv --dev_file kilt_trex/trex_toy_dev.csv --test_file kilt_trex/trex_toy_test.csv --save_path kilt_trex/bi-scale20-trex_toy_base_p1-5_ckpt/ --emb_path t5-base --t5 --bi --filenum 41

#python cluster_embs.py --cluster_num 5 --basedir kilt_trex/gr-scale20-trex_toy_base_p1-5_ckpt
