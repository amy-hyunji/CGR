#python extract_embedding.py --corpus nq_toy_corpus.csv --train_file nq_toy_train.csv --dev_file nq_toy_dev.csv --test_file nq_toy_test.csv --save_path test --emb_path t5-base
python extract_embedding.py --corpus nq_toy_corpus.csv --train_file nq_toy_train.csv --dev_file nq_toy_dev.csv --test_file nq_toy_test.csv --save_path t5-base-emb-only-first --emb_path /mnt/entailment/bi_contextualized_GENRE/contextualized_GENRE/outputs/nq_toy_only_first/best_tfmr_149 --t5 