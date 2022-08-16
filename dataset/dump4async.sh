EMB="/data/contextualized_GENRE/outputs/base.total_nq.wo_para.w_cluster5.512/best_tfmr_5"
SAVE_PATH="temp.wo_para"
DATA="nq"

python dump_bi_full_corpus.py --corpus n_kilt_total_corpus.csv --train_file total_$DATA/$DATA_train.csv --dev_file total_$DATA/$DATA_dev.csv --test_file total_$DATA/$DATA_dev.csv --save_path total_$DATA/$SAVE_PATH --emb_path $EMB --t5 --dump_batch 100 --cluster_num 5 --data_name $DATA --action dump

python dump_bi_full_corpus.py --corpus n_kilt_total_corpus.csv --train_file total_$DATA/$DATA_train.csv --dev_file total_$DATA/$DATA_dev.csv --test_file total_$DATA/$DATA_dev.csv --save_path total_$DATA/$SAVE_PATH  --emb_path $EMB --t5 --dump_batch 100 --cluster_num 5 --data_name $DATA --action dataset

python dump_bi_full_corpus.py --corpus n_kilt_total_corpus.csv --train_file total_$DATA/$DATA_train.csv --dev_file total_$DATA/$DATA_dev.csv --test_file total_$DATA/$DATA_dev.csv --save_path total_$DATA/$SAVE_PATH --emb_path $EMB --t5 --dump_batch 100 --cluster_num 5 --data_name $DATA --action cluster

python dump_bi_full_corpus.py --corpus n_kilt_total_corpus.csv --train_file total_$DATA/$DATA_train.csv --dev_file total_$DATA/$DATA_dev.csv --test_file total_$DATA/$DATA_dev.csv --save_path total_$DATA/$SAVE_PATH --emb_path $EMB --t5 --dump_batch 100 --cluster_num 5 --data_name $DATA --action cluster_dataset