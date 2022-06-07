### Config

`output_dir`: ckpt 저장할 output path  
`dataset`: extract_embedding으로 뽑은 dataset이 있는 directory   
`model`: t5-base (현재 t5-base만 support)  
`max_input_length`: 40 (default for kilt-nq title)   
`max_output_length`: 30 (default for kilt-nq title)   
`learning_rate`: 1e-4 (default)  
`lr_scheduler`: exponential (default)  
`accelerator`: deepspeed (ddp|deepspeed -> for gr use deepspeed, for bi use ddp)  
`num_train_epochs`: 150 (default)  
`train_batch_size`: 64 (default)  
`eval_batch_size`: 60 (default)  
`gradient_accumulation_steps`:   
`n_gpu`:   
`num_workers`: 0  
`resume_from_checkpoint`: ckpt path to resume training | null   
`train_file`:   
`dev_file`:   
`test_file`:   
`do_train`: true | false  
`do_test`: true | false   
`test_model_path`: model ckpt when do_test == True   
`val_beam_size`: 5 (default)  
`wandb_log`: true  
`wandb_project`:   
`wandb_run_name`:   
`check_val_every_n_epoch`:   
`contextualized_emb_num`:   
`contextualized_file`: tokId_emb.pickle  
`groupId2tokIdList`: tokGroupId_tokIdList.pickle  
`tokId2groupId`: tokId_tokGroupId.pickle  
`tokId2tokText`: tokId_tokText.pickle  
`tokId2corpus`: tokId_corpus.pickle  
`tree_type`: groupId |  nodeId  
`groupId_tree`: groupId_tree.pickle  
`nodeId_tree`: nodeId_tree.pickle  
`nodeId_sup`: nodeId_sup_set.pickle  
`freeze_encoder`: true | false  
`freeze_vocab_emb`: true | false  
`embedding_model`: t5-base  
`max_beam_search`: false  
`bi_encoder`: true | false   
`periflow`: true | false   
`periflow_dir`: /workspace/ckpt (default path to save ckpt when periflow==True)   
`limit_val_batches`: 1.0  
