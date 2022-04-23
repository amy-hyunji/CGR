### Order

1. Change the corpus file in *t5_embedding.py*
2. Run *t5_embedding.py* : ```python t5_embedding.py```
3. Change *Parameters to Change* in *construct_dataset.py*
4. Run *construct_dataset.py*: ```python construct_dataset.py```
5. Change *Parameters to Change* in *construct_corpus_tree.py* 
5. Run *construct_corpus_tree.py* in base directory: ```python construct_corpus_tree.py```

### How to download the Files
```azcopy cp https://stg4hyunji.blob.core.windows.net/hyunji/contextualized_GENRE/dataset/kilt_nq . --recursive```  
=> this will create *kilt_nq* file and save all necessary files under *kilt_nq*

```azcopy cp https://stg4hyunji.blob.core.windows.net/hyunji/contextualized_GENRE/vocab_emb.pickle .```
