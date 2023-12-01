# CGR

### Env
```pip install -r requirements.txt```

### Train
`python train.py --config config/nq_toy.json`


### Files to look at in *transformers*
1. `src/transformers/models/t5/modeling_t5.py`: T5ConditionalGeneration, T5Stack
2. `src/transformers/generation_utils.py`: def generate()
