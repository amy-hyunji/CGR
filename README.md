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