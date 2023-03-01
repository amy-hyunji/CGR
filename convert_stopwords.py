from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import T5Tokenizer, BartTokenizer
from tqdm import tqdm

npm_tokenizer = AutoTokenizer.from_pretrained("facebook/npm")
t5_tokenizer = T5Tokenizer.from_pretrained('t5-large')
with open('./roberta_stopwords.txt') as f:
    lines = f.readlines()
stopwords = []
for line in tqdm(lines):
    stopwords.append(npm_tokenizer.decode(int(line)).strip())
    # stopwords.extend(list(t5_tokenizer(npm_tokenizer.decode(int(line)), add_special_tokens=False).input_ids))
    # print(npm_tokenizer.decode(int(line)))
    # print(t5_tokenizer.tokenize(npm_tokenizer.decode(int(line)), return_tensors='pt', add_special_tokens=False))

stop_voc = set(stopwords)

with open('./t5_stopwords.txt','w') as s:
    for i in stop_voc:
        s.write(str(i)+'\n')
