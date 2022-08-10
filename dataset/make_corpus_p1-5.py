import json
import pandas as pd
from tqdm import tqdm

dataset = "nq"
scale = "full" 
if scale == "full":
   df_corpus = pd.read_csv(f'kilt_total_corpus.csv', header=0, names=['corpusId', 'corpus'])
else:
   df_corpus = pd.read_csv(f'kilt_{dataset}/scale_{scale}_kilt_corpus.csv', header=0, names=['corpusId', 'corpus'])

print(f'Open kilt_title_text ... ')
with open('kilt_title_text.json', 'r') as f:
    title_context = json.load(f)
print(f'Done Opening ..!')

assert len(title_context['title']) == len(title_context['text'])

title2text = {}
for title, text in zip(title_context['title'], title_context['text']):
    assert title not in title2text.keys(), title
    title2text[title] = text

def text_normalize(text):
    #text = text.strip()
    text = text.replace('\n', '')
    text = text.replace('&amp;', '&')
    text = text.replace('&quot;', "\"")
    return text

df_ret = []
no_context = []

for corpusId in tqdm(df_corpus['corpusId']):
    title = df_corpus['corpus'][corpusId]
    if title not in title2text.keys():
        context = title 
        context_list = [title] 
    else:
        context = title2text[title]
        if title != text_normalize(context[0]): 
            print(f'##{title}## || ##{text_normalize(context[0])}##')
            print(title, text_normalize(context[0]))
            title2text[title][0] = title

        context_list = []
        for text in context[1:]:
            if len(context_list) == 5:
                break 
            text = text.strip()
            if "::::" in text:
                continue 
            context_list.append(text)

    if len(context_list) == 0:
        no_context.append(title)

    assert len(context_list) <= 5 
    try: 
        context_list = " ".join(context_list)
    except:
        print(f'context_list')
        context_list = ""
    context_list = context_list.strip() 
    df_ret.append([corpusId, title, context_list])

df_ret = pd.DataFrame(df_ret, columns=['corpusId', 'corpus', 'context'])
if scale == "full":
   df_ret.to_csv(f'kilt_total_corpus_p1-5.csv')
else:
   df_ret.to_csv(f'kilt_{dataset}/scale_{scale}_kilt_corpus_p1-5.csv')

print(f'Done Saving!')
print(f'No Context: {len(set(no_context))}')

