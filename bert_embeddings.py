import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import torch
from pytorch_transformers import BertModel
from pytorch_transformers import BertTokenizer

def embedd_with_bert(df_2,bert_tokenizer,bert_model):
    # Encode text
    print('tokenize...')
    input_ids = [bert_tokenizer.encode(s) for s in df_2['question']]
    input_ids_padded = pad_sequences(input_ids, maxlen=12, dtype='int64', padding='post', truncating='post', value=0)
    input_ids_tensor = torch.tensor(input_ids_padded)
    print('embed with BERT...')
    with torch.no_grad():
        ret = []
        batch_size = 32
        for ind in tqdm(range(0, len(input_ids), batch_size)):
            batch_input = input_ids_tensor[ind:ind + batch_size]
            last_hidden_states = bert_model(batch_input)[0]  # Models outputs are now tuples
            ret.append(last_hidden_states.numpy())
        ret = np.concatenate(ret, axis=0)
        print(f'shape of encoded input: {ret.shape}')
        df_2['bert'] = [enc for enc in ret]


def load_bert_to_cache(cache):
    bert_model_name = 'bert-base-multilingual-cased'
    if 'bert_tokenizer' not in cache:
        cache['bert_tokenizer'] = BertTokenizer.from_pretrained(bert_model_name)
    if 'bert_model' not in cache:
        cache['bert_model'] = BertModel.from_pretrained(bert_model_name)