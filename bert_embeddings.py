from typing import Dict

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch
from pytorch_transformers import BertModel
from pytorch_transformers import BertTokenizer

def embedd_with_bert_using_df(df_2, bert_tokenizer, bert_model,max_seqlen=12):
    # Encode text
    questions = [q for q in df_2['question']]
    bert_seq, bert_pooled = embedd_with_bert(questions,bert_tokenizer,bert_model,max_seqlen)

    if max_seqlen is not None:
        bert_seq = pad_sequence([x for xs in bert_seq for x in xs],padding_value=0)
        bert_pooled = pad_sequence([x for xs in bert_pooled for x in xs],padding_value=0)

    df_2['bert'] = [x.numpy() for x in bert_seq]
    df_2['bert_pooled'] = [x.numpy() for x in bert_pooled]

def embedd_with_bert(questions,bert_tokenizer,bert_model,max_seqlen=None,batch_size = 32):
    print('tokenize...')
    def tokenize_eventually_truncate(s:str):
        seq = bert_tokenizer.encode(s)
        return seq[:max_seqlen] if max_seqlen is not None else seq
    sequences = [tokenize_eventually_truncate(s) for s in questions]
    sorted_ids = sorted(range(len(sequences)),key=lambda k:len(sequences[k]))
    lenght_sorted_sequences = [sequences[k] for k in sorted_ids]
    print('embed with BERT...')
    bert_seq, bert_pooled = [],[]
    with torch.no_grad():
        for ind in tqdm(range(0, len(lenght_sorted_sequences), batch_size)):
            batch_seqs = lenght_sorted_sequences[ind:ind + batch_size]
            input_ids = [torch.LongTensor(s) for s in batch_seqs]
            batch_input = pad_sequence(input_ids, padding_value=0).transpose(0,1)
            bert_output = bert_model(batch_input)
            bert_seq.extend([s for s in bert_output[0]])
            bert_pooled.extend([s for s in bert_output[1]])

    restore_order = sorted(range(len(sequences)),key=lambda k:sorted_ids[k])
    bert_seq=[bert_seq[k] for k in restore_order]
    bert_pooled=[bert_pooled[k] for k in restore_order]
    return bert_seq,bert_pooled

def load_bert_to_cache(cache:Dict):
    bert_model_name = 'bert-base-multilingual-cased'
    if 'bert_tokenizer' not in cache:
        cache['bert_tokenizer'] = BertTokenizer.from_pretrained(bert_model_name)
    if 'bert_model' not in cache:
        cache['bert_model'] = BertModel.from_pretrained(bert_model_name)