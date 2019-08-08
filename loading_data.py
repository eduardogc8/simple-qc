import io
import numpy as np
import pandas as pd


def load_embedding(emb_path, nmax=50000):
    embedding = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in embedding, 'word found twice'
            embedding[word] = vect
            if len(embedding) == nmax:
                break
    return embedding


def load_uiuc(language):
    # language: 'en', 'pt' or 'es'
    return pd.read_csv('datasets/UIUC_' + language + '/train_features.csv'), pd.read_csv('datasets/UIUC_' + language + '/test_features.csv')


def load_disequa(language):
    df = pd.read_csv('datasets/DISEQuA/disequa_features.csv')
    return df[df['language'] == language]