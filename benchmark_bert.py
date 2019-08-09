import sys

import torch
from pytorch_transformers import BertModel, BertTokenizer
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from torch.nn.utils.rnn import pad_sequence

from bert_embeddings import embedd_with_bert, load_bert_to_cache

sys.path.append('.')
from sklearn.preprocessing import OneHotEncoder

from benchmarking_methods import run_benchmark
from building_classifiers import lstm_default, svm_linear
from feature_creation import create_feature
from loading_data import load_uiuc
import numpy as np


def bert_seqs_keras_lstm(y_train,y_test,bert_seq_train,bert_seq_test):
    X_train = pad_sequence(bert_seq_train, padding_value=0).transpose(0, 1).numpy()
    X_test = pad_sequence(bert_seq_test, padding_value=0).transpose(0, 1).numpy()
    ohe = OneHotEncoder()
    y_train = ohe.fit_transform([[y_] for y_ in y_train]).toarray()
    y_test = ohe.transform([[y_] for y_ in y_test]).toarray()
    model = {'name': 'lstm', 'model': lstm_default}
    run_benchmark(model, X_train, y_train, X_test, y_test, sizes_train=[X_train.shape[0]],  # 1000, 2000, 3000, 4000, 5500
                  runs=1, onehot=ohe, save='results/UIUC_lstm_bert_' + language + '.csv',
                  epochs=100, in_dim=768)


def bert_pooled_sklearn_clf(y_train,y_test,bert_pooled_train,bert_pooled_test):

    X_train = torch.cat([x.unsqueeze(0) for x in bert_pooled_train], dim=0).numpy()
    X_test = torch.cat([x.unsqueeze(0) for x in bert_pooled_test], dim=0).numpy()
    model = {'name': 'svm', 'model': svm_linear}
    run_benchmark(model, X_train, y_train, X_test, y_test, sizes_train=[X_train.shape[0]],  # 1000, 2000, 3000, 4000, 5500
                  runs=1, save='results/UIUC_lstm_bert_' + language + '.csv',
                  epochs=100, in_dim=768)


if __name__ == '__main__':
    # model = {'name': 'lstm', 'model': lstm_default}
    # model = {'name': 'linear_clf', 'model': lambda : SGDClassifier()}

    bert_model_name = 'bert-base-multilingual-cased'
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name)

    language = 'en'
    dataset_train, dataset_test = load_uiuc(language)
    dataset_train = dataset_train[:550].copy()
    dataset_test = dataset_test[:100].copy()

    bert_seq_train,bert_pooled_train = embedd_with_bert(dataset_train['question'],bert_tokenizer,bert_model)
    bert_seq_test,bert_pooled_test = embedd_with_bert(dataset_test['question'],bert_tokenizer,bert_model)

    y_train = dataset_train['class'].values
    y_test = dataset_test['class'].values

    bert_pooled_sklearn_clf(y_train,y_test,bert_pooled_train,bert_pooled_test)

    bert_seqs_keras_lstm(y_train,y_test,bert_seq_train,bert_seq_test)