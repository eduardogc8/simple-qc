import sys
sys.path.append('.')

from flair_cnn_doc_embedding import DocumentCNNEmbeddings
from torch.utils.data import Dataset

import time
from loading_data import load_uiuc

from pprint import pprint
from typing import List

import torch
from flair.data import Sentence, Corpus
from flair.embeddings import DocumentRNNEmbeddings, BertEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from sklearn import metrics


def build_flair_sentences(text_label_tuples):
    sentences = [Sentence(text, labels=[label], use_tokenizer=True) for text,label in text_label_tuples]
    return [s for s in sentences if len(s.tokens) > 0]

def get_labels(sentences:List[Sentence]):
    return [[l.value for l in s.labels] for s in sentences]


def calc_metrics_with_sklearn(clf:TextClassifier,sentences:List[Sentence]):
    targets = get_labels(sentences)
    clf.predict(sentences)
    prediction = get_labels(sentences)
    report = metrics.classification_report(y_true=targets, y_pred=prediction, digits=3,
                                           output_dict=True)
    return report

if __name__ == '__main__':
    start = time.time()

    language = 'en'
    dataset_train, dataset_test = load_uiuc(language)
    # dataset_train = dataset_train[:550].copy()
    # dataset_test = dataset_test[:100].copy()

    sentences_train:Dataset = build_flair_sentences([(text, label) for text, label in zip(dataset_train['question'], dataset_train['class'])])
    sentences_dev:Dataset = sentences_train
    sentences_test:Dataset = build_flair_sentences([(text, label) for text, label in zip(dataset_test['question'], dataset_test['class'])])

    corpus: Corpus = Corpus(sentences_train, sentences_dev, sentences_test)
    label_dict = corpus.make_label_dictionary()
    word_embeddings = [
        # WordEmbeddings('glove'),
        BertEmbeddings('bert-base-multilingual-cased',layers='-1')
    ]

    document_embeddings = DocumentCNNEmbeddings(word_embeddings,
                                                dropout=0.0,
                                                hidden_size=64,
                                                )

    clf = TextClassifier(document_embeddings, label_dictionary=label_dict, multi_label=False)
    print(clf)
    trainer = ModelTrainer(clf, corpus,torch.optim.RMSprop)
    base_path = 'flair_resources/qc_en_uiuc'
    trainer.train(base_path,
                  learning_rate=0.001,
                  mini_batch_size=32,
                  anneal_factor=0.5,
                  patience=2,
                  max_epochs=4)

    pprint('train-macro-f1: %0.2f'%calc_metrics_with_sklearn(clf,sentences_train)['macro avg']['f1-score'])
    pprint('test-macro-f1: %0.2f'%calc_metrics_with_sklearn(clf,sentences_test)['macro avg']['f1-score'])

'''
DocumentRNNEmbeddings (GRU inside):
trained for 9 epochs (~15min)
'train-macro-f1: 0.95'
'test-macro-f1: 0.92' 

DocumentCNNEmbeddings: 
trained for 4 epochs (~7min)
'train-macro-f1: 0.95'
'test-macro-f1: 0.94'
'''