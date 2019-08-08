from sklearn.preprocessing import OneHotEncoder

from benchmarking_methods import run_benchmark
from building_classifiers import lstm_default
from feature_creation import create_feature
from loading_data import load_uiuc
import numpy as np

if __name__ == '__main__':
    model = {'name': 'lstm', 'model': lstm_default}

    for language in ['en']:
        print('\n\nLanguage: ', language)
        # embedding = load_embedding(path_wordembedding + 'wiki.multi.' + language + '.vec')
        dataset_train, dataset_test = load_uiuc(language)
        # debug
        print('WARNING: use subset (first 1000 entries) of training data')
        # dataset_train = dataset_train[:5500].copy()

        create_feature('bert', dataset_train, dataset_train)
        create_feature('bert', dataset_train, dataset_test)

        X_train = np.array([x for x in X_train])
        X_test = np.array([x for x in X_test])

        # X_train = pad_sequences(X_train, maxlen=12, dtype='float', padding='post', truncating='post', value=0.0)
        # X_test = pad_sequences(X_test, maxlen=12, dtype='float', padding='post', truncating='post', value=0.0)
        y_train = dataset_train['class'].values
        y_test = dataset_test['class'].values
        ohe = OneHotEncoder()
        y_train = ohe.fit_transform([[y_] for y_ in y_train]).toarray()
        y_test = ohe.transform([[y_] for y_ in y_test]).toarray()
        run_benchmark(model, X_train, y_train, X_test, y_test, sizes_train=[5500],  # 1000, 2000, 3000, 4000, 5500
                      runs=1, save='results/UIUC_lstm_bert_' + language + '.csv',
                      epochs=100, onehot=ohe, in_dim=768)
        # run_benchmark(model, X_train, y_train_sub, X_test_sub_, y_test_sub_, sizes_train=[1000, 2000, 3000, 4000, 5500],
        #              save='results/UIUCsub_svm_tfidf_' + language + '.csv')

