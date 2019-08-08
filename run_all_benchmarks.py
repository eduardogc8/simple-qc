#!/usr/bin/env python
# coding: utf-8

# ## How to run the experiments

# Run the code blocs bellow in sequence. You can read the descriptions to understand it.
# 
# 
# The dependencies can be found in https://github.com/eduardogc8/simple-qc
# 
# Before starting to run the experiments, change the variable ``path_wordembedding``, in the code block below, for the correct directory path. Make sure that the word embedding inside follow the template `wiki.multi.*.vec`.

# In[1]:


from sklearn.feature_selection import SelectKBest, chi2
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np

from benchmark_bert import model, X_train, X_test
from benchmarking_methods import run_benchmark, run_benchmark_cv
from building_classifiers import lstm_default, svm_linear, random_forest, cnn
from download_word_embeddings import muse_embeddings_path, download_if_not_existing
from feature_creation import create_feature
from loading_data import load_embedding, load_uiuc, load_disequa
path_wordembedding = muse_embeddings_path
download_if_not_existing()

import warnings
warnings.filterwarnings("ignore")


# Different classifier models are tested with different dependency levels of external linguistic resources (Low, Medium and High)

# #### SVM + TF-IDF

# In[110]:


for language in ['en', 'es', 'pt']:
    print('\n\nLanguage: ', language)
    dataset_train, dataset_test = load_uiuc(language)
    create_feature('tfidf', dataset_train, dataset_train, max_features=2000)
    create_feature('tfidf', dataset_train, dataset_test, max_features=2000)
    
    model = {'name': 'svm', 'model': svm_linear}
    
    tfidf_train = np.array([list(r) for r in dataset_train['tfidf'].values])
    tfidf_test = np.array([list(r) for r in dataset_test['tfidf'].values])
    tfidf_train = normalize(tfidf_train, norm='max')
    tfidf_test = normalize(tfidf_test, norm='max')
    
    X_train = np.array([list(x) for x in dataset_train['tfidf'].values])
    X_test = np.array([list(x) for x in dataset_test['tfidf'].values])
    y_train = dataset_train['class'].values
    y_test = dataset_test['class'].values
    
    run_benchmark(model, X_train, y_train, X_test, y_test, sizes_train=[1000, 2000, 3000, 4000, 5500],
                  save='results/UIUC_svm_tfidf_' + language + '.csv', runs=1)

# #### SVM + TF-IDF + WB

# In[37]:


for language in ['en', 'es', 'pt']:
    print('\n\nLanguage: ', language)
    embedding = load_embedding(path_wordembedding + 'wiki.multi.' + language + '.vec')
    dataset_train, dataset_test = load_uiuc(language)
    create_feature('tfidf', dataset_train, dataset_train, max_features=2000)
    create_feature('tfidf', dataset_train, dataset_test, max_features=2000)
    create_feature('embedding_sum', None, dataset_train, embedding)
    create_feature('embedding_sum', None, dataset_test, embedding)
    
    model = {'name': 'svm', 'model': svm_linear}
    
    tfidf_train = np.array([list(r) for r in dataset_train['tfidf'].values])
    tfidf_test = np.array([list(r) for r in dataset_test['tfidf'].values])
    tfidf_train = normalize(tfidf_train, norm='max')
    tfidf_test = normalize(tfidf_test, norm='max')
    
    embedding_train = np.array([list(r) for r in dataset_train['embedding_sum'].values])
    embedding_test = np.array([list(r) for r in dataset_test['embedding_sum'].values])
    embedding_train = normalize(embedding_train, norm='max')
    embedding_test = normalize(embedding_test, norm='max')
    
    X_train = np.array([list(x) + list(xx) for x, xx in zip(tfidf_train, embedding_train)])
    X_test = np.array([list(x) + list(xx) for x, xx in zip(tfidf_test, embedding_test)])
    y_train = dataset_train['class'].values
    y_test = dataset_test['class'].values
    
    run_benchmark(model, X_train, y_train, X_test, y_test, sizes_train=[1000, 2000, 3000, 4000, 5500],
                  runs=1, save='results/UIUC_svm_cortes_' + language + '.csv')


# #### SVM + TF-IDF + WB + POS + NER

# In[101]:


for language in ['en', 'es', 'pt']:
    print('\n\nLanguage: ', language)
    embedding = load_embedding(path_wordembedding + 'wiki.multi.' + language + '.vec')
    dataset_train, dataset_test = load_uiuc(language)
    create_feature('tfidf', dataset_train, dataset_train, max_features=2000)
    create_feature('tfidf', dataset_train, dataset_test, max_features=2000)
    create_feature('embedding_sum', dataset_train, dataset_train, embedding)
    create_feature('embedding_sum', dataset_train, dataset_test, embedding)
    create_feature('pos_hotencode', dataset_train, dataset_train)
    create_feature('pos_hotencode', dataset_train, dataset_test)
    create_feature('ner_hotencode', dataset_train, dataset_train)
    create_feature('ner_hotencode', dataset_train, dataset_test)
    model = {'name': 'svm', 'model': svm_linear}
    
    tfidf_train = np.array([list(r) for r in dataset_train['tfidf'].values])
    tfidf_test = np.array([list(r) for r in dataset_test['tfidf'].values])
    tfidf_train = normalize(tfidf_train, norm='max')
    tfidf_test = normalize(tfidf_test, norm='max')
    
    embedding_train = np.array([list(r) for r in dataset_train['embedding_sum'].values])
    embedding_test = np.array([list(r) for r in dataset_test['embedding_sum'].values])
    embedding_train = normalize(embedding_train, norm='max')
    embedding_test = normalize(embedding_test, norm='max')
    
    pos_train = np.array([list(r) for r in dataset_train['pos_hotencode'].values])
    pos_test = np.array([list(r) for r in dataset_test['pos_hotencode'].values])
    
    ner_train = np.array([list(r) for r in dataset_train['ner_hotencode'].values])
    ner_test = np.array([list(r) for r in dataset_test['ner_hotencode'].values])
    
    X_train = np.array([list(x) + list(xx) + list(xxx) + list(xxxx) for x, xx, xxx, xxxx in zip(tfidf_train, embedding_train, pos_train, ner_train)])
    X_test = np.array([list(x) + list(xx) + list(xxx) + list(xxxx) for x, xx, xxx, xxxx in zip(tfidf_test, embedding_test, pos_test, ner_test)])
    
    y_train = dataset_train['class'].values
    y_test = dataset_test['class'].values
    
    classes = list(dataset_train['class'].unique())
    y_train_ = [classes.index(c) for c in y_train]
    
    run_benchmark(model, X_train, y_train, X_test, y_test, sizes_train=[1000, 2000, 3000, 4000, 5500],
                  runs=1, save='results/UIUC_svm_high_' + language + '.csv')


# ## Run UIUC Benchmark - Cross-validation

# Different classifier models are tested with different dependency levels of external linguistic resources (Low, Medium and High)

# #### SVM + TF-IDF

# In[175]:


for language in ['en', 'es', 'pt']:
    print('\n\nLanguage: ', language)
    dataset_train, dataset_test = load_uiuc(language)
    dataset = pd.concat([dataset_train, dataset_test])
    create_feature('tfidf', dataset, dataset, max_features=2000)
    
    model = {'name': 'svm', 'model': svm_linear}
    
    tfidf = np.array([list(r) for r in dataset['tfidf'].values])
    tfidf = normalize(tfidf, norm='max')
    
    X = np.array([list(x) for x in dataset['tfidf'].values])
    y = dataset['class'].values
    
    run_benchmark_cv(model, X, y, [50, 100] + list(range(500, 5501, 500)),
                     save='results/UIUC_cv_svm_tfidf_' + language + '.csv')


# #### SVM + TF-IDF + WB

# In[176]:


for language in ['en', 'es', 'pt']:
    print('\n\nLanguage: ', language)
    embedding = load_embedding(path_wordembedding + 'wiki.multi.' + language + '.vec')
    dataset_train, dataset_test = load_uiuc(language)
    dataset = pd.concat([dataset_train, dataset_test])
    create_feature('tfidf', dataset, dataset, max_features=2000)
    create_feature('embedding_sum', None, dataset, embedding)
    
    model = {'name': 'svm', 'model': svm_linear}
    
    tfidf = np.array([list(r) for r in dataset['tfidf'].values])
    tfidf = normalize(tfidf, norm='max')
    
    embedding = np.array([list(r) for r in dataset['embedding_sum'].values])
    embedding = normalize(embedding, norm='max')
    
    X = np.array([list(x) + list(xx) for x, xx in zip(tfidf, embedding)])
    y = dataset['class'].values
    
    run_benchmark_cv(model, X, y, [50, 100] + list(range(500, 5501, 500)),
                     save='results/UIUC_cv_svm_cortes_' + language + '.csv')


# #### SVM + TF-IDF + WB + POS + NER

# In[177]:


for language in ['en', 'es', 'pt']:
    print('\n\nLanguage: ', language)
    embedding = load_embedding(path_wordembedding + 'wiki.multi.' + language + '.vec')
    dataset_train, dataset_test = load_uiuc(language)
    dataset = pd.concat([dataset_train, dataset_test])
    create_feature('tfidf', dataset, dataset, max_features=2000)
    create_feature('embedding_sum', dataset, dataset, embedding)
    create_feature('pos_hotencode', dataset, dataset)
    create_feature('ner_hotencode', dataset, dataset)
    model = {'name': 'svm', 'model': svm_linear}
    
    tfidf = np.array([list(r) for r in dataset['tfidf'].values])
    tfidf = normalize(tfidf, norm='max')
    
    embedding = np.array([list(r) for r in dataset['embedding_sum'].values])
    embedding = normalize(embedding, norm='max')
    
    pos = np.array([list(r) for r in dataset['pos_hotencode'].values])
    
    ner = np.array([list(r) for r in dataset['ner_hotencode'].values])
    
    X = np.array([list(x) + list(xx) + list(xxx) + list(xxxx) for x, xx, xxx, xxxx in zip(tfidf, embedding, pos, ner)])
    
    y = dataset['class'].values
    
    run_benchmark_cv(model, X, y, [50, 100] + list(range(500, 5501, 500)),
                     save='results/UIUC_cv_svm_high_' + language + '.csv')


# ## Run DISEQuA Benchmark - Cross-validation

# Different classifier models are tested with different dependency levels of external linguistic resources (Low, Medium and High)

# #### SVM + <font color=#007700>TF-IDF</font>

# In[152]:


for language in ['en', 'es', 'it', 'nl']:
    print('\n\nLanguage: ', language)
    dataset = load_disequa(language)
    create_feature('tfidf', dataset, dataset, max_features=2000)
    
    model = {'name': 'svm', 'model': svm_linear}
    
    tfidf = np.array([list(r) for r in dataset['tfidf'].values])
    tfidf = normalize(tfidf, norm='max')
    
    X = np.array([list(x) for x in dataset['tfidf'].values])
    y = dataset['class'].values
    
    run_benchmark_cv(model, X, y, sizes_train=[100, 200, 300, 400],
                     save='results/DISEQuA_svm_tfidf_' + language + '.csv')


# #### SVM + <font color=#007700>TF-IDF</font> + <font color=#0055CC>WB</font>

# In[163]:


for language in ['en', 'es', 'it', 'nl']:
    print('\n\nLanguage: ', language)
    embedding = load_embedding(path_wordembedding + 'wiki.multi.' + language + '.vec')
    dataset = load_disequa(language)
    create_feature('tfidf', dataset, dataset, max_features=2000)
    create_feature('embedding_sum', None, dataset, embedding)
    
    model = {'name': 'svm', 'model': svm_linear}
    
    tfidf = np.array([list(r) for r in dataset['tfidf'].values])
    tfidf = normalize(tfidf, norm='max')
    
    embedding = np.array([list(r) for r in dataset['embedding_sum'].values])
    embedding = normalize(embedding, norm='max')
    
    X = np.array([list(x) + list(xx) for x, xx in zip(tfidf, embedding)])
    y = dataset['class'].values
    
    run_benchmark_cv(model, X, y, sizes_train=[100, 200, 300, 400],
                     save='results/DISEQuA_svm_cortes_' + language + '.csv')


# #### SVM + <font color=#007700>TF-IDF</font> + <font color=#0055CC>WB</font> + <font color=#CC6600>POS</font> + <font color=#CC6600>NER</font>

# In[164]:


for language in ['en', 'es', 'it', 'nl']:
    print('\n\nLanguage: ', language)
    embedding = load_embedding(path_wordembedding + 'wiki.multi.' + language + '.vec')
    dataset = load_disequa(language)
    create_feature('tfidf', dataset, dataset, max_features=2000)
    create_feature('embedding_sum', dataset, dataset, embedding)
    create_feature('pos_hotencode', dataset, dataset)
    create_feature('ner_hotencode', dataset, dataset)
    model = {'name': 'svm', 'model': svm_linear}
    
    tfidf = np.array([list(r) for r in dataset['tfidf'].values])
    tfidf = normalize(tfidf, norm='max')
    
    embedding = np.array([list(r) for r in dataset['embedding_sum'].values])
    embedding = normalize(embedding, norm='max')
    
    pos = np.array([list(r) for r in dataset['pos_hotencode'].values])
    
    ner = np.array([list(r) for r in dataset['ner_hotencode'].values])
    
    X = np.array([list(x) + list(xx) + list(xxx) + list(xxxx) for x, xx, xxx, xxxx in zip(tfidf, embedding, pos, ner)])
    
    y = dataset['class'].values
    
    run_benchmark_cv(model, X, y, sizes_train=[100, 200, 300, 400],
                     save='results/DISEQuA_svm_high_' + language + '.csv')


# ## Old stuffs bellow

# #### CNN

# In[ ]:


# 'en', 'es'
for language in ['es']:
    print('\n\nLanguage: ', language)
    #embedding = load_embedding(path_wordembedding + 'wiki.multi.' + language + '.vec')
    dataset_train, dataset_test = load_uiuc(language)
    text_representation = 'vocab_index'
    vocabulary_inv = create_feature(text_representation, dataset_train, dataset_train)
    create_feature(text_representation, dataset_train, dataset_test)
    model = {'name': 'cnn', 'model': cnn}
    X_train = np.array([list(x) for x in dataset_train[text_representation].values])
    X_test = np.array([list(x) for x in dataset_test[text_representation].values])
    #X_train = pad_sequences(X_train, maxlen=12, dtype='float', padding='post', truncating='post', value=0.0)
    #X_test = pad_sequences(X_test, maxlen=12, dtype='float', padding='post', truncating='post', value=0.0)
    y_train = dataset_train['class'].values
    y_test = dataset_test['class'].values
    ohe = OneHotEncoder()
    y_train = ohe.fit_transform([[y_] for y_ in y_train]).toarray()
    y_test = ohe.transform([[y_] for y_ in y_test]).toarray()
    # , 
    run_benchmark(model, X_train, y_train, X_test, y_test, sizes_train=[1000, 2000, 3000, 4000, 5500],
                  runs=30, save='results/UIUC_cnn_' + language + '.csv', epochs=100, onehot=ohe,
                  vocabulary_size=len(vocabulary_inv))


# #### LSTM + WordEmbedding

# In[73]:


for language in ['es']:
    print('\n\nLanguage: ', language)
    embedding = load_embedding(path_wordembedding + 'wiki.multi.' + language + '.vec')
    dataset_train, dataset_test = load_uiuc(language)
    dataset_train = dataset_train[:100]
    #dataset_test = dataset_test[:10]
    create_feature('embedding', dataset_train, dataset_train, embedding)
    create_feature('embedding', dataset_train, dataset_test, embedding)
    model = {'name': 'lstm', 'model': lstm_default}
    #print(dataset_train['embedding'].values.shape)
    #print(dataset_train['embedding'].values.dtype)
    #print(dataset_test['embedding'].values.shape)
    X_train = np.array([list(x) for x in dataset_train['embedding'].values])
    X_test = np.array([list(x) for x in dataset_test['embedding'].values])
    X_train = pad_sequences(X_train, maxlen=12, dtype='float', padding='post', truncating='post', value=0.0)
    X_test = pad_sequences(X_test, maxlen=12, dtype='float', padding='post', truncating='post', value=0.0)
    y_train = dataset_train['class'].values
    y_test = dataset_test['class'].values
#     y_train_sub = dataset_train['sub_class'].values
#     sub_classes = set()
#     for sc in y_train_sub:
#         sub_classes.add(sc)
#     y_test_sub = dataset_test['sub_class'].values
#     X_test_sub_ = []
#     y_test_sub_ = []
#     for i in range(len(X_test)):
#         if y_train_sub[i] in sub_classes:
#             X_test_sub_.append(X_test[i])
#             y_test_sub_.append(y_train_sub[i])
#     X_test_sub_ = np.array(X_test_sub_)
#     y_test_sub_ = np.array(y_test_sub_)
    ohe = OneHotEncoder()
    y_train = ohe.fit_transform([[y_] for y_ in y_train]).toarray()
    y_test = ohe.transform([[y_] for y_ in y_test]).toarray() 
    run_benchmark(model, X_train, y_train, X_test, y_test, sizes_train=[1000, 2000, 3000, 4000, 5500],
                  runs=30, save='results/UIUC_lstm_embedding_' + language + '_2.csv', epochs=100, onehot=ohe)
    #run_benchmark(model, X_train, y_train_sub, X_test_sub_, y_test_sub_, sizes_train=[1000, 2000, 3000, 4000, 5500],
    #              save='results/UIUCsub_svm_tfidf_' + language + '.csv')


# #### LSTM + BERT

# In[ ]:


for language in ['en']:
    print('\n\nLanguage: ', language)
    #embedding = load_embedding(path_wordembedding + 'wiki.multi.' + language + '.vec')
    dataset_train, dataset_test = load_uiuc(language)
    # debug
    print('WARNING: use subset (first 1000 entries) of training data')
    #dataset_train = dataset_train[:5500].copy()
    
    create_feature('bert', dataset_train, dataset_train)
    create_feature('bert', dataset_train, dataset_test)

    X_train = np.array([x for x in X_train])
    X_test = np.array([x for x in X_test])
    
    #X_train = pad_sequences(X_train, maxlen=12, dtype='float', padding='post', truncating='post', value=0.0)
    #X_test = pad_sequences(X_test, maxlen=12, dtype='float', padding='post', truncating='post', value=0.0)
    y_train = dataset_train['class'].values
    y_test = dataset_test['class'].values
    ohe = OneHotEncoder()
    y_train = ohe.fit_transform([[y_] for y_ in y_train]).toarray()
    y_test = ohe.transform([[y_] for y_ in y_test]).toarray() 
    run_benchmark(model, X_train, y_train, X_test, y_test, sizes_train=[5500],  # 1000, 2000, 3000, 4000, 5500
                  runs=1, save='results/UIUC_lstm_bert_' + language + '.csv',
                  epochs=100, onehot=ohe, in_dim=768)
    #run_benchmark(model, X_train, y_train_sub, X_test_sub_, y_test_sub_, sizes_train=[1000, 2000, 3000, 4000, 5500],
    #              save='results/UIUCsub_svm_tfidf_' + language + '.csv')


# ## DISEQuA Benchmark

# ### RUN DISEQuA Benchmark

# ##### SVM + TFIDF

# In[ ]:


for language in ['DUT', 'ENG', 'ITA', 'SPA']:
    print('\n\nLanguage: ', language)
    dataset = load_disequa(language)
    create_feature('tfidf', dataset, dataset, embedding)
    model = {'name': 'svm', 'model': svm_linear}
    X = np.array([list(x) for x in dataset['tfidf'].values])
    y = dataset['class'].values
    run_benchmark(model, X, y, sizes_train=[100, 200, 300, 400, 405],
                  save='results/DISEQuA_svm_tfidf_' + language + '.csv')


# ##### RFC + TFIDF

# In[ ]:


for language in ['DUT', 'ENG', 'ITA', 'SPA']:
    print('\n\nLanguage: ', language)
    dataset = load_disequa(language)
    create_feature('tfidf', dataset, dataset, embedding)
    model = {'name': 'rfc', 'model': random_forest}
    X = np.array([list(x) for x in dataset['tfidf'].values])
    y = dataset['class'].values
    run_benchmark(model, X, y, sizes_train=[100, 200, 300, 400],
                  save='results/DISEQuA_rfc_tfidf_' + language + '.csv')


# ##### SVM + TFIDF_3gram + SKB

# In[ ]:


for language in ['DUT', 'ENG', 'ITA', 'SPA']:
    print('\n\nLanguage: ', language)
    dataset = load_disequa(language)
    create_feature('tfidf_3gram', dataset, dataset)
    model = {'name': 'svm', 'model': svm_linear}
    X = np.array([list(x) for x in dataset['tfidf'].values])
    y = dataset['class'].values
    skb = SelectKBest(chi2, k=2000).fit(X, y)
    X = skb.transform(X)
    run_benchmark(model, X, y, sizes_train=[100, 200, 300, 400],
                  save='results/DISEQuA_svm_tfidf_3gram_' + language + '.csv')


# ##### LSTM + Embedding

# In[ ]:


for language, embd_l in zip(['SPA'], ['es']):
    print('\n\nLanguage: ', language)
    embedding = load_embedding(path_wordembedding + 'wiki.multi.' + embd_l + '.vec')
    dataset = load_disequa(language)
    create_feature('embedding', dataset, dataset, embedding)
    model = {'name': 'lstm', 'model': lstm_default}
    X = np.array([list(x) for x in dataset['embedding'].values])
    y = dataset['class'].values
    X = pad_sequences(X, maxlen=12, dtype='float', padding='post', truncating='post', value=0.0)
    ohe = OneHotEncoder()
    y = ohe.fit_transform([[y_] for y_ in y]).toarray()
    run_benchmark(model, X, y, sizes_train=[100, 200, 300, 400, 405], onehot=ohe,
                  save='results/DISEQuA_lstm_embedding_' + language + '.csv')


# ##### CNN

# In[ ]:


for language, embd_l in zip(['DUT', 'ENG', 'ITA', 'SPA'], ['nl', 'eng', 'it', 'es']):
    print('\n\nLanguage: ', language)
    #embedding = load_embedding(path_wordembedding + 'wiki.multi.' + embd_l + '.vec')
    dataset = load_disequa(language)
    text_representation = 'vocab_index'
    vocabulary_inv = create_feature(text_representation, dataset, dataset)
    model = {'name': 'cnn', 'model': cnn}
    X = np.array([list(x) for x in dataset[text_representation].values])
    y = dataset['class'].values
    #X = pad_sequences(X, maxlen=12, dtype='float', padding='post', truncating='post', value=0.0)
    ohe = OneHotEncoder()
    y = ohe.fit_transform([[y_] for y_ in y]).toarray()
    run_benchmark(model, X, y, sizes_train=[100, 200, 300, 400], onehot=ohe, vocabulary_size=len(vocabulary_inv),
                  save='results/DISEQuA_cnn_' + language + '.csv', epochs=100)


# In[ ]:




