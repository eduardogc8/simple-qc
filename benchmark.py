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


from keras.layers import Input, Bidirectional, MaxPooling1D, Flatten, concatenate, GlobalMaxPooling1D, Concatenate
from keras.layers import Dense, Dropout, LSTM, TimeDistributed,Conv1D,Embedding, Reshape, Conv2D, MaxPool2D
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.svm import LinearSVC, SVR
from keras.models import Sequential
from keras.optimizers import Adam
from collections import Counter
from keras.models import Model
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import datetime
import random
import keras
import nltk
import time
import io
import os

# path_wordembedding = '/home/eduardo/word_embedding/'
from download_word_embeddings import muse_embeddings_path, download_if_not_existing

path_wordembedding = muse_embeddings_path
download_if_not_existing()
#path_wordembedding = '/mnt/DATA2/NLP/MUSE_wordembeddings/'

cache = {}

embedding_dt = None
embedding_en = None
embedding_es = None
embedding_it = None
embedding_pt = None


# ### Extract features

# The function *create_features* transform the questions in numerical vector to a classifier model.<br>It returns the output in the df_2 dataframe that is a parameter (*df_2.feature_type*, according to the *feature_type*).<br><br>
# **feature_type:** type of feature. (bow, tfidf, embedding, embedding_sum, vocab_index, pos_index, pos_hotencode, ner_index, ner_hotencode)<br> 
# **df:** the dataframe used to fit the transformers models (df.questions).<br>
# **df_2:** dataframe wich the data will be transformed (df_2.questions).<br>
# **embedding:** embedding model for word embedding features type.<br>
# **max_features:** used in bag-of-words and TFIDF.

# In[3]:


def create_feature(feature_type, df, df_2, embedding=None, max_features=5000):
    
    # Bag of Words
    if feature_type == 'bow':
        model = CountVectorizer(analyzer='word', strip_accents=None, 
                                ngram_range=(1, 1), lowercase=True, 
                                max_features=max_features)
        model.fit(df['question'])
        ret = model.transform(df_2['question']).toarray()
        df_2['bow'] = [x for x in ret]

    # TF-IDF
    if feature_type == 'tfidf':
        model = TfidfVectorizer(analyzer='word', strip_accents=None, 
                                ngram_range=(1, 1), lowercase=True, 
                                max_features=max_features)
        model.fit(df['question'])
        ret = model.transform(df_2['question']).toarray()
        df_2['tfidf'] = [x for x in ret]
    
    # Word embedding (used in LSTM)
    if feature_type == 'embedding':
        if embedding is None:
            print('Error: embedding is None')
            return
        embds = []
        for question in df_2['question']:
            tokens = nltk.word_tokenize(question.replace("¿", ""))
            embed = []
            for token in tokens:
                if token.lower() in embedding:
                    embed.append(embedding[token.lower()])
                else:
                    embed.append(np.zeros(300))
            embds.append(embed)
        df_2['embedding'] = embds
    
    # Word embedding or Sentence embedding (sum the vector)
    if feature_type == 'embedding_sum':
        if embedding is None:
            print('Error: embedding is None')
            return
        embds = []
        model = MeanEmbeddingVectorizer(embedding)
        # model.fit(df['question'])
        questions = [nltk.word_tokenize(question.replace("¿", "")) for question in df_2['question']]
        ret = model.transform(questions)
        df_2['embedding_sum'] = [x for x in ret]
    
    # Vocabulary index (used in CNN)
    if feature_type == 'vocab_index':
        questions_split = [nltk.word_tokenize(q.replace("¿", "")) for q in df['question']]
        questions_split_2 = [nltk.word_tokenize(q.replace("¿", "")) for q in df_2['question']]
        padding_word = "<PAD/>"
        padded_questions = []
        padded_questions_2 = []
        sequence_length = max(len(x) for x in questions_split)
        for i in range(len(questions_split)):
            question = questions_split[i]
            num_padding = sequence_length - len(question)
            new_question = question + [padding_word] * num_padding
            padded_questions.append(new_question)
        word_counts = Counter(itertools.chain(*padded_questions))
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv = list(sorted(vocabulary_inv))
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        for i in range(len(questions_split_2)):
            question = questions_split_2[i]
            nq = []
            for w in question:
                if w in vocabulary:
                    nq.append(w)
                else:
                    nq.append(padding_word)
            question = nq
            num_padding = sequence_length - len(question)
            new_question = question + [padding_word] * num_padding
            padded_questions_2.append(new_question)
        
        df_2['vocab_index'] = [[vocabulary[word] for word in question] for question in padded_questions_2]
        return vocabulary_inv
    
    # Postag Index
    if feature_type == 'pos_index':
        MAX_WORDS = 20
        if type(df.pos[0]) == str:
            new_pos = []
            for pos_vec in df.pos:
                new_pos.append(eval(pos_vec))
            df.pos = new_pos
        if type(df_2.pos[0]) == str:
            new_pos = []
            for pos_vec in df_2.pos:
                new_pos.append(eval(pos_vec))
            df_2.pos = new_pos
        
        pos2idx = {'X':0}
        for row in df.pos.values:
            for pos in row:
                if pos not in pos2idx:
                    pos2idx[pos] = len(pos2idx)

        ret = []
        for pos_vec in df_2.pos.values:
            r = []
            for i in range(MAX_WORDS):
                if i < len(pos_vec):
                    if pos_vec[i] in pos2idx:
                        r.append(pos2idx[pos_vec[i]])
                    else:
                        r.append(0)
                else:
                    # PAD
                    r.append(len(pos2idx))
            ret.append(r)
        df_2['pos_index'] = ret
    
    # Postag hot-encode
    if feature_type == 'pos_hotencode':
        MAX_WORDS = 20
        if type(df.pos.values[0]) == str:
            new_pos = []
            for pos_vec in df.pos:
                new_pos.append(eval(pos_vec))
            df.pos = new_pos
        if type(df_2.pos.values[0]) == str:
            new_pos = []
            for pos_vec in df_2.pos:
                new_pos.append(eval(pos_vec))
            df_2.pos = new_pos
        
        pos2idx = {'X':0}
        for row in df.pos.values:
            for pos in row:
                if pos not in pos2idx:
                    pos2idx[pos] = len(pos2idx)
        identityPos = np.identity(len(pos2idx))
        
        ret = []
        for pos_vec in df_2.pos.values:
            r = []
            for i in range(MAX_WORDS):
                if i < len(pos_vec):
                    if pos_vec[i] in pos2idx:
                        r += list(identityPos[pos2idx[pos_vec[i]]])
                    else:
                        r += list(identityPos[0])
                else:
                    # PAD
                    r += list(np.zeros(len(pos2idx)))
            ret.append(r)
        df_2['pos_hotencode'] = ret
    
    # Named entity recognition Index
    if feature_type == 'ner_index':
        MAX_WORDS = 20
        if type(df.ner.values[0]) == str:
            new_ner = []
            for ner_vec in df.ner:
                new_ner.append(eval(ner_vec))
            df.ner = new_ner
        if type(df_2.ner.values[0]) == str:
            new_ner = []
            for ner_vec in df_2.ner:
                new_ner.append(eval(ner_vec))
            df_2.ner = new_ner
        
        ner2idx = {'':0}
        for row in df.ner.values:
            for ner in row:
                if ner not in ner2idx:
                    ner2idx[ner] = len(ner2idx)

        ret = []
        for ner_vec in df_2.ner.values:
            r = []
            for i in range(MAX_WORDS):
                if i < len(ner_vec):
                    if ner_vec[i] in ner2idx:
                        r.append(ner2idx[ner_vec[i]])
                    else:
                        r.append(0)
                else:
                    # PAD
                    r.append(len(ner2idx))
            ret.append(r)
        df_2['ner_index'] = ret
    
    # Named entity recognition Hot-encode
    if feature_type == 'ner_hotencode':
        MAX_WORDS = 20
        if type(df.ner.values[0]) == str:
            new_ner = []
            for ner_vec in df.ner:
                new_ner.append(eval(ner_vec))
            df.ner = new_ner
        if type(df_2.ner.values[0]) == str:
            new_ner = []
            for ner_vec in df_2.ner:
                new_ner.append(eval(ner_vec))
            df_2.ner = new_ner
        
        ner2idx = {'X':0}
        for row in df.ner.values:
            for ner in row:
                if ner not in ner2idx:
                    ner2idx[ner] = len(ner2idx)
        identityNer = np.identity(len(ner2idx))
        
        ret = []
        for ner_vec in df_2.ner.values:
            r = []
            for i in range(MAX_WORDS):
                if i < len(ner_vec):
                    if ner_vec[i] in ner2idx:
                        r += list(identityNer[ner2idx[ner_vec[i]]])
                    else:
                        r += list(identityNer[0])
                else:
                    # PAD
                    r += list(np.zeros(len(ner2idx)))
            ret.append(r)
        df_2['ner_hotencode'] = ret

    if feature_type == 'bert':
        # requires: pip install pytorch-transformers
        import torch
        
        bert_model_name = 'bert-base-multilingual-cased'
        # Load pretrained model/tokenizer
        if 'bert_tokenizer' not in cache:
            print('load bert_tokenizer...')
            from pytorch_transformers import BertTokenizer
            cache['bert_tokenizer'] = BertTokenizer.from_pretrained(bert_model_name)
        if 'bert_model' not in cache: 
            print('load bert_model...')
            from pytorch_transformers import BertModel
            cache['bert_model'] = BertModel.from_pretrained(bert_model_name)

        # Encode text
        print('tokenize...')
        input_ids = [cache['bert_tokenizer'].encode(s) for s in df_2['question']]
        input_ids_padded = pad_sequences(input_ids, maxlen=12, dtype='int64', padding='post', truncating='post', value=0)
        input_ids_tensor = torch.tensor(input_ids_padded)
        print('embed with BERT...')
        with torch.no_grad():
            ret = []
            batch_size = 32
            for ind in tqdm(range(0, len(input_ids), batch_size)):
                batch_input = input_ids_tensor[ind:ind+batch_size]
                last_hidden_states = cache['bert_model'](batch_input)[0]  # Models outputs are now tuples
                ret.append(last_hidden_states.numpy())
            ret = np.concatenate(ret, axis=0)
            print(f'shape of encoded input: {ret.shape}')
            df_2['bert'] = [enc for enc in ret]
            

class MeanEmbeddingVectorizer(object):
    def __init__(self, word_embedding):
        self.word_embedding = word_embedding
        self.dim = len(list(embedding.values())[0])

    def fit(self, X, y):
        return self

    def transform(self, X):
        ret = np.array([
            np.sum([self.word_embedding[w] for w in words if w in self.word_embedding]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
        return ret


# ### Create classifier models

# The models are created through functions that return them. These functions will be used to create a new model in each experiment. Therefore, an instance of a model is created by the benchmark function and not explicitly in a code block.

# In[5]:


def svm_linear():
    svc = LinearSVC(C=1.0)
    return svc

def lstm_default(in_dim=300, out_dim=7, drop=0.2):
    model = Sequential()
    model.add(LSTM(256, input_dim=in_dim, name='0_LSTM'))
    model.add(Dropout(drop, name='1_Droupout'))
    model.add(Dense(128, activation='relu', name='2_Dense'))
    model.add(Dropout(drop, name='3_Droupout'))
    model.add(Dense(out_dim, activation='softmax', name='4_Dense'))
    #otimizer = keras.optimizers.Adam(lr=0.01) #decay = 0.0001
    #model.compile(optimizer=otimizer, loss='categorical_crossentropy')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    return model

def random_forest():
    return RandomForestClassifier(n_estimators=500)

def mlp(in_dim=5000, out_dim=7, drop=0.65):
    model = Sequential()
    model.add(Dense(128, input_dim=in_dim, activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(out_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def cnn(sequence_length, vocabulary_size, embedding_dim=300, filter_sizes=[3,4,5], num_filters=512, drop=0.5, out_dim=7):
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    reshape = Reshape((sequence_length,embedding_dim,1))(embedding)
    
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
    
    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=out_dim, activation='softmax')(dropout)
    
    model = Model(inputs=inputs, outputs=output)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    
    return model
    


# ### UTILS

# In[6]:


import warnings
warnings.filterwarnings("ignore")


# In[7]:


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


# #### Load UIUC dataset

# In[8]:


def load_uiuc(language):
    # language: 'en', 'pt' or 'es'
    return pd.read_csv('datasets/UIUC_' + language + '/train_features.csv'), pd.read_csv('datasets/UIUC_' + language + '/test_features.csv')


# #### Load DISEQuA dataset

# In[9]:


def load_disequa(language):
    df = pd.read_csv('datasets/DISEQuA/disequa_features.csv')
    return df[df['language'] == language]


# ## Benchmark UIUC - Normal

# **Normal:** it uses the default fixed split of UIUC between train dataset (at last 5500 instances) and test dataset (500 instances). Therefore, it does not use cross-validation.

# When the *run_benchmark* function is executed, it will save each result in the *save* path.
# 
# **model:** a dictionary with the classifier name and the function to create and return the model (not an instance of the model). <br> Example: *model = {'name': 'SVM', 'model': svm_linear}*<br>
# **X:** all the training set.<br>
# **y:** all the labels of the training set.<br>
# **x_test:** test set.<br>
# **y_test:** labels of the test set.<br>
# **sizes_train:** sizes of training set. For each size, an experiment is executed.<br>
# **runs:** number of time that each experiment is executed (used in models which has parameters with random values, like weights in an ANN).<br>
# **save:** csv path where the results will be saved.<br>
# **metric_average:** used in f1, recall and precision metrics<br>
# **onehot:** one-hot model to transform labels.<br>
# **out_dim:** the total of classes for ANN models.<br>
# **epochs:** epochs for ANN models.<br>
# **batch_size:** batch_size for ANN models.<br>
# **vocabulary_size:** vocabulary size (used in CNN model).
# 

# In[53]:


def run_benchmark(model, X, y, x_test, y_test, sizes_train, runs=30, save='default.csv', 
                  metric_average="macro", onehot=None, out_dim=6, epochs=10, batch_size=30, 
                  vocabulary_size=5000, in_dim=300):
    start_benchmark = time.time()
    results = pd.DataFrame()

    for size_train in sizes_train:

        print('\n'+str(size_train), end='|')

        for run in range(runs):
            print('.', end='')
            x_train = X[:size_train]
            y_train = y[:size_train]

            if 'lstm' in model['name'] or 'mlp' in model['name']:
                m = model['model'](in_dim=in_dim, out_dim=len(onehot.categories_[0]))
                start_time = time.time()
                m.fit(x_train, y_train, verbose=0, epochs=epochs, batch_size=batch_size)
                train_time = time.time() - start_time
                start_time = time.time()
                result = m.predict(x_test)
                test_time = time.time() - start_time
                result = np.nan_to_num(result)
                result = onehot.inverse_transform(result)
                y_test_ = onehot.inverse_transform(y_test)
            elif 'cnn' in model['name']:
                sequence_length = x_train.shape[1]
                out_dim = len(onehot.categories_[0])
                m = model['model'](sequence_length, vocabulary_size, out_dim=out_dim)
                start_time = time.time()
                m.fit(x_train, y_train, verbose=0, epochs=epochs, batch_size=batch_size)
                train_time = time.time() - start_time
                start_time = time.time()
                result = m.predict(x_test)
                test_time = time.time() - start_time
                result = np.nan_to_num(result)
                result = onehot.inverse_transform(result)
                y_test_ = onehot.inverse_transform(y_test)
            else:
                m = model['model']()
                start_time = time.time()
                m.fit(x_train, y_train)
                train_time = time.time() - start_time

                start_time = time.time()
                result = m.predict(x_test)
                test_time = time.time() - start_time
                y_test_ = y_test

            data = {'datetime': datetime.datetime.now(),
                    'model': model['name'],
                    'accuracy': accuracy_score(result, y_test_),
                    'precision': precision_score(result, y_test_, average=metric_average),
                    'recall': recall_score(result, y_test_, average=metric_average),
                    'f1': f1_score(result, y_test_, average=metric_average),
                    'mcc': matthews_corrcoef(result, y_test_),
                    'confusion': confusion_matrix(result, y_test_),
                    'run': run + 1,
                    'train_size': size_train,
                    'execution_time': train_time,
                    'test_time': test_time}
            results = results.append([data])
            results.to_csv(save)
    print('')
    aux = time.time() - start_benchmark
    print('Run time benchmark:', aux)
    return pd.DataFrame(results)


# ## Benchmark UIUC and DISEQuA - Cross-validation

# **Cross-validation:** instead of uses default fixed splits, it uses the all the dataset with cross-validation with 10 folds.

# When the *run_benchmark* function is executed, it will save each result in the *save* path.
# 
# **model:** a dictionary with the classifier name and the function to create and return the model (not an instance of the model). <br> Example: *model = {'name': 'SVM', 'model': svm_linear}*<br>
# **X:** Input features.<br>
# **y:** Input labels.<br>
# **sizes_train:** sizes of training set. For each size, an experiment is executed.<br>
# **folds:** Amount of folds for cross-validations.<br>
# **save:** csv path where the results will be saved.<br>
# **metric_average:** used in f1, recall and precision metrics<br>
# **onehot:** one-hot model to transform labels.<br>
# **epochs:** epochs for ANN models.<br>
# **batch_size:** batch_size for ANN models.<br>
# **vocabulary_size:** vocabulary size (used in CNN model).
# 

# In[11]:


def run_benchmark_cv(model, X, y, sizes_train, folds=10, save='default.csv', metric_average="macro",
                  onehot=None, epochs=10, batch_size=30, vocabulary_size=5000):
    start_benchmark = time.time()
    results = pd.DataFrame()
    for size_train in sizes_train:
        print('\n'+str(size_train)+'|', end='')
        size_test = len(X) - size_train
        # StratifiedShuffleSplit maybe use it insted
        rs = StratifiedShuffleSplit(n_splits=folds, train_size=size_train, 
                                    test_size=size_test, random_state=1)
        fold = 0
        for train_indexs, test_indexs in rs.split(X, y):
            print('.', end='')
            x_train = X[train_indexs]
            y_train = y[train_indexs]
            x_test = X[test_indexs]
            y_test = y[test_indexs]

            if 'lstm' in model['name']:
                m = model['model']()
                start_time = time.time()
                m.fit(x_train, y_train, verbose=0, epochs=epochs)
                train_time = time.time() - start_time
                start_time = time.time()
                result = m.predict(x_test)
                test_time = time.time() - start_time
                result = onehot.inverse_transform(result)
                y_test = onehot.inverse_transform(y_test)
            elif 'cnn' in model['name']:
                sequence_length = x_train.shape[1]
                out_dim = len(onehot.categories_[0])
                m = model['model'](sequence_length, vocabulary_size, out_dim=out_dim)
                start_time = time.time()
                m.fit(x_train, y_train, verbose=0, epochs=epochs, batch_size=batch_size)
                train_time = time.time() - start_time
                start_time = time.time()
                result = m.predict(x_test)
                test_time = time.time() - start_time
                result = np.nan_to_num(result)
                result = onehot.inverse_transform(result)
                y_test = onehot.inverse_transform(y_test)
            else:
                m = model['model']()
                start_time = time.time()
                m.fit(x_train, y_train)
                train_time = time.time() - start_time

                start_time = time.time()
                result = m.predict(x_test)
                test_time = time.time() - start_time
            
                
            data = {'datetime': datetime.datetime.now(),
                    'accuracy': accuracy_score(result, y_test),
                    'precision': precision_score(result, y_test, average=metric_average),
                    'recall': recall_score(result, y_test, average=metric_average),
                    'f1': f1_score(result, y_test, average=metric_average),
                    'mcc': matthews_corrcoef(result, y_test),
                    'confusion': confusion_matrix(result, y_test),
                    'train_size': size_train,
                    'fold': fold,
                    'execution_time': train_time,
                    'test_time': test_time}
            results = results.append([data])
            results.to_csv(save)
            fold += 1
    print('')
    aux = time.time() - start_benchmark
    print('Run time benchmark:', aux)
    return pd.DataFrame(results)


# ## Run UIUC Benchmark - Normal

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
    
    run_benchmark_cv(model, X, y, sizes_train=[100,200,300,400],
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
    
    run_benchmark_cv(model, X, y, sizes_train=[100,200,300,400],
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
    
    run_benchmark_cv(model, X, y, sizes_train=[100,200,300,400],
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
    model = {'name': 'lstm', 'model': lstm_default}
    X_train = dataset_train['bert'].values
    X_test = dataset_test['bert'].values
    
    X_train = np.array([x for x in X_train])
    X_test = np.array([x for x in X_test])
    
    #X_train = pad_sequences(X_train, maxlen=12, dtype='float', padding='post', truncating='post', value=0.0)
    #X_test = pad_sequences(X_test, maxlen=12, dtype='float', padding='post', truncating='post', value=0.0)
    y_train = dataset_train['class'].values
    y_test = dataset_test['class'].values
    ohe = OneHotEncoder()
    y_train = ohe.fit_transform([[y_] for y_ in y_train]).toarray()
    y_test = ohe.transform([[y_] for y_ in y_test]).toarray() 
    run_benchmark(model, X_train, y_train, X_test, y_test, sizes_train=[5500], # 1000, 2000, 3000, 4000, 5500
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
    run_benchmark(model, X, y, sizes_train=[100,200,300,400,405],
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
    run_benchmark(model, X, y, sizes_train=[100,200,300,400],
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
    run_benchmark(model, X, y, sizes_train=[100,200,300,400],
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
    run_benchmark(model, X, y, sizes_train=[100,200,300,400,405], onehot=ohe,
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
    run_benchmark(model, X, y, sizes_train=[100,200,300,400], onehot=ohe, vocabulary_size=len(vocabulary_inv),
                  save='results/DISEQuA_cnn_' + language + '.csv', epochs=100)


# In[ ]:




