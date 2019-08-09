import itertools
from collections import Counter

import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from bert_embeddings import embedd_with_bert_using_df, load_bert_to_cache

cache = {}

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
        load_bert_to_cache(cache)
        embedd_with_bert_using_df(df_2, cache['bert_tokenizer'], cache['bert_model'])


class MeanEmbeddingVectorizer(object):
    def __init__(self, word_embedding):
        self.word_embedding = word_embedding
        self.dim = len(list(word_embedding.values())[0])

    def fit(self, X, y):
        return self

    def transform(self, X):
        ret = np.array([
            np.sum([self.word_embedding[w] for w in words if w in self.word_embedding]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
        return ret


