import json
import pandas as pd


path_train = 'qald-9-train-multilingual.json'
path_test = 'qald-9-test-multilingual.json'

train = json.load(open(path_train, 'r', encoding='utf-8'))
test = json.load(open(path_test, 'r', encoding='utf-8'))


def make_csv(j_dic):
    data = []
    for questions in j_dic['questions']:
        for question in questions['question']:
            d = {}
            d['id'] = questions['id']
            if 'string' not in question:
                continue
            d['question'] = question['string']
            d['class'] = questions['answertype']
            d['language'] = question['language']
            data.append(d)
    df = pd.DataFrame(data)
    return df


df_train = make_csv(train)
df_train.to_csv('train.csv')
df_train = make_csv(test)
df_train.to_csv('test.csv')
