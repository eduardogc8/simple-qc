import pandas as pd


# EN
data = []
for line in open('UIUC_en/train_5500.label.txt', 'r'):
    class_ = line[:line.index(':')].replace('\n', '').strip()
    sub_class = line[line.index(':')+1:line.index(' ')].replace('\n', '').strip()
    question = line[line.index(' ')+1:].replace('\n', '').strip()
    d = {'question': question, 'class': class_, 'sub_class': sub_class}
    data.append(d)

df = pd.DataFrame(data)
df.to_csv('UIUC_en/train.csv')

data = []
for line in open('UIUC_en/TREC_10.label.txt', 'r'):
    class_ = line[:line.index(':')].replace('\n', '').strip()
    sub_class = line[line.index(':')+1:line.index(' ')].replace('\n', '').strip()
    question = line[line.index(' ')+1:].replace('\n', '').strip()
    d = {'question': question, 'class': class_, 'sub_class': sub_class}
    data.append(d)

df = pd.DataFrame(data)
df.to_csv('UIUC_en/test.csv')


# PT
data = []
for line in open('UIUC_pt/train.txt', 'r', encoding='utf-8'):
    class_ = line[:line.index(':')].replace('\n', '').strip()
    sub_class = line[line.index(':')+1:line.index(' ')].replace('\n', '').strip()
    question = line[line.index(' ')+1:].replace('\n', '').strip()
    d = {'question': question, 'class': class_, 'sub_class': sub_class}
    data.append(d)

df = pd.DataFrame(data)
df.to_csv('UIUC_pt/train.csv')

data = []
for line in open('UIUC_pt/test.txt', 'r'):
    class_ = line[:line.index(':')].replace('\n', '').strip()
    sub_class = line[line.index(':')+1:line.index(' ')].replace('\n', '').strip()
    question = line[line.index(' ')+1:].replace('\n', '').strip()
    d = {'question': question, 'class': class_, 'sub_class': sub_class}
    data.append(d)

df = pd.DataFrame(data)
df.to_csv('UIUC_pt/test.csv')


# ES
data_train = []
data_test = []
for i, line in enumerate(open('UIUC_es/Clasificacion-QA-6305.label_.txt', 'r')):
    class_ = line[:line.index(':')].replace('\n', '').strip()
    sub_class = line[line.index(':')+1:line.index(' ')].replace('\n', '').strip()
    question = line[line.index(' ')+1:].replace('\n', '').strip()
    d = {'question': question, 'class': class_, 'sub_class': sub_class}
    if i >= 5452:
        data_test.append(d)
    else:
        data_train.append(d)

df = pd.DataFrame(data_train)
df.to_csv('UIUC_es/train.csv')
df = pd.DataFrame(data_test)
df.to_csv('UIUC_es/test.csv')