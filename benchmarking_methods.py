import datetime
import time
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit


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
            pred_train = None

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


                def pred_target_transform_fun(x,target):
                    start_time = time.time()
                    prediction = m.predict(x)
                    duration = time.time() - start_time
                    return prediction,target,duration

                result,y_test_,test_time = pred_target_transform_fun(x_test,y_test)
                pred_train,y_train_,_ = pred_target_transform_fun(x_train,y_train)


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
            if pred_train is not None:
                data['f1_train']= f1_score(pred_train, y_train_, average=metric_average)
            pprint(data)
            results = results.append([data])
            results.to_csv(save)
    aux = time.time() - start_benchmark
    print('Run time benchmark:', aux)
    return pd.DataFrame(results)


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