from math import log
import arff
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

def train(x_train, y_train):
    #frequency = sum of frequency of equivalent number / total number of each class
    d = {}
    classes = set(y_train)
    for cl in classes:
        d[cl] = {}
        x_c = x_train[y_train == cl]
        m, n = x_c.shape
        for j in range(n):
            x_class = set(x_c[:, j])
            d[cl][j] = {}
            for x in x_class:
                f = 0
                for t in range(len(x_c[:, j])):
                    if x == x_c[t, j]:
                        f += 1
                d[cl][j][x] = f
    return d


def test(x_test, d):
    s = []
    for x in x_test:
        c = []
        for key, value in d.items():
            p = 0
            for i, j in enumerate(x):
                for k in range(len(value)):
                    if i == k:
                        if j not in value[k]:
                            num = 1e-5
                        else:
                            num = value[k][j]
                        class_num = sum([v for k, v in value[i].items()])
                        p += log(num/class_num)
            c.append(p)
        s.append(np.argmax(c))
    return s


def load_data(name, train_size):
    #Eden's load data function
    data = arff.load(open('%s.arff' % name, 'r'))
    data = np.array(pd.DataFrame(data['data']).dropna())
    label_encoder = LabelEncoder()
    encoded = np.zeros(data.shape)
    for i in range(len(data[0])):
        encoded[:, i] = label_encoder.fit_transform(data[:, i])
    x, y = encoded[:, :-1], encoded[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=train_size)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    for name in ['weather', 'soybean']:
        x_train, x_test, y_train, y_test = load_data(name, 0.7)
        d = train(x_train, y_train)
        s = test(x_test, d)
        acc = accuracy_score(s, y_test)
        print("accuracy score:")
        print(name, acc)
