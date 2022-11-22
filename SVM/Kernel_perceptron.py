import numpy as np
import random
import math
from scipy.optimize import minimize
import warnings

def open_file(file_path):
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            dataset.append(line.strip().split(','))
    for data in dataset:
        for i in range(len(data)):
            data[i] = float(data[i])
        if data[-1] == 0.0:
            data[-1] = -1
    return dataset

def data_change(data):
    for i in range(len(data)):
        x, y = data[i][:-1], data[i][-1]
        x.append(1)
        x.append(y)
        data[i] = x
        data[i].append(0)
    return data

def sign(x):
    if x > 0:
        return 1
    else:
        return -1

def kernel_function(xi, xj, gamma):
    return math.e**(-(np.linalg.norm(np.array(xi)-np.array(xj))**2)/gamma)

def prediction_perceptron(dataset, sample, gamma):
    return sign(sum([data[-1]*data[-2]*kernel_function(data[:-2], sample, gamma) for data in dataset]))

def testing(dataset, test, gamma):
    error = 0
    for i in range(len(test)):
        if prediction_perceptron(dataset, test[i][:-2], gamma) != test[i][-2]:
            error += 1
    return round(error/len(test), 4)

warnings.filterwarnings("ignore", category=RuntimeWarning)
train_org = open_file('bank-note/train.csv')
test_org = open_file('bank-note/test.csv')
label = [data[-1] for data in train_org]

train = data_change(train_org)
test = data_change(test_org)

Set_gamma = [0.1, 0.5, 1, 5, 100]
# Set_gamma = [0.1]

for gamma in Set_gamma:
    train_set = train.copy()
    for i in range(len(train_set)):
        if prediction_perceptron(train_set, train_set[i][:-2], gamma) != train_set[i][-2]:
            train[i][-1] += 1
    train_acc = testing(train_set, train, gamma)
    test_acc = testing(train_set, test, gamma)
    print(gamma, 'train acc: ', train_acc)
    print(gamma, 'test acc: ', test_acc, '\n')


