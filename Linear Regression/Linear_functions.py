import random
import numpy as np
import math


def open_file(file_path):
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            dataset.append(line.strip().split(','))
    for data in dataset:
        for i in range(len(data)):
            data[i] = float(data[i])
    return dataset

def data_split(dataset):
    X, y = [], []
    for data in dataset:
        X.append(data[0:6])
        y.append(data[-1])
    return X, y

def loss_func(w, dataset):
    loss = 0.5*sum([(data[-1]-np.inner(w, data[0:7]))**2 for data in dataset ])
    return loss

def grad(w, dataset, d):
    grad = []
    for j in range(d):
        grad.append(-sum([(data[-1]-np.inner(w, data[0:7]))*data[j] for data in dataset]))
    return grad

def batch_grad(bond, learning_rate, w, dataset, d):
    loss = []
    while np.linalg.norm(grad(w, dataset, d)) >= bond:
        loss.append(loss_func(w, dataset))
        w = w - [learning_rate*x for x in grad(w, dataset, d)]
    return w, loss

def testing(w, dataset):
    return loss_func(w, dataset)

def SGD_grad(w, i_d, dataset):
    sgd_grad = []
    for i in range(len(dataset[0][:-1])):
        sgd_grad.append(-(dataset[i_d][-1] - np.inner(w, dataset[i_d][:-1]))*dataset[i_d][i])
    return sgd_grad

def SGD(bond, w, learning_rate, dataset):
    SGD_Loss = []
    while np.linalg.norm(SGD_grad(w, random.randint(0, len(dataset)-1), dataset)) >= bond:
        SGD_Loss.append(loss_func(w, dataset))
        w = w - [learning_rate*x for x in SGD_grad(w, random.randint(0, len(dataset)-1), dataset)]
    return w, SGD_Loss

def Optimal(dataset):
    Weight = []
    Label = []
    for data in dataset:
        Weight.append(data[:-1])
        Label.append(data[-1])
    return np.array(Weight).transpose()@np.array(Label)



