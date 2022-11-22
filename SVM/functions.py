import numpy as np
import random
import math
from scipy.optimize import minimize

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

train_fun = open_file('bank-note/train.csv')
label = [data[-1] for data in train_fun]

def data_change(data):
    for i in range(len(data)):
        x, y = data[i][:-1], data[i][-1]
        x.append(1)
        x.append(y)
        data[i] = x
    return data

def sign(x):
    if x > 0:
        return 1
    else:
        return -1

def learning_rate_update(sign, gamma, t, a):
    if sign == 1:
        return gamma/(1+(gamma/a)*t)
    if sign == 2:
        return gamma/(1+t)

def loss_function(w, x, C, N):
    return 0.5*np.linalg.norm(w)**2+C*N*max(0, 1-x[-1]*np.inner(w, x[:-1]))

def sub_gradient_update(last_weight, x, C, N, lr):
    comp_weight = list(last_weight[:-1])
    comp_weight.append(0)
    if x[-1]*np.inner(last_weight, x[:-1]) <= 1:
        new_weight = last_weight - lr*np.array(comp_weight) + lr*C*N*x[-1]*np.array(x[:-1])
    else:
        new_weight = (1-lr)*np.array(comp_weight)
    return new_weight

def testing(w, data):
    error = 0
    for i in range(len(data)):
        y_pred = sign(np.inner(w, data[i][:-1]))
        if y_pred != data[i][-1]:
            error += 1
    return error/len(data)

def SVM_primal(train, test, EPOCH, C, gamma, a, signal):
    # Initial parameters
    w = [0 for _ in range(len(train[0]) - 1)]
    # print('train sample: ', train[0])
    LOSS = []
    t = 1
    Train_loss = []
    Test_loss = []
    N = len(train)
    for i in range(EPOCH):
        data = train.copy()
        random.shuffle(data)
        Loss = 0
        for j in range(len(data)):
            learning_rate = learning_rate_update(signal, gamma, t, a)
            w = sub_gradient_update(w, data[j], C, N, learning_rate)
            loss = loss_function(w, data[j], C, N)
            t += 1
            Loss += loss
            LOSS.append(loss)
        train_loss = testing(w, train)
        test_loss = testing(w, test)
        Train_loss.append(train_loss)
        Test_loss.append(test_loss)
    return train_loss, test_loss, w

def constraint(data):
    return np.inner(data, np.array(label))

def data_matrix(data):
    M = [[] for _ in range(len(data))]
    for i in range(len(data)):
        for j in range(len(data)):
            M[i].append(data[i][-1]*data[j][-1]*np.array(data[i][:-1])@np.array(data[j][:-1]))
    return np.array(M)

M = data_matrix(train_fun)

def dual_SVM(alpha):
    p1 = alpha.dot(M)
    p1 = np.dot(p1, alpha)
    p2 = sum(alpha)
    return 0.5*p1-p2

def retrieve_w(alpha, data):
    w = np.zeros(len(data[0][:-1]))
    for i in range(len(alpha)):
        w += alpha[i]*data[i][-1]*np.array(data[i][:-1])
    return w

def retrieve_b(w, data):
    b = 0
    for i in range(len(data)):
        b += data[i][-1] - w.T@data[i][:-1]
    return b/len(data)

def SVM_dual(C, train, test):
    bound = (0, C)
    bounds = tuple([bound for _ in range(len(train))])
    alpha = np.zeros(len(train))
    constrain = {'type': 'eq', 'fun': constraint}
    optimization = minimize(dual_SVM, alpha, method='SLSQP', bounds=bounds, constraints=constrain)

    w = retrieve_w(optimization.x, train)
    train_acc = testing(w, train)
    test_acc = testing(w, test)
    return w, train_acc, test_acc




