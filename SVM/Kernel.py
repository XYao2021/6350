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
    return data

def sign(x):
    if x > 0:
        return 1
    else:
        return -1

def constraint(data):
    return np.inner(data, np.array(label))

def kernel_function(xi, xj, gamma):
    return math.e**(-(np.linalg.norm(np.array(xi)-np.array(xj))**2)/gamma)

def dual_SVM_kernel(alpha):
    p1 = alpha.dot(Mk)
    p1 = np.dot(p1, alpha)
    p2 = sum(alpha)
    return 0.5*p1-p2

def data_matrix_kernel(data, gamma):
    M = [[] for _ in range(len(data))]
    for i in range(len(data)):
        for j in range(len(data)):
            M[i].append(data[i][-1]*data[j][-1]*kernel_function(data[i][:-1], data[j][:-1], gamma))
    return np.array(M)

def testing_kernel(alpha, data, target, gamma):
    error = 0
    for i in range(len(target)):
        pred = 0
        for j in range(len(data)):
            pred += alpha[j]*data[j][-1]*kernel_function(data[j][:-1], target[i][:-1], gamma)
        if sign(pred) != target[i][-1]:
            error += 1
    return error/len(target)

def count_support_vectors(alpha):
    vector_id = []
    for i in range(len(alpha)):
        if alpha[i] != 0:
            vector_id.append(i)
    return vector_id

def find_overlap(vec0, vec1):
    overlap = [i for i in vec0 if i in vec1]
    return len(overlap)

warnings.filterwarnings("ignore", category=RuntimeWarning)
train_org = open_file('bank-note/train.csv')
test_org = open_file('bank-note/test.csv')
label = [data[-1] for data in train_org]

train = data_change(train_org)
test = data_change(test_org)

a = 1
Set_gamma = [0.1, 0.5, 1, 5, 100]
signal = 1

# Hyper-parameters
Set_C = [100/873, 500/873, 700/873]
EPOCH = 100

for C in Set_C:
    bound = (0, C)
    bounds = tuple([bound for _ in range(len(train))])
    alpha = np.zeros(len(train))
    constrain = {'type': 'eq', 'fun': constraint}
    vec_set = []
    Train_Acc = []
    Test_Acc = []
    for gamma in Set_gamma:
        Mk = data_matrix_kernel(train_org, gamma)
        optimization = minimize(dual_SVM_kernel, alpha, method='SLSQP', bounds=bounds, constraints=constrain)

        vec_set.append(count_support_vectors(optimization.x))
        train_acc = testing_kernel(optimization.x, train, train, gamma)
        test_acc = testing_kernel(optimization.x, train, test, gamma)
        Train_Acc.append(train_acc)
        Test_Acc.append(test_acc)
    overlap = []
    for i in range(len(vec_set)-1):
        overlap.append(find_overlap(vec_set[i], vec_set[i+1]))
    print(C, 'Gamma Set: ', Set_gamma)
    print(C, 'Train Acc: ', Train_Acc)
    print(C, 'Test Acc: ', Test_Acc)
    print(C, 'Vec Number: ', [len(i) for i in vec_set])
    print(C, 'overlap: ', overlap, '\n')


