import random
import numpy as np

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
    new_data = [data[:-1] for data in dataset]
    label = []
    for data in dataset:
        if data[-1] == 0:
            label.append(-1)
        else:
            label.append(1)
    return new_data, label

def sgn(x):
    if x > 0:
        return 1
    elif x <= 0:
        return -1

def testing(w, b, dataset):
    data, label = data_split(dataset)
    error = 0
    for i in range(len(data)):
        pred = np.array(w)@data[i] + b
        if sgn(pred) != label[i]:
            error += 1
    return error/len(dataset)

def standard_perceptron(train, test, EPOCH, LR, w, b):
    for iter in range(EPOCH):
        random.shuffle(train)
        train_dataset, label = data_split(dataset=train)
        # print(iter, train[0], train_dataset[0], label[0])
        for i in range(len(label)):
            pred = np.array(w).T @ np.array(train_dataset[i]) + b
            # print(pred, label[i])
            if label[i] * pred <= 0:
                # print(label[i], train_dataset[i])
                w = w + LR * label[i] * np.array(train_dataset[i])
                b = b + LR * label[i] * 1
        test_error = testing(w, b, test)
        print(iter, w, b, test_error, '\n')

def vote_testing(W, dataset):
    data, label = data_split(dataset)
    error = 0
    for i in range(len(data)):
        sign = 0
        for Wi in W:
            w, b, Cm = Wi
            # print(w, b, Cm)
            # print(np.array(w)@np.array(data[i])+b, '\n')
            sign += Cm*sgn(np.array(w)@np.array(data[i])+b)
        # print(i, label[i], sgn(sign), '\n')
        if label[i] != sgn(sign):
            error += 1
    # print(error)
    return error/len(dataset)

def vote_perceptron(train, test, EPOCH, LR, w, b):
    Cm = 0  # number of predictions made by Wm (correct number)
    for iter in range(EPOCH):
        W = []
        train_dataset, label = data_split(train)
        for i in range(len(label)):
            pred = np.array(w).T @ np.array(train_dataset[i]) + b
            if label[i] * pred <= 0:
                W.append([w, b, Cm])
                w = w + LR * label[i] * np.array(train_dataset[i])
                b = b + LR * label[i] * 1
                Cm = 1
            else:
                Cm += 1
        Test_loss = vote_testing(W, test)
        print(W)
        print(Test_loss, '\n')

def Average_testing(W, dataset):
    data, label = data_split(dataset)
    error = 0
    for i in range(len(data)):
        sign = 0
        for Wi in W:
            w, b, Cm = Wi
            # print(w, b, Cm)
            # print(np.array(w)@np.array(data[i])+b, '\n')
            sign += Cm*(np.array(w)@np.array(data[i])+b)
        # print(i, label[i], sgn(sign), '\n')
        if label[i] != sgn(sign):
            error += 1
    # print(error)
    return error/len(dataset)

def Average_perceptron(train, test, EPOCH, LR, w, b):
    a = np.array([float(0) for i in range(len(train[0][:-1]))])
    B = 0
    for iter in range(EPOCH):
        train_dataset, label = data_split(train)
        for i in range(len(label)):
            pred = np.array(w).T @ np.array(train_dataset[i]) + b
            if label[i] * pred <= 0:
                w = w + LR * label[i] * np.array(train_dataset[i])
                b = b + LR * label[i] * 1
            a += w
            B += b
        print(iter, a, B)
        test_loss = testing(a, B, test)
        print(iter, test_loss, '\n')

