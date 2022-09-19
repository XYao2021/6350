import random
import numpy as np
from functions import *


"label in the last column of each dataset"
# train_dataset = open_file('car/train.csv')
# test_dataset = open_file('car/test.csv')

train_dataset = open_file('bank/train.csv')
test_dataset = open_file('bank/test.csv')

tree_depth = 5

dic = training(train_dataset, tree_depth)

# test_data = test_dataset[435]
# print(test_data)
# B = []
# A = []
a = testing(train_dataset, dic)
b = testing(test_dataset, dic)
print(a)
print(b)
# print(dic[A[B.index(max(B))]][0])

# label = dic[A.index(B.index(max(B)))][0]
# print(test_data)
# print(dic[A[-1]][0])
# print(len(dic))
# print(test_dataset[6][:-1])
# A = []

# test_data = test_dataset[7]
# print(test_data)
# for key in dic.keys():
#     pred, order = dic[key]
#     print(order)
#     a = [key for i in order if test_data[i] == list(key)[i]]
#     print(test_data)
#     print(a, '\n')

# correct = 0
# for data in test_dataset:
#     test = data[:-1]
#     label = data[-1]
#     A = []
#     for key in dic.keys():
#         pred, order = dic[key]
#         a = [key for i in order if test[i] == list(key)[i]]
#         # print(a, '\n')
#         A.append(a)
#     A = [i for i in A if i != []]
#     print(len(data), data)
#     print(test_dataset.index(data), A, '\n')
# #     pred = dic[list(dic.keys())[A.index(max(A))]]
#     if pred == label:
#         correct += 1
#     B = []
#     m = max(A)
#     for i in range(len(A)):
#         if A[i] == m:
#             B.append(i)
#     print(len(B), B, '\n')
    # print(pred, label)
# print(correct/len(test_dataset))

# for key in dic.keys():
#     print(key)
# for test_data in test_dataset:
#     label = test_data[-1]
#     features = test_data[:-1]
#     print(features)
#     print(label, '\n')
