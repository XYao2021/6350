import numpy as np
import matplotlib.pyplot as plt
from functions import *


train_dataset = open_bank('bank/train.csv')
test_dataset = open_bank('bank/test.csv')

features, counts = get_features_and_num(dataset=train_dataset)
features, counts, train_dataset = binary_feature_trans(features, counts, train_dataset)
train_dataset = Add_org_weights(train_dataset)
# print(dic.keys())
# for key in dic.keys():
#     # print(train_dataset[0])
#     # print(list(key))
#     if train_dataset[0][15] == list(key)[15]:
#         print(train_dataset[0], list(dic[key])[0])
# print(len(train_dataset), len(train_dataset[0]))

dataset = train_dataset[:20]
print(dataset)




