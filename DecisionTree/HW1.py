import random
import numpy as np
from functions import *


"label in the last column of each dataset"
train_dataset = open_file('car/train.csv')
test_dataset = open_file('car/test.csv')

features = get_features(dataset=test_dataset)
print(features)

counts = get_num_for_features(features=features,
                              dataset=train_dataset)
print(counts)

entropy = []
for count in counts:
    entropy.append(Entropy(count))
print(features[entropy.index(max(entropy))], entropy.index(max(entropy)))

target = features[entropy.index(max(entropy))]
new_train_dataset = [[] for _ in range(len(target))]
for data in train_dataset:
    for tar in target:
        if data[entropy.index(max(entropy))] == tar:
            new_train_dataset[target.index(tar)].append(data)
print(new_train_dataset)
for i in range(len(new_train_dataset)):
    print(len(new_train_dataset[i]))

new_features = get_features(dataset=new_train_dataset[1])
print(new_features)

new_counts = get_num_for_features(features=new_features,
                              dataset=new_train_dataset)
print(new_counts)
