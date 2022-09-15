import random
import numpy as np


def open_file(file_path):
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            dataset.append(line.strip().split(','))
    return dataset

def get_features(dataset):
    features = [[] for _ in range(len(dataset[0]))]
    for data in dataset:
        for i in range(len(data)):
            if data[i] not in features[i]:
                features[i].append(data[i])
    for f in features:
        if len(f) == 1:
            features.remove(f)
    return features

def get_num_for_features(dataset, features):
    counts = []
    features.pop(-1)
    for f in features:
        count = [0 for _ in range(len(f))]
        for d in dataset:
            if d[features.index(f)] in f:
                count[f.index(d[features.index(f)])] += 1
        counts.append(count)
    return counts

def Entropy(dataset):
    P = []
    for num in dataset:
        P.append(num/sum(dataset))
    entropy = 0
    for i in range(len(P)):
        entropy += -P[i]*np.log(P[i])
    return entropy

class compute_method(object):

    def __init__(self, dataset, attribute, depth):
        self.S = dataset
        self.A = attribute
        self.layer = depth

    def information_gain(self, A):
        pass

