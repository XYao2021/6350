import random
import numpy as np


def open_file(file_path):
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            dataset.append(line.strip().split(','))
    return dataset

def get_features_and_num(dataset):
    features = [[] for _ in range(len(dataset[0]))]
    for data in dataset:
        for i in range(len(data)):
            if data[i] not in features[i]:
                features[i].append(data[i])
    counts = []
    for f in features:
        count = [0 for _ in range(len(f))]
        for d in dataset:
            if d[features.index(f)] in f:
                count[f.index(d[features.index(f)])] += 1
        counts.append(count)
    return features, counts

# def get_num_labels(dataset, labels):
#     count = [0 for _ in range(len(labels))]
#     for d in dataset:
#         # print(d)
#         for label in labels:
#             if d[-1] == label:
#                 count[labels.index(label)] += 1
#     return count

def get_num_feature(dataset, feature, id):
    # print(id)
    counts = [0 for _ in range(len(feature))]
    for data in dataset:
        # print('\n', data)
        for f in feature:
            # print(data[id])
            # print(f, '\n')
            if data[id] == f:
                # print(data[id])
                # print(counts[feature.index(f)])
                counts[feature.index(f)] += 1
    return counts

def get_new_data(dataset, target, id):
    new_dataset = [[] for _ in range(len(target))]
    # print(id, target)
    # print(dataset[0])
    for data in dataset:
        for tar in target:
            if data[id] == tar:
                new_dataset[target.index(tar)].append(data)
    return [i for i in new_dataset if i != []]

def Entropy(dataset):
    P = []
    for num in dataset:
        P.append(num/sum(dataset))
    entropy = 0
    for i in range(len(P)):
        if P[i] == 0:
            entropy += 0
        else:
            entropy += -P[i]*np.log2(P[i])
    return entropy

def information_gain(new_features, counts, dataset, labels, E, features):  # E (Entropy) is the total Entropy for dataset S
    Gain = []
    # print(features)
    for i in range(len(new_features[:-1])):
        # print(i, 'new_features: ', new_features)
        # print(i, 'new_features[i]: ', new_features[i])
        # print(i, 'dataset: ', dataset)
        f_data = get_new_data(dataset, new_features[i], features.index(new_features[i]))
        # print(i, len(f_data), f_data, '\n')
        # print(i, 'fd', new_features[i])
        En = [0 for i in range(len(f_data))]
        for sub_data in f_data:
            a = get_num_feature(sub_data, labels, -1)
            # print(i, f_data.index(sub_data), 'a', a)
            En[f_data.index(sub_data)] = Entropy(a)
        P = []
        # print(i, 'counts[i]', counts[i])
        for num in counts[i]:
            # print(i, num, '\n')
            P.append(num / sum(counts[i]))
        # print('Compute Gain: ', En, P, len(f_data))
        if len(En) != len(P):
            P = [i for i in P if i != 0]
        gain = 0
        for j in range(len(P)):
            gain += P[j] * En[j]
        Gain.append(E - gain)
    return Gain


