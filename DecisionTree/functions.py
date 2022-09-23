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

def get_features_and_num_with_unknown_feature(dataset):
    features = [[] for _ in range(len(dataset[0]))]
    for data in dataset:
        for i in range(len(data)):
            if data[i] not in features[i]:
                features[i].append(data[i])
    counts = []
    for f in range(len(features)):
        count = [0 for _ in range(len(features[f]))]
        for d in dataset:
            if d[f] in features[f]:
                count[features[f].index(d[f])] += 1
        # counts.append(count)
        if 'unknown' in features[f]:
            idx = features[f].index('unknown')
            new_f = features[f].copy()
            new_c = count.copy()
            new_f.remove(features[f][idx])
            new_c.remove(count[idx])
            major_label = new_f[new_c.index(max(new_c))]  # Print or modify the label according to major label if needed
            new_c[new_c.index(max(new_c))] += count[idx]
            count = new_c
            features[f] = new_f
        counts.append(count)
    return features, counts

def get_num_feature(dataset, feature, id):
    # print(id)
    counts = [0 for _ in range(len(feature))]
    for data in dataset:
        for f in feature:
            if data[id] == f:
                counts[feature.index(f)] += 1
    return counts

def get_new_data(dataset, target, id):
    new_dataset = [[] for _ in range(len(target))]
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

def Majority_Error(dataset):  # Only for binary dataset
    return min(dataset)/sum(dataset)

def Gini_index(dataset):
    P = []
    for num in dataset:
        P.append(num / sum(dataset))
    Gini = 1
    for i in range(len(P)):
        Gini -= P[i]*P[i]
    return Gini

def information_gain(new_features, counts, dataset, labels, E, features):  # E (Entropy) is the total Entropy for dataset S
    Gain = []
    for i in range(len(new_features[:-1])):
        f_data = get_new_data(dataset, new_features[i], features.index(new_features[i]))
        L = [len(item) for item in f_data]
        En = [0 for i in range(len(f_data))]
        P = []
        for sub_data in f_data:
            P.append(len(sub_data) / sum(L))
            if len(sub_data) != 0:
                a = get_num_feature(sub_data, labels, -1)
                "Manually select the proper computing method here, will be reconstructed to a class function in the future"
                # En[f_data.index(sub_data)] = Entropy(a)
                En[f_data.index(sub_data)] = Gini_index(a)
                # En[f_data.index(sub_data)] = Majority_Error(a)
        gain = 0
        for j in range(len(P)):
            gain += P[j] * En[j]
        Gain.append(E - gain)
    return Gain

def training(train_dataset, tree_depth):
    # features, counts = get_features_and_num_with_unknown_feature(train_dataset)  # Manually select the proper function for dataset with or without missing features
    features, counts = get_features_and_num(dataset=train_dataset)
    print('The dataset has ', len(features), 'features: ', features)
    labels = features[-1]
    org_features = features.copy()
    train_dataset = [train_dataset]
    counts = [counts]
    org_d = [[0 for _ in range(len(features)-1)]]
    features = [features]
    dic = {}
    DIC = {}
    TREE = [[]]
    for i in range(tree_depth):
        new_counts = []
        new_d = []
        new_f = []
        new_dict_label = []
        new_Tree = []
        for j in range(len(train_dataset)):
            "Manually select the proper computing method here, will be reconstructed to a class function in the future"
            # E = Entropy(counts[j][-1])4
            E = Gini_index(counts[j][-1])
            # E = Majority_Error(counts[j][-1])

            Gain = information_gain(features[j], counts[j], train_dataset[j], labels, E, org_features)
            # print(i, Gain)
            new = get_new_data(train_dataset[j], features[j][Gain.index(max(Gain))], org_features.index(features[j][Gain.index(max(Gain))]))
            new_features = features[j].copy()
            new_features.remove(features[j][Gain.index(max(Gain))])
            n_counts = []
            NEW = new.copy()
            n_f = []
            Tree = TREE[j].copy()
            Tree.append(org_features.index(features[j][Gain.index(max(Gain))]))
            for n in new:
                dic_label = org_d[j].copy()
                sub_labels = get_num_feature(n, labels, -1)
                dic_label[org_features.index(features[j][Gain.index(max(Gain))])] = features[j][Gain.index(max(Gain))][new.index(n)]
                if len([item for item in sub_labels if item != 0]) == 1:
                    dic[tuple(dic_label)] = (labels[sub_labels.index(sum([i for i in sub_labels if i != 0]))], Tree)
                    # DIC[tuple(dic_label)] = Tree
                    NEW.remove(n)
                else:
                    dic[tuple(dic_label)] = (labels[sub_labels.index(max(sub_labels))], Tree)
                    b = []
                    for nf in new_features[:-1]:
                        b.append(get_num_feature(n, nf, org_features.index(nf)))
                    b.append(get_num_feature(n, labels, -1))
                    n_counts.append(b)
                    n_f.append(dic_label)
            Tree = [Tree for _ in range(len(NEW))]
            new_Tree += Tree
            new_d += NEW
            new_counts += n_counts
            # new_f.append(new_features)
            new_f += [new_features for _ in range(len(new))]
            new_dict_label += n_f
        train_dataset = new_d
        counts = new_counts
        features = new_f
        org_d = new_dict_label
        TREE = new_Tree
    print('[TRAINING COMPLETE]')
    return dic

def testing(test_dataset, dic):
    print('[TESTING START]')
    correct = 0
    for test_data in test_dataset:
        A = []
        label = test_data[-1]
        for item in dic.keys():
            a = [0 for _ in range(len(item))]
            for i in list(dic[item])[1]:
                if test_data[i] == list(item)[i]:
                    a[i] = list(item)[i]
            if tuple(a) == item:
                A.append(tuple(a))
        if len(A) != 0:
            pred = dic[A[-1]][0]
            if pred == label:
                correct += 1
    return round((correct / len(test_dataset)), 4)
