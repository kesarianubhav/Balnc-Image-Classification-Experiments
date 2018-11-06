from collections import OrderedDict
from os import listdir
from os.path import join
from os.path import isfile
import numpy as np
import pandas as pd
import keras


def missclassification_rate(pred, truth):
    count = 0
    for i in range(0, len(pred)):
        if truth[i] != pred[i]:
            count = count + 1
        i = i + 1
    return count


def one_hot(a, num_classes):
    vector = np.zeros(shape=(num_classes, 1))
    vector[a][0] = 1
    return vector


def one_hot_to_integer(a):
    for i in range(0, a.shape[0]):
        if a[i][0] == 1:
            break
        i += 1

    return i


def getFilesInDir(foldername='Dataset'):
    file_map = OrderedDict()

    for f in listdir(foldername):
        for h in listdir(join(foldername, f)):
            all_files = [str(join(join(foldername, f), h))
                         for f in listdir(join(foldername, f)) if isfile(join(join(foldername, f), h))]
            if f not in file_map.keys():
                file_map[f] = [str(join(join(foldername, f), h))]
            else:
                file_map[f].append(str(join(join(foldername, f), h)))

    # print("Getting the names of the files in the directory:\n" + str(all_files))
    return (file_map)


def create_labels(file_map, output_classes):
    for i, j in file_map.items():
        print(str(len(j)) + "\n")
    labels = []
    idx = 1
    for i in file_map:
        p = file_map[i]
        for j in p:
            categorical_labels = one_hot(idx, output_classes)
            labels.append(categorical_labels)
        idx = idx + 1
    print((labels))
    return labels


def test_train_split(file_map, num):
    test_file_map = OrderedDict()
    train_file_map = OrderedDict()
    for i, j in file_map.items():
        train_file_map[i] = j[:-num]
        test_file_map[i] = j[-num:]

    return (train_file_map, test_file_map)
