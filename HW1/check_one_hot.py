from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt
import random
import csv, os, struct, math, sys

CHECK_COLUMN = sys.argv[1]
F_NAME = 'titanic_train_kaggle.csv'
N_TRAIN_DATA = 800

def file_IO():
    with open(F_NAME, newline = '') as csvfile:
        rows = csv.reader(csvfile)

        rows = list(rows)
        label = rows[0]
        train_data = rows[1: N_TRAIN_DATA + 1]
        test_data = rows[N_TRAIN_DATA + 1:]
        all_data = train_data + test_data

    return label, train_data, test_data, all_data

def parse(data):
    data = np.array(data)
    col = int(CHECK_COLUMN)
    to_check = data[:,col]
    m = dict()
    for i in to_check:
        if i in m:
            m[i] += 1
        else:
            m[i] = 1

    for k, v in m.items():
        print(k, ' ', v)
    return m

def cmp_map(m1, m2):
    diff_dict = m1.keys() - m2.keys()
    print('Total differences ', len(diff_dict))

if __name__ == '__main__':
    label, train_data, test_data, all_data = file_IO()
    m1 = parse(train_data)
    m2 = parse(test_data)
    cmp_map(m1, m2)


