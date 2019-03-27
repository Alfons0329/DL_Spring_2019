"""
Env: Python 3.7 on Ubuntu 18.04.2
"""
import numpy as np
import matplotlib as plt
import random as rd
import csv, os, struct

################# GLOBAL DEF ###########

F_NAME = 'titanic.csv'
N_TRAIN_DATA = 800
N_TEST_DATA = 91
N_DIM = 6

N_UNIT_1 = 6 # unit for layer 1
N_UNIT_2 = 6 # unit for layer 2
################# FILE IO ##############

def file_IO():
    with open(F_NAME, newline = '') as csvfile:
        rows = csv.reader(csvfile)

        rows = list(rows)
        label = rows[0]
        train_data = rows[1: N_TRAIN_DATA + 1]
        test_data = rows[N_TRAIN_DATA + 1:]

    return label, train_data, test_data
################# SGD #################


################## NN ##################
class NN(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.w = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]# weight of the layer
        self.b = [np.random.randn(y, 1) for y in sizes[1:]]

if __name__ == '__main__':
    label, train_data, test_data = file_IO()
    net = NN([N_DIM , N_UNIT_1, 1])
    print('Weight matrix: ', net.w)
    print('\nBias matrix: ', net.b)
