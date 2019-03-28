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
N_UNIT_2 = 3 # unit for layer 2
N_EPOCH_LIMIT = 100
LEARNING_RATE = 0.50
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

    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward(self, x):
        for b, w in zip(self.b, self.w):
            x = self.sigmoid(np.dot(w, x) + b)

        return x

    def backpropogation(self, x, y):
        grad_b = [np.zeros(b.shape) for b in self.b]
        grad_w = [np.zeros(w.shape) for w in self.w]

    """
    cross_entrophy_derivative: refer to https://blog.csdn.net/jasonzzj/article/details/52017438

    x as the input batch and y as the result of batch
    """
    def cross_entrophy_derivative(self, output_activations, x, y):
        output = [np.zeros(w.shape) for w in self.w]
        for i in range(1, N_TRAIN_DATA + 1):
            output = output + ((self.sigmoid() - y) * x)


        return output / N_TRAIN_DATA

if __name__ == '__main__':
    label, train_data, test_data = file_IO()
    net = NN([N_DIM , N_UNIT_1, N_UNIT_2, 1])
    #print('Weight matrix: ', net.w)
    #print('\nBias matrix: ', net.b)
