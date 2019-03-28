"""
Env: Python 3.7 on Ubuntu 18.04.2
"""
import numpy as np
import matplotlib as plt
import random
import csv, os, struct

################# GLOBAL DEF ###########

F_NAME = 'titanic.csv'
N_TRAIN_DATA = 800
N_TEST_DATA = 91
N_DIM = 6

N_UNIT_1 = 6 # unit for layer 1
N_BATCH_SIZE = 80
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

def extract(input_list, list_len, col_start, col_end):
    res_list = []
    for i in range(0, list_len):
        res_list.append(input_list[i][col_start: col_end + 1])

    return res_list

################# ACTV #################

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return self.sigmoid(z) * (1 - self.sigmoid(z))

################## NN ##################
class NN(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weight = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]# weight of the layer
        self.bias = [np.random.randn(y, 1) for y in sizes[1:]]


    ################## FWD #################
    def forward(self, x):
        for b, w in zip(self.bias, self.weight):
            x = self.sigmoid(np.dot(w, x) + b)

        return x
    ################## BP ##################
    # BP, 1st, input
    def backpropogation(self, x, y):
        gra_b = [np.zeros(b.shape) for b in self.bias]
        gra_w = [np.zeros(w.shape) for w in self.weight]

        activation = x
        activations = [x]
        zs = []

        # BP, 2nd, feedforward to chain together
        # prevent safe rule error
        activation = np.array(activation)
        for b, w in zip(self.bias, self.weight):
            w = w.astype(float)
            activation = activation.astype(float)
            b = b.astype(float)

            z = np.dot(w, activation) + b
            zs.append(z)
            #print(w, '\n')
            #print(activation, '\n')
            #print(b, '\n')
            #print(z, '\n')
            activation = sigmoid(z)
            activations.append(activation)

        # BP, 3rd, output error
        z_L = zs[-1]
        print("last activation ", activations[-1])
        print("last w, b", w, '\n\n\n', b)
        delta_L = self.cross_entrophy_derivative(activations[-1], y) * sigmoid_prime(z_L)
        gra_b = delta_L
        gra_w = np.dot(delta_L, activations[-2].transpose())

        # BP, 4th, back propogation from the second-last layer
        for layer in range(2, self.num_layers):
            z_layer = zs[-layer]
            s_prime = sigmoid_prime(z_layer)
            delta_L = np.dot(self.weight[-layer + 1].transpose(), delta_L) * s_prime
            gra_b[-layer] = delta_L
            gra_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())

        return gra_b, gra_w

    ################## CROSS ENTROPY #######
    """
    cross_entrophy_derivative: refer to https://blog.csdn.net/jasonzzj/article/details/52017438

    x as the input batch and y as the result of batch
    """
    def cross_entrophy_derivative(self, output_activations, x, y):
        output = [np.zeros(w.shape) for w in self.weight]
        for i in range(1, N_TRAIN_DATA + 1):
            output = output + ((self.sigmoid() - y) * x)

        return output / N_TRAIN_DATA

    ################## BATCH ################
    def update_mini_batch(self, mini_batch, eta, train_expected_output):
        gra_b = [np.zeros(b.shape) for b in self.bias]
        gra_w = [np.zeros(w.shape) for w in self.weight]
        for x in mini_batch:
            print('batch ', x)
            input()
            delta_gra_b, delta_gra_w = self.backpropogation(x, train_expected_output)
            gra_b = [nb + dnb for nb, dnb in zip(gra_b, delta_gra_b)]
            gra_w = [nw + dnw for nw, dnw in zip(gra_w, delta_gra_w)]

        self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.bias, gra_b)]
        self.weights = [w - (eta/len(mini_batch))* nw for w, nw in zip(self.weight, gra_w)]


    ################## SGD ##################
    def SGD(self, train_data, epochs, mini_batch_size, eta, test_input, test_expected_output):
        for j in range(0, epochs):
            random.shuffle(train_data)
            mini_batch_all = [train_data[k: k + mini_batch_size] for k in range(0, N_TRAIN_DATA, mini_batch_size)]

            for each_mini_batch in mini_batch_all:
                self.update_mini_batch(each_mini_batch, eta, train_expected_output)

            if test_data:
                print ("Epoch ", j, " ", self.evaluate(test_input, test_expected_output), " / ", N_TEST_DATA)
            else:
                print ("Epoch ", j, " complete")
                #print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), N_TEST_DATA)

    ################## EVAL RESULT ############
    # fix this no need for argmax, result (alive or dead put in another list for comparison)
    def evaluate(self, test_input, test_expected_output):
        test_results = [self.feedforward(x) for x in test_input]

        return sum(int(x == y) for x, y in zip(test_results, test_expected_output))


if __name__ == '__main__':
    label, train_data, test_data = file_IO()
    train_input = extract(train_data, N_TRAIN_DATA, 1, 6)
    train_expected_output = extract(train_data, N_TRAIN_DATA, 0, 0)
    test_input = extract(test_data, N_TEST_DATA, 1, 6)
    test_expected_output = extract(train_data, N_TEST_DATA, 0, 0)

    """
    for to_print in train_input, train_expected_output, test_input, test_expected_output:
        input()
        print(to_print)
        print('\n')
    """
    net = NN([N_DIM , N_UNIT_1, 1])
    #print('\nBias matrix: ', net.bias)
    #print('Weight matrix: ', net.weight)

    net.SGD(train_input, N_EPOCH_LIMIT, N_BATCH_SIZE, LEARNING_RATE, test_input, test_expected_output)
