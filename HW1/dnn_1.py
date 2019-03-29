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

N_UNIT_1 = 4 # unit for layer 1
N_BATCH_SIZE = 40
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
    return sigmoid(z) * (1 - sigmoid(z))

################## NN ##################
class NN(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weight = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]# weight of the layer
        self.bias = [np.random.randn(y, 1) for y in sizes[1:]]


    ################## FWD #################
    def forward(self, x):
        print('input x ', x)
        for b, w in zip(self.bias, self.weight):
            nb = np.array(self.bias)
            nw = np.array(self.weight)
            nx = np.array(x)

            nb = nb.astype(float)
            nw = nb.astype(float)
            nx = nb.astype(float)
            x = sigmoid(np.dot(w, x) + b)

        print('result x', x)
        return x
    ################## BP ##################
    # BP, 1st, input
    def backpropogation(self, x, y):
        gra_b = [np.zeros(b.shape) for b in self.bias]
        gra_w = [np.zeros(w.shape) for w in self.weight]

        activation = x
        activations = [[[]]]
        zs = []
        # BP, 2nd, feedforward to chain together
        # prevent safe rule error
        activation = np.array(activation)
        activation = activation.astype(float)
        activation = activation.reshape(1, N_DIM)
        activations.append(activation)

        for b, w in zip(self.bias, self.weight):
            w = w.astype(float)
            b = b.astype(float)
            z = np.dot(activation, w.T) + b.T
            # print('dim input ', activation.shape, 'dim w.T ', w.T.shape, 'dim b', b.shape)
            #z = np.dot(activation, w.T) + b
            # print('dim z(input * W.t + b) is', z.shape)
            zs.append(z)
            activation = sigmoid(z)
            # print('activation shape ', activation)
            #input()
            activations.append(activation)

        # BP, 3rd, output error
        z_L = zs[-1]
        delta_L = self.cross_entrophy_derivative(activations[-1], y) * sigmoid_prime(z_L)
        gra_b[-1] = delta_L
        gra_w[-1] = np.dot(delta_L, np.array(activations[-2]))

        # BP, 4th, back propogation from the second-last layer
        # print('delta_L first.shape ', delta_L.shape)
        # print('gra_w ', gra_w)

        for layer in range(2, self.num_layers):
            # print('num_layers ', self.num_layers)
            z_layer = zs[-layer]
            s_prime = sigmoid_prime(z_layer)
            delta_L = np.dot(self.weight[-layer + 1].T, delta_L) * s_prime.T
            gra_b[-layer] = delta_L
            # print('delta_L shape ', delta_L.shape, ' activation shape ', np.array(activations[-layer - 1]).shape)
            gra_w[-layer] = np.dot(delta_L, np.array(activations[-layer - 1]).astype(float))

        return gra_b, gra_w

    ################## CROSS ENTROPY #######
    """
    cross_entrophy_derivative: refer to https://blog.csdn.net/jasonzzj/article/details/52017438

    x as the input batch and y as the result of batch
    """
    def cross_entrophy_derivative(self, network_output_a, expected_output_y):
        #a_float = [float(i, j) for i, j in network_output_a]
        #y_float = [float(i, j) for i, j in expected_output_y]
        a_todo = float(network_output_a[0][0])
        y_todo = float(expected_output_y[0][0])
        return (a_todo - y_todo) / (a_todo * (1 - a_todo))

    ################## BATCH ################
    def update_mini_batch(self, mini_batch, eta, mini_batch_expected_output):
        gra_b = [np.zeros(b.shape) for b in self.bias]
        gra_w = [np.zeros(w.shape) for w in self.weight]

        for i, j in zip(mini_batch, mini_batch_expected_output):
            delta_gra_b, delta_gra_w = self.backpropogation(i, j)
            gra_b = [nb + dnb for nb, dnb in zip(gra_b, delta_gra_b)]
            gra_w = [nw + dnw for nw, dnw in zip(gra_w, delta_gra_w)]
            print('gra_w ', gra_w)

        self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.bias, gra_b)]
        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weight, gra_w)]
        print('weight update to ', self.weight)
        input()

    ################## SGD ##################
    def SGD(self, train_input, train_expected_output, epochs, mini_batch_size, eta, test_input, test_expected_output):
        for j in range(0, epochs):
            together = list(zip(train_input, train_expected_output))
            random.shuffle(together)
            train_input, train_expected_output = zip(*together)
            mini_batch_all_input = [train_input[k: k + mini_batch_size] for k in range(0, N_TRAIN_DATA, mini_batch_size)]
            mini_batch_all_expected_output = [train_expected_output[k: k + mini_batch_size] for k in range(0, N_TRAIN_DATA, mini_batch_size)]

            for mini_batch_input, mini_batch_expected_output in zip(mini_batch_all_input, mini_batch_all_expected_output):
                self.update_mini_batch(mini_batch_input, eta, mini_batch_expected_output)

            if test_data:
                print('temp end')
                return;
                # print ("Epoch ", j, " ", self.evaluate(test_input, test_expected_output), " / ", N_TEST_DATA)
            else:
                print ("Epoch ", j, " complete")
                #print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), N_TEST_DATA)

    ################## EVAL RESULT ############
    # fix this no need for argmax, result (alive or dead put in another list for comparison)
    def evaluate(self, test_input, test_expected_output):
        test_results = [self.forward(x) for x in test_input]
        return sum(int(x == y) for x, y in zip(test_results, test_expected_output))


if __name__ == '__main__':
    label, train_data, test_data = file_IO()
    train_input = extract(train_data, N_TRAIN_DATA, 1, 6)
    train_expected_output = extract(train_data, N_TRAIN_DATA, 0, 0)
    test_input = extract(test_data, N_TEST_DATA, 1, 6)
    test_expected_output = extract(train_data, N_TEST_DATA, 0, 0)

    net = NN([N_DIM , N_UNIT_1, 1])
    print('\nBias matrix: ', net.bias)
    print('Weight matrix: ', net.weight)
    net.SGD(train_input, train_expected_output, N_EPOCH_LIMIT, N_BATCH_SIZE, LEARNING_RATE, test_input, test_expected_output)
