"""
Env: Python 3.7 on Ubuntu 18.04.2
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import csv, os, struct, math, sys

################# GLOBAL DEF ###########

F_NAME = 'titanic.csv'
N_TRAIN_DATA = 800
N_TEST_DATA = 91
N_DIM = 6
RANDOM_SEED = 4

N_UNIT_1 = 3 # unit for hidden layer 1
N_UNIT_2 = 3 # unit for hidden layer 2
N_BATCH_SIZE = int(sys.argv[2])
N_EPOCH_LIMIT = 3000
LEARNING_RATE = float(sys.argv[3])

epoch_list = []
learning_curve = []
train_error_curve = []
test_error_curve = []


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
    if col_end == col_start:
         for i in range(0, list_len):
            res_list.append(input_list[i][col_start])
    else:
        for i in range(0, list_len):
            res_list.append(input_list[i][col_start: col_end + 1])

    return res_list

################# GRAPH ################

def make_graph():
    title_str = 'Learning Curve, BATCH_SIZE = ' + str(N_BATCH_SIZE) + ', ETA = ' + str(LEARNING_RATE)
    plt.title(title_str)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(epoch_list, learning_curve)
    plt.savefig(sys.argv[1] + '_' + 'LC_P2' + '.png', dpi = 100)

    plt.clf()
    title_str = 'Train Error, BATCH_SIZE = ' + str(N_BATCH_SIZE) + ', ETA = ' + str(LEARNING_RATE)
    plt.title(title_str)
    plt.xlabel('Epochs')
    plt.ylabel('Error rate')
    plt.plot(epoch_list, train_error_curve)
    plt.savefig(sys.argv[1] + '_' + 'TRE_P2' + '.png', dpi = 100)

    plt.clf()
    title_str = 'Test Error, BATCH_SIZE = ' + str(N_BATCH_SIZE) + ', ETA = ' + str(LEARNING_RATE)
    plt.title(title_str)
    plt.xlabel('Epochs')
    plt.ylabel('Error rate')
    plt.plot(epoch_list, test_error_curve)
    plt.savefig(sys.argv[1] + '_' + 'TEE_P2' + '.png', dpi = 100)
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
        np.random.seed(RANDOM_SEED)
        self.weight = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]# weight of the layer
        self.bias = [np.random.randn(y, 1) for y in sizes[1:]]


    ################## FWD #################
    def forward(self, x):
        x = np.array(x)
        x = x.astype(float)
        x = x.reshape(1, N_DIM)

        for b, w in zip(self.bias, self.weight):
            x = sigmoid(np.dot(x, w.T) + b.T)

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

        activation = np.array(activation)
        activation = activation.astype(float)
        activation = activation.reshape(1, N_DIM)

        activations.append(activation)

        for b, w in zip(self.bias, self.weight):
            w = w.astype(float)
            b = b.astype(float)

            z = np.dot(activation, w.T) + b.T
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # BP, 3rd, output error
        z_L = zs[-1]
        delta_L = self.cross_entrophy_derivative(activations[-1], y) * sigmoid_prime(z_L)
        gra_b[-1] = delta_L.T
        gra_w[-1] = np.dot(delta_L.T, np.array(activations[-2]))
        # BP, 4th, back propogation from the second-last layer
        delta_L = delta_L.T

        for layer in range(2, self.num_layers):
            z_layer = zs[-layer]
            s_prime = sigmoid_prime(z_layer)
            delta_L = np.dot(self.weight[-layer + 1].T, delta_L) * s_prime.T
            gra_b[-layer] = delta_L
            gra_w[-layer] = np.dot(delta_L, np.array(activations[-layer - 1]).astype(float))

        return gra_b, gra_w

    ################## CROSS ENTROPY #######
    """
    cross_entrophy_derivative: refer to https://blog.csdn.net/jasonzzj/article/details/52017438

    x as the input batch and y as the result of batch
    """
    def cross_entrophy_derivative(self, network_output_a, expected_output_y):
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

        #print('eta ', eta, 'minilen ', len(mini_batch), ' div ', (eta / len(mini_batch)))
        self.bias = [b - (eta) * nb for b, nb in zip(self.bias, gra_b)]
        self.weight = [w - (eta) * nw for w, nw in zip(self.weight, gra_w)]

    ################## SGD ##################
    def SGD(self, train_input, train_expected_output, epochs, mini_batch_size, eta, test_input, test_expected_output):
        for j in range(0, N_EPOCH_LIMIT):
            together = list(zip(train_input, train_expected_output))
            random.shuffle(together)
            train_input, train_expected_output = zip(*together)

            mini_batch_all_input = [train_input[k: k + mini_batch_size] for k in range(0, N_TRAIN_DATA, mini_batch_size)]
            mini_batch_all_expected_output = [train_expected_output[k: k + mini_batch_size] for k in range(0, N_TRAIN_DATA, mini_batch_size)]

            for mini_batch_input, mini_batch_expected_output in zip(mini_batch_all_input, mini_batch_all_expected_output):
                self.update_mini_batch(mini_batch_input, eta, mini_batch_expected_output)

            if test_data:
                print ("Epoch ", j, ", Cross Entropy = ", self.evaluate(test_input, test_expected_output))
                print ("Epoch ", j, ", E = ", self.evaluate_error(test_input, test_expected_output))
                epoch_list.append(j)
                learning_curve.append(self.evaluate(test_input, test_expected_output) / N_TEST_DATA)
                train_error_curve.append(self.evaluate_error(train_input, train_expected_output))
                test_error_curve.append(self.evaluate_error(test_input, test_expected_output))

    ################## EVAL RESULT ############
    # fix this no need for argmax, result (alive or dead put in another list for comparison)
    def evaluate(self, inpu, expected_output):
        test_results = [self.forward(x) for x in inpu]
        ce = 0.0 # for alive
        ce_2 = 0.0 # for death
        ce = float(ce)
        ce_2 = float(ce_2)
        for i, j in zip(test_results, expected_output):
            ce += float(j) * math.log2(float(i[0][0]))
            ce_2 += float(j) * math.log2(float(i[0][1]))
        return (ce + ce_2) * (-1.0)

    def evaluate_error(self, inpu, expected_output):
        test_results = [self.forward(x) for x in inpu]
        correct = 0
        alive_dead = []
        test_results = np.array(test_results)
        for i in range(len(test_results)):
            if test_results[i][0][0] > test_results[i][0][1]:
                alive_dead.append(1) #alive
            else:
                alive_dead.append(0) #dead

        for i, j in zip(alive_dead, expected_output):
            if int(i) == int(j):
                correct += 1
        correct = float(correct)
        one = 1.0
        one = float(one)
        return one - correct / (float)(len(test_results))

if __name__ == '__main__':
    label, train_data, test_data = file_IO()
    train_input = extract(train_data, N_TRAIN_DATA, 1, 6)
    train_expected_output = extract(train_data, N_TRAIN_DATA, 0, 0)
    test_input = extract(test_data, N_TEST_DATA, 1, 6)
    test_expected_output = extract(test_data, N_TEST_DATA, 0, 0)
    random.seed(RANDOM_SEED)
    net = NN([N_DIM , N_UNIT_1, N_UNIT_2, 2])
    net.SGD(train_input, train_expected_output, N_EPOCH_LIMIT, N_BATCH_SIZE, LEARNING_RATE, test_input, test_expected_output)
    make_graph()
