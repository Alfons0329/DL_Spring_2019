"""
env: Python 3.7 on Ubuntu 18.04.2
cpu: Core i5 - 7500, gpu: NVIDIA GTX 1070
"""
import numpy as np
import matplotlib as plt
import csv
import random as rd

################# GLOBAL DEF ###########

F_NAME = "titanic.csv"
N_TRAIN_DATA = 800
N_TEST_DATA = 91
N_HIDDEN_LAYER = 3
N_UNITS = [6, 3, 3, 2]

################# FILE IO ##############

def file_IO():
    with open(F_NAME, newline = '') as csvfile:
        rows = csv.reader(csvfile)

        rows = list(rows)
        label = rows[0]
        train_data = rows[1: N_TRAIN_DATA + 1]
        test_data = rows[N_TRAIN_DATA + 1:]

    return label, train_data, test_data
################## SGD #################


################## NN ##################
class NN(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = N_HIDDEN_LAYER
        self.w = # weight of the layer
if __name__ == '__main__':
    label, train_data, test_data = file_IO()
