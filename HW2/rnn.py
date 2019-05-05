import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pandas import DataFrame, read_csv
import pandas as pd

import numpy as np
import regex as re
import os, sys, math

############# GLOBAL DEF #############

F_NAME_ACCEPT = 'ICLR_accepted.xlsx'
F_NAME_REJECT = 'ICLR_rejected.xlsx'

ARGV_CNT = 4
if len(sys.argv) != ARGV_CNT:
    print('Error: usage: python3 rnn.py $learning_rate $batch_size { adam | sgd }')
    sys.exit(1)

N_LEARN_RATE = float(sys.argv[1])
N_BATCH_SIZE = int(sys.argv[2])
adaptive_lr = str(sys.argv[3])

N_EPOCH_LIMIT = 200
N_TEST_SIZE = 50

############# PARSE XLSX #############
def parse_xls(f_name):
    df = pd.read_excel(f_name)
    df = df[[0]]
    print(df)

    return df, df


if __name__ == '__main__':
    train_input_acc, test_input_acc = parse_xls(F_NAME_ACCEPT)
