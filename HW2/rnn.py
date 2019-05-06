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

N_VEC_SIZE = 10
N_EPOCH_LIMIT = 200
N_TEST_SIZE = 50

word_dict = dict()
dict_cnt = 1
dbg_cnt = 0

############# PARSE XLSX #############
def parse_xls(f_name):
    df = pd.read_excel(f_name)
    df = df[[0]]

    return df[N_TEST_SIZE:], df[0: N_TEST_SIZE]

############# WORD MBED AND DICT #####
def build_dict(data):
    for each_sentence in data:
        to_split = str(each_sentence)
        to_split = to_split.split()

        for each_word in to_split:
            if each_word not in word_dict:
                global dict_cnt
                word_dict[each_word] = dict_cnt
                dict_cnt += 1

def lookup(data, embeds):
    for each_sentence in data:
        to_split = str(each_sentence)
        to_split = to_split.split()
        for each_word in to_split:
            lookup_tensor = torch.tensor([word_dict[each_word]], dtype = torch.long)
            word_embed = embeds(lookup_tensor)
            #print('Word: ', each_word, 'lookup_tensor ', lookup_tensor, 'embed to ', word_embed)

if __name__ == '__main__':
    train_input_acc, test_input_acc = parse_xls(F_NAME_ACCEPT)
    train_input_rej, test_input_rej = parse_xls(F_NAME_REJECT)

    train_input_acc = train_input_acc.values.tolist()
    test_input_acc = test_input_acc.values.tolist()
    train_input_rej = train_input_rej.values.tolist()
    test_input_rej = test_input_rej.values.tolist()


    for each_set in train_input_acc + train_input_rej:
        build_dict(each_set)

    embeds = nn.Embedding(len(word_dict) + 1, N_VEC_SIZE) # padding to prevent runtime error
    for each_set in train_input_acc + train_input_rej:
        lookup(each_set, embeds)
