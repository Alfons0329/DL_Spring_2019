from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pandas import DataFrame, read_csv
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
import regex as re
import os, sys, math

############# MY DATASET #############
import preprocessing_2 as pre

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

N_HID_SIZE = 16
N_RNN_STEP = 10 # 10 step for the sentence title length of 10 words

N_EPOCH_LIMIT = 1000
N_TEST_SIZE = 50
N_TRAIN_SIZE_ACC = 0
N_TRAIN_SIZE_REJ = 0

############# FOR GRAPHING ############
epoch_list = []
learning_curve = []
train_acc_list = []
test_acc_list = []

def make_graph():
    # plot the accuracy of training set and testing set
    plt.clf()
    title_str = 'Acc, BAT=' + str(N_BATCH_SIZE) + ' ETA = ' + str(N_LEARN_RATE)
    plt.title(title_str)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.plot(epoch_list, train_acc_list, color = 'blue', label = 'train acc')
    plt.plot(epoch_list, test_acc_list, color = 'red', label = 'test acc')
    plt.legend()
    plt.savefig(adaptive_lr + '_' + str(N_LEARN_RATE) + '_' + str(N_BATCH_SIZE) + '_' + 'ACC' + '.png', dpi = 150)

    # plot the learning curve
    plt.clf()
    title_str = 'LC, BAT=' + str(N_BATCH_SIZE) + ' ETA = ' + str(N_LEARN_RATE)
    plt.title(title_str)
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy')

    plt.plot(epoch_list, learning_curve, color = 'blue', label = 'no norm')
    plt.legend()
    plt.savefig(adaptive_lr + '_' + str(N_LEARN_RATE) + '_' + str(N_BATCH_SIZE) + '_' + 'ACC' + '.png', dpi = 150)

############# NN MAIN PART ###########
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        # RNN layer
        self.rnn = nn.RNN(
                input_size = 10,
                hidden_size = N_HID_SIZE,
                num_layers = 1,
                dropout = 0.5,
                batch_first = False,
                bidirectional = False
                )
        self.out = nn.Linear(N_HID_SIZE, 2) # accepted %, rejected %

        # forward dnn classifier
    def forward(self, x):
        x, _ = self.rnn(x)
        print('x ', x)
        print('after rnn x ', x[:, -1, :])
        x = self.out(x)
        return x

############# TRAIN NN #################
def train(train_loader, model, criterion, optimizer, cur_epoch, device):
    train_loss = 0.0
    total = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        print('input: ', inputs.shape)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    learning_curve.append(float(train_loss) / float(N_BATCH_SIZE))
    print('Epoch %5d CE loss: %.3f' %(cur_epoch, float(train_loss) / float(N_BATCH_SIZE)))

############# VALIDATE NN ##############
def validate(val_loader, model, criterion, cur_epoch, device, what):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total != 0:
        print('Accuracy on %6s set of %d sentences is %f' %(what, total, float(correct) / float(total)))
        return float(correct) / float(total)
    else:
        return 0

############# MAIN FUNCT #############
if __name__ == '__main__':

    ############# LOAD DATASET ###########
    train_loader, test_loader = pre.load_custom_dataset(N_BATCH_SIZE)

    ############# INSTANTIATE RNN ########
    model = RNN()

    ############# CUUUUUUUDA #############
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        print('Train with CUDA ')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ############# PARALLELISM ############
        model.features = torch.nn.DataParallel(model.rnn)
        model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()

    ############# TRAINING ###############
    print('Start training, N_BATCH_SIZE = %4d, N_EPOCH_LIMIT = %4d, N_LEARN_RATE %f\n' %(N_BATCH_SIZE, N_EPOCH_LIMIT, N_LEARN_RATE))
    cur_acc = 0.0

    for cur_epoch in range(N_EPOCH_LIMIT):
        print('cur_epoch %d N_LEARN_RATE %f' %(cur_epoch, N_LEARN_RATE))
        epoch_list.append(cur_epoch)

        # determine the optimization method
        if adaptive_lr == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr = N_LEARN_RATE, momentum = 0.9, weight_decay = 5e-4)
        elif adaptive_lr == 'adam':
            optimizer = optim.Adam(model.parameters(), lr = N_LEARN_RATE, weight_decay = 5e-4)

        train(train_loader, model, criterion, optimizer, cur_epoch, device)
        train_acc_list.append(train_loader, model, criterion, cur_epoch, device, 'train')

        print('')
        cur_acc = validate(test_loader, model, criterion, cur_epoch, device, 'test')
        test_acc_list.append(cur_acc)
        print('-----------------------------------------------\n')

        # save the model and corresponding accuracy if this is the final epoch with better result
        """
        if cur_epoch == N_EPOCH_LIMIT - 1 and cur_acc > best_acc:
            print('Last epoch, better model with cur_acc %.3f over best_acc %.3f, save model and acc'%(cur_acc, best_acc))
            torch.save(model, model_path)
            f = open(acc_path, 'w')
            f.write(str(cur_acc))
            f.close()
        """

    torch.cuda.empty_cache()
    make_graph()
