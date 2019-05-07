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
N_HID_SIZE = 16
N_RNN_STEP = 10 # 10 step for the sentence title length of 10 words

N_EPOCH_LIMIT = 1000
N_TEST_SIZE = 50

############# WORD EMBEDDING ###########
word_dict = dict()
word_dict['XXX'] = 0 # padding for the empty word in fixed length sentence
dict_cnt = 1
dbg_cnt = 0
longest_sentence_len = 0
longest_sentence = []

############# FOR GRAPHING ############
epoch_list = []
learning_curve = []
train_acc_list = []
test_acc_list = []

def make_graph():
    # plot the accuracy of training set and testing set
    plt.clf()
    title_str = 'STRI=' + str(N_STRID_SIZE) + ' KER=' + str(N_KERNE_SIZE) + ' Acc, BAT=' + str(N_BATCH_SIZE) + ' ETA = ' + str(N_LEARN_RATE)
    plt.title(title_str)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.plot(epoch_list, train_acc_list, color = 'blue', label = 'train acc')
    plt.plot(epoch_list, test_acc_list, color = 'red', label = 'test acc')
    plt.legend()
    plt.savefig(adaptive_lr + '_' + str(N_LEARN_RATE) + '_' + str(N_BATCH_SIZE) + '_' + str(N_STRID_SIZE) + '_' + str(N_KERNE_SIZE) + '_' + 'ACC' + '.png', dpi = 150)

    # plot the learning curve
    plt.clf()
    title_str = 'STRI=' + str(N_STRID_SIZE) + ' KER=' + str(N_KERNE_SIZE) + ' LC, BAT=' + str(N_BATCH_SIZE) + ' ETA = ' + str(N_LEARN_RATE)
    plt.title(title_str)
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy')

    plt.plot(epoch_list, learning_curve, color = 'blue', label = 'no norm')
    plt.legend()
    plt.savefig(adaptive_lr + '_' + str(N_LEARN_RATE) + '_' + str(N_BATCH_SIZE) + '_' + str(N_STRID_SIZE) + '_' + str(N_KERNE_SIZE) + '_' + 'LC' + '.png', dpi = 150)

############# PARSE XLSX #############
def parse_xls(f_name):
    df = pd.read_excel(f_name)
    df = df[[0]]

    return df[N_TEST_SIZE:], df[0: N_TEST_SIZE]

############# WORD MBED AND DICT #####
def build_dict(data):
    for each_sentence in data:
        # print('each_sentence', each_sentence, ' len ', len(each_sentence))
        for no_bracket_sentence in each_sentence:
            str_sentence  = str(no_bracket_sentence)
            no_bracket_sentence = str_sentence.split()
            if str_sentence != 'No Title':
                # print('has title ', str_sentence)
                for each_word in no_bracket_sentence:
                    # print('each_word', each_word)
                    # input()
                    if each_word not in word_dict:
                        global dict_cnt
                        word_dict[each_word] = dict_cnt
                        dict_cnt += 1

def lookup(data, embeds):
    for each_sentence in data:
        # print('each_sentence', each_sentence, ' len ', len(each_sentence))
        for no_bracket_sentence in each_sentence:
            str_sentence  = str(no_bracket_sentence)
            no_bracket_sentence = str_sentence.split()
            if str_sentence != 'No Title':
                # print('has title ', str_sentence)
                for each_word in no_bracket_sentence:
                    if each_word in word_dict:
                        lookup_tensor = torch.tensor([word_dict[each_word]], dtype = torch.long)
                        word_embed = embeds(lookup_tensor)
                        # print('Word: ', each_word, 'lookup_tensor ', lookup_tensor, 'embed to ', word_embed)

############# SENTENCES 2 TENSOR #####
"""
convert the training and test loader into the matrix of torch tensor to
be fed into the recurrent neural network
"""

def sentense2tensor(data):
    data_to_tensor = list()

    for each_sentence in data:
        # print('each_sentence', each_sentence, ' len ', len(each_sentence))
        each_sentence_embed = list()
        for no_bracket_sentence in each_sentence:
            str_sentence  = str(no_bracket_sentence)
            no_bracket_sentence = str_sentence.split()
            if str_sentence != 'No Title':
                # print('has title ', str_sentence)
                for cnt in range(N_VEC_SIZE):
                    if cnt < len(no_bracket_sentence):
                        if no_bracket_sentence[cnt] in word_dict:
                            query = no_bracket_sentence[cnt]
                        else:
                            query = 'XXX'
                        # print('query: ', query)
                        lookup_tensor = torch.tensor([word_dict[query]], dtype = torch.long)
                    else:
                        lookup_tensor = torch.tensor([word_dict['XXX']], dtype = torch.long)
                        # print('XXX')

                    word_embed = embeds(lookup_tensor)
                    each_sentence_embed.append(word_embed)

            #print( word_embed, len(word_embed))

        data_to_tensor.append(each_sentence_embed)
        print('each_sentence: ', each_sentence, ' mbed tensor: ', each_sentence_embed)

    # print('data_to_tensor type is ', type(data_to_tensor))
    # print('data_to_tensor: ', data_to_tensor)
    print('size: ', len(data_to_tensor), len(data_to_tensor[0]), len(data_to_tensor[0][0][0]))
    data_to_tensor = torch.tensor(data_to_tensor)

############# NN MAIN PART ###########
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        # RNN layer
        self.rnn = nn.RNN(
                input_size = 10,
                hidden_size = N_HID_SIZE,
                num_layers = 1,
                batch_first = True
                )
        self.out = nn.Linear(N_HID_SIZE, 2) # accepted %, rejected %

        # forward dnn classifier
        def forward(self, x):
            x, _ = self.rnn(x, None)
            x = self.out(x[:, -1, :])
            return x

############# TRAIN NN #################
def train(train_loader, model, criterion, optimizer, cur_epoch, device):
    train_loss = 0.0
    total = 0

    # train_loader = torch.FloatTensor(train_loader)
    train_loader, train_loader_label = zip(*train_loader)

    for inputs, labels in zip(train_loader, train_loader_label):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
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

    # train_loader = torch.FloatTensor(train_loader)
    train_loader, train_loader_label = zip(*train_loader)

    with torch.no_grad():
        for inputs, labels in zip(train_loader, train_loader_label):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            class_predicted = (predicted == labels).squeeze()
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
    train_loader_acc, test_loader_acc = parse_xls(F_NAME_ACCEPT)
    train_loader_rej, test_loader_rej = parse_xls(F_NAME_REJECT)

    train_loader_acc = train_loader_acc.values.tolist()
    test_loader_acc = test_loader_acc.values.tolist()
    train_loader_rej = train_loader_rej.values.tolist()
    test_loader_rej = test_loader_rej.values.tolist()

    train_loader = train_loader_acc + train_loader_rej
    test_loader = test_loader_acc + test_loader_rej

    ############# WORD MBED AND DICT #####
    build_dict(test_loader)
    embeds = nn.Embedding(len(word_dict) + 1, N_VEC_SIZE) # padding to prevent runtime error
    lookup(test_loader, embeds)

    #print('longest_sentence ', longest_sentence, 'longest_sentence_len ', longest_sentence_len)
    #input()
    ############# INSTANTIATE RNN ########
    model = RNN()

    ############# CUUUUUUUDA #############
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        print('Train with CUDA ')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.features = torch.nn.DataParallel(model.rnn)
        model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()

    ############# TRAINING ###############
    print('Start training, N_BATCH_SIZE = %4d, N_EPOCH_LIMIT = %4d, N_LEARN_RATE %f\n' %(N_BATCH_SIZE, N_EPOCH_LIMIT, N_LEARN_RATE))
    cur_acc = 0.0

    train_loader = sentense2tensor(train_loader)
    test_loader = sentense2tensor(test_loader)

    for cur_epoch in range(N_EPOCH_LIMIT):
        print('cur_epoch %d N_LEARN_RATE %f' %(cur_epoch, N_LEARN_RATE))
        epoch_list.append(cur_epoch)

        # determine the optimization method
        if adaptive_lr == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr = N_LEARN_RATE, momentum = 0.9, weight_decay = 5e-4)
        elif adaptive_lr == 'adam':
            optimizer = optim.Adam(model.parameters(), lr = N_LEARN_RATE, weight_decay = 5e-4)

        train(train_loader, model, criterion, optimizer, cur_epoch, device)
        train_acc_list.append(validate(train_loader, model, criterion, cur_epoch, device, 'train'))

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
