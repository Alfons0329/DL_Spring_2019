from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import os

############# GLOBAL DEF #############
"""
override the dataset class for custom dataset
"""
f_1 = 'paper/ICLR_accepted.csv'
f_2 = 'paper/ICLR_rejected.csv'
N_VEC_SIZE = 10

############# WORD EMBEDDING ###########
word_dict = dict()
word_dict['XXX'] = 0  # padding for the empty word in fixed length sentence
dict_cnt = 1
dbg_cnt = 0
longest_sentence_len = 0
longest_sentence = []


class custom_dataset(Dataset):
    def __init__(self, f_name, file_cnt, train):
        self.data = pd.read_csv(f_name)

        if train is True:
            self.data = self.data[50:]
        else:
            self.data = self.data[: 50]

        self.data = np.array(self.data)
        self.data = np.delete(self.data, 0, 1) # remove the first number column
        self.build_dict(self.data) # build dictionary for each vocab
        self.embeds = nn.Embedding(len(word_dict) + 1, N_VEC_SIZE)
        self.data_tensor = self.sentence2tensor()

        if file_cnt == 1:
            self.labels = np.zeros((len(self.data), ), dtype=int)
        else:
            self.labels = np.ones((len(self.data), ), dtype=int)
        #print(self.labels)

    def __getitem__(self, index):
        title = self.data[index]
        title_tensor = self.data_tensor[index]
        label = self.labels[index]
        return title_tensor, label

    def __len__(self):
        return len(self.data)

    ############# WOR AND DICT #####

    def build_dict(self, data):
        for each_sentence in data:
            for no_bracket_sentence in each_sentence:
                str_sentence = str(no_bracket_sentence)
                no_bracket_sentence = str_sentence.split()
                if str_sentence is not None:
                    for each_word in no_bracket_sentence:
                        if each_word not in word_dict:
                            global dict_cnt
                            word_dict[each_word] = dict_cnt
                            dict_cnt += 1

    def sentence2tensor(self):
        data_to_tensor = list()

        for each_sentence in self.data:
            # print('each_sentence', each_sentence, ' len ', len(each_sentence))
            each_sentence_embed = list()
            for no_bracket_sentence in each_sentence:
                str_sentence = str(no_bracket_sentence)
                no_bracket_sentence = str_sentence.split()
                if str_sentence is not None:
                    # print('has title ', str_sentence)
                    for cnt in range(N_VEC_SIZE):
                        if cnt < len(no_bracket_sentence):
                            if no_bracket_sentence[cnt] in word_dict:
                                query = no_bracket_sentence[cnt]
                            else:
                                query = 'XXX'
                            lookup_tensor = torch.tensor(
                                [word_dict[query]], dtype=torch.long)
                        else:
                            lookup_tensor = torch.tensor(
                                [word_dict['XXX']], dtype=torch.long)

                        word_embed = self.embeds(lookup_tensor)
                        each_sentence_embed.append(word_embed.detach().numpy())

                    # only append the sentence tensor iff the title is not 'No Title'
                    # print('each_sentence_mbed shape ', np.array(each_sentence_embed).flatten().shape, 'with value ', np.array(each_sentence_embed).flatten())
                    data_to_tensor.append(np.array(each_sentence_embed).flatten())
                    # data_to_tensor.append(np.array(each_sentence_embed))

                #print( word_embed, len(word_embed))

            # print('each_sentence: ', each_sentence, ' mbed tensor: ', each_sentence_embed)

        # print('data_to_tensor type is ', type(data_to_tensor))

        data_to_tensor = np.array(data_to_tensor)
        data_to_tensor = torch.tensor(data_to_tensor)
        # print('data_to_tensor', data_to_tensor)
        return data_to_tensor


def load_custom_dataset(N_BATCH_SIZE):

    train_acc = custom_dataset(f_1, 0, True)
    test_acc = custom_dataset(f_1, 0, False)
    train_rej = custom_dataset(f_2, 1, True)
    test_rej = custom_dataset(f_2, 1, False)

    train_loader = torch.utils.data.DataLoader(train_acc + train_rej, batch_size = N_BATCH_SIZE, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_acc + test_rej, batch_size = N_BATCH_SIZE, shuffle = False)

    return train_loader, test_loader

