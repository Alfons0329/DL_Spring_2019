############# IMPORT MODULE #########
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import regex as re
import os, sys

from PIL import Image
############# MY PREPROCESSS #########
import preprocessing_1 as pre

############# GLOBAL DEF #############
classes = ('dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel')  # represent in [0, 9] will be better
train_path = 'animal/train/'
valid_path = 'animal/val/'

N_LEARN_RATE = int(sys.argv[1])
N_BATCH_SIZE = int(sys.argv[2])
N_STRID_SIZE = int(sys.argv[3])

N_TRAIN_DATA = 10000

############# GLOBAL DEF #############
class Net(nn.Module):
    def __init__(self):
        ########## NN ARCHITECTURE ### 20190426 todo
        super(Net, self).__init__() ## inherit from father nn.module object

############# CUUUUUUUDA #############
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############# DEBUG SHOW #############
def img_show(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    train_input, test_input = pre.IO_preprocess(N_BATCH_SIZE)
    train_input_label, test_input_label = pre.add_label()
    print(len(train_input), len(test_input))

    cnt = 0
    for img, label in train_input:
        print(train_input_label[cnt: (cnt + 1) * 1000])
        print(test_input_label[cnt: (cnt + 1) * 400])
        #mg_show(torchvision.utils.make_grid(img))
        cnt += 1

