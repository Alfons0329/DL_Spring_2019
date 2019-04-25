############# IMPORT MODULE #########
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import regex as re
import os, sys

from PIL import Image
############# MY PROPROCESSS #########
import preprocessing_1 as pre

############# GLOBAL DEF ####### #####
classes = ('dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel')
train_path = 'animal/train/'
valid_path = 'animal/val/'
N_BATCH_SIZE = int(sys.argv[2])
N_TRAIN_DATA = 10000

############# DEBUG SHOW #############
def img_show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    train_input, test_input = pre.IO_preprocess(1)
    train_input_wlabel = pre.add_label(train_input, N_TRAIN_DATA)#with label now

    for each in range(len(train_input_wlabel)):
        print(type(train_input_wlabel[each][0]), train_input_wlabel[each][1])
        img_show(train_input_wlabel[each][0])

