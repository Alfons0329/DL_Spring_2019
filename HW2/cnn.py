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
import preprocessing_1

############# GLOBAL DEF ####### #####
classes = ('dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel')
train_path = 'animal/train/'
valid_path = 'animal/val/'
N_BATCH_SIZE = int(sys.argv[2])

#class NN(object):


############# DEBUG SHOW #############
def img_show(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show

if __name__ == '__main__':
    train_input, test_input = preprocessing_1.IO_preprocess()
    my_transform_2 = transforms.Compose([transforms.ToPILImage()])
    print(type(train_input), len(train_input), type(test_input), len(test_input))
    #train_input = my_transform_2(train_input)

    for epoch in range(100):
        for img, label in train_input:
            print(1)
            #do something of NN

