############# IMPORT MODULE #########
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import regex as re
import os, sys

from PIL import Image

############# MY PREPROCESSS #########
import preprocessing_1 as pre

############# GLOBAL DEF #############
# define the classes, represent in [0, 9] will be better
classes = ('dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel')
# define the VGG 16 layer architecture
net_one_layer16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M5', "FC1", "FC2", "FC"]
train_path = 'animal/train/'
valid_path = 'animal/val/'

N_LEARN_RATE = int(sys.argv[1])
N_BATCH_SIZE = int(sys.argv[2])
N_STRID_SIZE = int(sys.argv[3])

N_TRAIN_DATA = 10000
N_EPOCH_LIMIT = 100

############# NN MAIN PART ############
class vgg_net(nn.Module):
    def __init__(self, net_arch, num_classes):
        ########## NN one_layerITECTURE ###
        super(vgg_net, self).__init__() # inherit from father nn.module object
        self.num_classes = num_classes
        layers = []
        in_channels = 3
        for one_layer in net_arch:
            if one_layer == 'M': # the max pooling layer
                layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
            elif one_layer == 'M5':
                layers.append(nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1))
            elif one_layer == "FC1":
                layers.append(nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, padding = 6, dilation = 6))
                layers.append(nn.ReLU(inplace = True))
            elif one_layer == "FC2":
                layers.append(nn.Conv2d(1024,1024, kernel_size = 1))
                layers.append(nn.ReLU(inplace = True))
            elif one_layer == "FC":
                layers.append(nn.Conv2d(1024,self.num_classes, kernel_size = 1))
            else:
                layers.append(nn.Conv2d(in_channels = in_channels, out_channels = one_layer, kernel_size = 3, padding = 1))
                layers.append(nn.ReLU(inplace = True))
                in_channels = one_layer

        self.vgg = nn.ModuleList(layers)

    def forward(self, input_data):
        x = input_data
        for one_layer in vgg:
            x = one_layer(x)

        return x # the final output

    def train(train_input, train_input_label, test_input, test_input_label, batch_size):
        criteron = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.vgg_net.params(), lr = N_LEARN_RATE, momentum = 0.9)
        train_input_label_batch = [[]]
        test_input_label_batch = [[]]

        for batch_cnt in range(0,N_TRAIN_DATA / batch_size):
            train_input_label_batch.append(train_input_label[batch_cnt * batch_size: (batch_cnt + 1) * batch_size])

        for cur_epoch in range(0, N_EPOCH_LIMIT):
            print('len input batch %d len train batch %d' %len(train_input_label_batch), len(train_input))
            together = zip(train)
            #for img in zip(train_input, train_input_label):

############# CUUUUUUUDA #############
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############# DEBUG SHOW #############
def img_show(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    train_input, test_input = pre.IO_preprocess(1) # make them together
    train_input_label, test_input_label = pre.add_label()
    print(len(train_input), len(train_input_label))
    print(len(test_input), len(test_input_label))

    cnt = 0
    for each_img, each_label in zip(train_input, train_input_label):
        #img_show(torchvision.utils.make_grid(img))
        cnt += 1
        print('img %d with label %d' %(cnt, each_label))
        if cnt % 1000 == 0:
            input()

