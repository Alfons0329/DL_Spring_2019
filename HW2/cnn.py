############# IMPORT MODULE #########
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt

import numpy as np
import regex as re
import os, sys, math

from PIL import Image

############# MY PREPROCESSS #########
import preprocessing_1 as pre

############# GLOBAL DEF #############
# define the classes, represent in [0, 9] will be better
classes = ('dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel')

# define the VGG 16 layer architecture

VGG16_arch = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
VGG16_arch_small = [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M']

VGG_linear = str(sys.argv[4])
linear_size = 0
if VGG_linear == '--vgg_small':
    linear_size = 1024
else:
    linear_size = 4096

train_path = 'animal/train/'
valid_path = 'animal/val/'

N_LEARN_RATE = float(sys.argv[1])
N_BATCH_SIZE = int(sys.argv[2])
N_STRID_SIZE = int(sys.argv[3])

N_TRAIN_DATA = 10000
N_EPOCH_LIMIT = 100


############# NN MAIN PART ############
class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, linear_size),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(linear_size, linear_size),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(linear_size, len(classes)),
                )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # if m is the convolutional layer, init it
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        #print('forward x shape ', x.size())
        x = x.view(x.size(0), -1) # flatten to be input to classifier
        #print('after view x shape ', x.size())
        x = self.classifier(x)
        #print('after classifier x shape ', x.size())
        return x


def make_layers(arch, batch_norm=False):
    layers = []
    in_channels = 3 # first channel lies in RGB
    for v in arch:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

############# TRAIN NN #################

def train(train_loader, model, criterion, optimizer, cur_epoch, device):
    print('\nEpoch: %d' % cur_epoch)
    train_loss = 0.0
    correct = 0
    total = 0
    batch_cnt = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if i % N_BATCH_SIZE == 0:
            print('[Epoch %5d batch %5d] CE loss: %.3f\n' %(cur_epoch, batch_cnt, train_loss))
            train_loss = 0.0

############# DEBUG SHOW #############
def img_show(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    ############# LOAD DATASET ###########
    train_loader, test_loader = pre.IO_preprocess(N_BATCH_SIZE, True) # make them together
    print(len(train_loader), len(test_loader))

    ############# CUUUUUUUDA #############
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if VGG_linear == '--vgg_small':
        model = VGG(make_layers(VGG16_arch_small))
    else:
        model = VGG(make_layers(VGG16_arch))

    if device == 'cuda':
        print('Train with CUDA ')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(model.parameters(), lr = N_LEARN_RATE, momentum = 0.9)
    print('Start training, batch = %5d, total epoch = %5d\n'%(N_BATCH_SIZE, N_EPOCH_LIMIT))
    for cur_epoch in range(0, N_EPOCH_LIMIT):
        train(train_loader, model, criterion, optimizer, cur_epoch, device)
