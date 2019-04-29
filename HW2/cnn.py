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
# simple argument parsing
ARGV_CNT = 6
if len(sys.argv) != ARGV_CNT:
    print('Error: usage: python3 cnn.py $learning_rate $batch_size $stride_size { --vgg_normal | --vgg_small } { ada | no_ada }')
    sys.exit(1)

N_LEARN_RATE = float(sys.argv[1])
N_BATCH_SIZE = int(sys.argv[2])
N_STRID_SIZE = int(sys.argv[3])
VGG_linear = str(sys.argv[4])
adaptive_lr = str(sys.argv[5])

linear_size = 0
if VGG_linear == '--vgg_small':
    linear_size = 1024
elif VGG_linear == '--vgg_normal':
    linear_size = 4096

# train data and epoch limit
N_TRAIN_DATA = 10000
N_TEST_DATA = 4000
N_EPOCH_LIMIT = 100

# define the classes, represent in [0, 9] will be better
classes = ('dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel')

# define the VGG 16 layer architecture
VGG16_arch = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
VGG16_arch_small = [64, 'M', 128, 'M', 256, 'M', 512, 'M'] # condense the upper 'VGG16_arch model'

############# FOR GRAPHING ############
epoch_list = []
learning_curve = []
train_acc_list = []
test_acc_list = []

def make_graph():
    # plot the learning curve
    plt.clf()
    title_str = 'Learning Curve, BATCH_SIZE = ' + str(N_BATCH_SIZE) + ', ETA = ' + str(N_LEARN_RATE)
    plt.title(title_str)
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy')

    plt.plot(epoch_list, learning_curve, color = 'blue', label = 'no norm')
    plt.legend() # show what the line represents
    plt.savefig(sys.argv[1] + '_' + 'LC' + '.png', dpi = 150)

    # plot the accuracy of training set and testing set
    plt.clf()
    title_str = 'Accuracy, BATCH_SIZE = ' + str(N_BATCH_SIZE) + ', ETA = ' + str(N_LEARN_RATE)
    plt.title(title_str)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.plot(epoch_list, train_acc_list, color = 'blue', label = 'train acc')
    plt.plot(epoch_list, test_acc_list, color = 'red', label = 'train acc')
    plt.legend()
    plt.savefig(sys.argv[1] + '_' + 'ACC' + '.png', dpi = 150)

############# NN MAIN PART ############
class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
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
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # flatten to be input to classifier
        x = self.classifier(x)
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

        #if i % N_BATCH_SIZE == 0:
        learning_curve.append(train_loss)
        print('[Epoch %5d batch %5d ith_data %5d] CE loss: %.3f' %(cur_epoch, batch_cnt, i, train_loss))
        batch_cnt += 1
        train_loss = 0.0

############# VALIDATE NN ##############
def validate(val_loader, model, criterion, cur_epoch, device, what):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            class_predicted = (predicted == labels).squeeze()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(4):
                label = labels[i]
                class_correct[label] += class_predicted[i].item()
                class_total[label] += 1

    print('Accuracy on %5s set of %d images is %f' %(what, N_TEST_DATA, float(correct) / float(total)))

    for i in range(len(classes)):
        print('Accuracy on %5s set of %10s class is %f' %(what, classes[i], float(class_correct[i]) / float(class_total[i])))

############# DEBUG SHOW #############
def img_show(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

############# MAIN FUNCT #############
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

    print('Start training, batch = %5d, total epoch = %5d\n'%(N_BATCH_SIZE, N_EPOCH_LIMIT))
    for cur_epoch in range(N_EPOCH_LIMIT):
        if adaptive_lr == 'ada':
            N_LEARN_RATE /= 5

        epoch_list.append(cur_epoch)
        optimizer = optim.SGD(model.parameters(), lr = N_LEARN_RATE, momentum = 0.9)
        train(train_loader, model, criterion, optimizer, cur_epoch, device)
        validate(train_loader, model, criterion, cur_epoch, device, 'train')
        validate(train_loader, model, criterion, cur_epoch, device, 'test')

    make_graph()
