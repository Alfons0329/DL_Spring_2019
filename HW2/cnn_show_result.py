"""
Reduced VGG16 version in this python code, original version is VGG16_arch and reduced is VGG16_arch_small due to the hardware limitation of compute capability
Please run with --vgg_normal if you have HW such like TITAN V

My HW as follows:
    CPU = i7 8700K @ 4GHz
    GPU = 1070 8GB
    RAM = 32GB DDR4
    SSD = PM981 512GB
"""
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
ARGV_CNT = 7
if len(sys.argv) != ARGV_CNT:
    print('Error: usage: python3 cnn.py $learning_rate $batch_size $stride_size { --vgg_normal | --vgg_small } { adam | sgd }')
    sys.exit(1)

N_LEARN_RATE = float(sys.argv[1])
N_BATCH_SIZE = int(sys.argv[2])
N_STRID_SIZE = int(sys.argv[3])
N_KERNE_SIZE = int(sys.argv[4])
VGG_linear = str(sys.argv[5])
adaptive_lr = str(sys.argv[6])

linear_size = 0
if VGG_linear == '--vgg_small':
    linear_size = 256
elif VGG_linear == '--vgg_normal':
    linear_size = 4096

# train data and epoch limit
N_TRAIN_DATA = 10000
N_TEST_DATA = 4000
N_EPOCH_LIMIT = 10

# define the classes, represent in [0, 9] will be better
classes = ('dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel')

# define the VGG 16 layer architecture
VGG16_arch = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
VGG16_arch_small = [8, 'M', 16, 'M', 32, 'M', 64, 'M'] # condense the upper 'VGG16_arch model'O
# save the model
model_path = 'my_vgg.pt'
acc_path = 'best_acc.txt'
best_acc = 0.0

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
    """

    plt.clf()
    title_str = 'STRI=' + str(N_STRID_SIZE) + ' KER=' + str(N_KERNE_SIZE) + ' LC, BAT=' + str(N_BATCH_SIZE) + ' ETA = ' + str(N_LEARN_RATE)
    plt.title(title_str)
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy')

    plt.plot(epoch_list, learning_curve, color = 'blue', label = 'no norm')
    plt.legend() # show what the line represents
    plt.savefig(adaptive_lr + '_' + str(N_LEARN_RATE) + '_' + str(N_BATCH_SIZE) + '_' + str(N_STRID_SIZE) + '_' + str(N_KERNE_SIZE) + '_' + 'LC' + '.png', dpi = 150)

    """
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
                nn.Linear(64 * 7 * 7, linear_size),
                nn.ReLU(True),
                nn.Dropout(),
                # try to save some computational resource
                #nn.Linear(linear_size, linear_size),
                #nn.ReLU(True),
                #nn.Dropout(),
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
            layers += [nn.MaxPool2d(kernel_size = 2, stride = N_STRID_SIZE)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size = N_KERNE_SIZE, padding = 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace = True)]
            else:
                layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = v
    return nn.Sequential(*layers)

############# TRAIN NN #################
def train(train_loader, model, criterion, optimizer, cur_epoch, device):
    train_loss = 0.0
    total = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
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
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    #mkdir = 'mkdir - p ' + what
    #os.system(mkdir)
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            class_predicted = (predicted == labels).squeeze()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            wrong_cnt = 0
            for each_input, each_output, each_label in zip(inputs, outputs, labels):
                _, each_predicted = torch.max(each_output, 0)

                if each_predicted != each_label:
                    #print('each input ', each_input)
                    print('each output ', each_output)
                    #print('each predicted ', each_predicted)
                    print('each predicted class', classes[each_predicted])
                    print('each label class', classes[each_label])
                    #input()

                    each_input = each_input / 2 + 0.5
                    img_name = what + '_' + classes[each_predicted] + '_' + classes[each_label] + str(wrong_cnt) + '.png'
                    torchvision.utils.save_image(each_input, img_name)
                    wrong_cnt += 1

            #print('labels ', labels)
            #print('class_predicted ', class_predicted)
            #print('class_correct ', class_correct)
            #print('class_total ', class_total)
            for i in range(4):
                label = labels[i]
                class_correct[label] += class_predicted[i].item()
                class_total[label] += 1

    if total != 0:
        print('Accuracy on %6s set of %d images is %f' %(what, total, float(correct) / float(total)))

        for i in range(len(classes)):
            if class_total[i] != 0:
                print('Accuracy on %5s set of %10s class with is %.3f' %(what, classes[i], float(class_correct[i]) / float(class_total[i])))
    # return the accuracy
        return float(correct) / float(total)
    else:
        return 0

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

    ############# MODEL SELECT / LOAD ####
    has_pretrained = False
    if VGG_linear == '--vgg_small':
        model = VGG(make_layers(VGG16_arch_small))
        if os.path.isfile(model_path): # if has a self-pretrained model, just fucking load it
            model = torch.load(model_path)
            #model.eval()
            if os.path.isfile(acc_path):
                f = open(acc_path, 'r')
                read_acc = (f.read())
                best_acc = float(read_acc)
                has_pretrained = True
            print('Has my own pretrained model, directly load it!')
            print('Current best acc ', best_acc)
            print(model)
        else:
            print('NoSelfModuleError: No pretraied module. Quit!')
            exit(1)
    else:
        model = VGG(make_layers(VGG16_arch))

    ############# CUUUUUUUDA #############
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        print('Train with CUDA ')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()

    ############# TRAINING ###############
    print('Start training, N_BATCH_SIZE = %4d, N_EPOCH_LIMIT = %4d, N_LEARN_RATE %f\n' %(N_BATCH_SIZE, N_EPOCH_LIMIT, N_LEARN_RATE))
    adaptive_lr_phase = [0.2, 0.5, 0.9]
    phase_idx = 0
    cur_acc = 0.0

    for cur_epoch in range(N_EPOCH_LIMIT):
        ############# ADA LEARN RATE ###############

        epoch_list.append(cur_epoch)
        # determine to use pretrained or not
        if has_pretrained == False:
            if adaptive_lr == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr = N_LEARN_RATE, momentum = 0.9, weight_decay = 5e-4)
            elif adaptive_lr == 'adam':
                optimizer = optim.Adam(model.parameters(), lr = N_LEARN_RATE, weight_decay = 5e-4)

            # 0501 changed from SGD to adam
            #train(train_loader, model, criterion, optimizer, cur_epoch, device)
        train_acc_list.append(validate(train_loader, model, criterion, cur_epoch, device, 'train'))

        print('')
        cur_acc = validate(test_loader, model, criterion, cur_epoch, device, 'test')
        test_acc_list.append(cur_acc)
        print('-----------------------------------------------\n')

        # save the model and corresponding accuracy if this is the final epoch with better result
        if cur_epoch == N_EPOCH_LIMIT - 1 and cur_acc > best_acc:
            print('Last epoch, better model with cur_acc %.3f over best_acc %.3f, save model and acc'%(cur_acc, best_acc))
            torch.save(model, model_path)
            f = open(acc_path, 'w')
            f.write(str(cur_acc))
            f.close()

    torch.cuda.empty_cache()
    make_graph()
