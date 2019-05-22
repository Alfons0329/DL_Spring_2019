"""
Neuron style transfer that juvenile photos.

"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import argparse

########### ARGPARSE  #############
parser = argparse.ArgumentParser()
parser.add_argument('--style_path', type = str, default = 'style_img/')
parser.add_argument('--content_path', type = str, default = 'content_img/')
parser.add_argument('--output_path', type = str, default = 'output_img/')
parser.add_argument('--style_img', type = str, default = 's1')
parser.add_argument('--content_img', type = str, default = 'i1')
parser.add_argument('--output_img', type = str, default = 'o1')
# parser.add_argument('--img_num', type = int, default = 32000)
args = parser.parse_args()

########## USE CUDA      ##########
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LOAD IMAGE AND PREPROCESS #######
imsize = 512 if torch.cuda.is_available() else 128 # use small size if no cpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.CenterCrop(512),
    transforms.ToTensor()
    ])  # transform it into a torch tensor

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

style_img = image_loader(args.style_path + args.style_img)
content_img = image_loader(args.content_path + args.content_img)

assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

########## SHOW + SAVE IMG ########
unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

def imsave(tensor, title=None):
    plt.figure()
    image = tensor.cpu().clone()  # clone clone and move to GPU
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    # plt.imshow(image)
    if title is not None:
        plt.title(title)
        plt.imsave(title, image)

########## DEFINE LOSS ############
class content_loss(nn.Module):
    def __init__(self, target):
        super(content_loss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t()) # gram product
    G /= (a * b * c * d) # normalize
    return G

class style_loss(nn.Module):
    def __init__(self, target_feature):
        super(style_loss, self).__init__()
        self.target = gra,_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

########## PRETEAINED CNN #########
cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

def get_style_model_and_loss():
    return 0

if __name__ == '__main__':
    plt.figure()
    imshow(style_img, title='Style Image')

    plt.figure()
    imshow(content_img, title='Content Image')

