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

########## MY OWN MODULE ##########

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
    def __init__(style_loss, self).__init__():
        self.target = gra,_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

########## USE CUDA      ##########
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########## PRETEAINED CNN #########
cnn = models.cnn19(pretrained = True).features
cnn = cnn.cuda() # move to CUDA

def get_style_model_and_loss():
    return 0
