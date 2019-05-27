########## IMPORT #######
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt

import numpy as np
import os, sys, math

import argparse

from PIL import Image

###### MY PREPROCESS ####
import preprocessing as pre

########## ARGS #########
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = int, default = 64)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--img_size', type = int, default = 64)
args = parser.parse_args()

########## GLOBAL DEF ###
N_IMG_SIZE = 0
N_FC1_SIZE = 400
N_FC2_SIZE = 20

N_EPOCH_LIMIT = 200
N_LEARN_RATE = args.lr
N_BATCH_SIZE = args.batch_size
N_IMG_SIZE = args.img_size

# save the model
model_path = 'my_vae.pt'
loss_path = 'loss_acc.txt'
best_loss = 0.0

############# FOR GRAPHING ############
epoch_list = []
learning_curve = []
train_acc_list = []
test_acc_list = []

def make_graph():
    # plot the learning curve
    plt.clf()
    title_str = 'BAT=' + str(N_BATCH_SIZE) + ' ETA = ' + str(N_LEARN_RATE)
    plt.title(title_str)
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy')

    plt.plot(epoch_list, learning_curve, color = 'blue', label = 'no norm')
    plt.legend()
    plt.savefig(str(N_LEARN_RATE) + '_' + str(N_BATCH_SIZE) + '_' + 'LC' + '.png', dpi = 150)

########## VAE ##########
class VAE(nn.Module):
    def __init__(self): # init the vae layers
        super.fc1 = nn.Linear(N_IMG_SIZE * N_IMG_SIZE, N_FC1_SIZE)
        super.fc21 = nn.Linear(N_FC1_SIZE, N_FC2_SIZE) # mean vector
        super.fc22 = nn.Linear(N_FC1_SIZE, N_FC2_SIZE) # standard deviation vector
        super.fc3 = nn.Linear(N_FC2_SIZE, N_FC1_SIZE) # sampled latent vector sapce
        super.fc4 = nn.Linear(N_FC1_SIZE, N_IMG_SIZE * N_IMG_SIZE) # final output result

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, N_IMG_SIZE * N_IMG_SIZE))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE()
optimizer = optim.Adam(model.parameters, lr = 1e-4, weight_decay = 1e-4)

def show_reconstructed():
    return 0

def loss_function(recon_x, x, mu, logvar):
    # recon_x is the reconstructed tensor(or image)
    # sum up BCE
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, N_IMG_SIZE * N_IMG_SIZE), reduction = 'sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train():
    model.train()
    train_loss = 0

    for i, (data) in enumerate(train_loader):
        data = data.to(device)

    return 0

def gen_img():
    return 0

########## MAIN #########
if __name__ == '__main__':
    
    ##### LOAD DATASET ######
    train_loader, test_loader = pre.load_dataset(True, N_BATCH_SIZE, N_BATCH_SIZE) # make them together
    print(len(train_loader), len(test_loader))

    ########## MAIN #########
    device = 'cuda' if torch.cuda_is_available() else 'cpu'
    print('Start training, N_BATCH_SIZE = %4d, N_EPOCH_LIMIT = %4d, N_LEARN_RATE %f\n' %(N_BATCH_SIZE, N_EPOCH_LIMIT, N_LEARN_RATE))
