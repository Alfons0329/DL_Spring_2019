########## IMPORT #######
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt

import numpy as np
import os, sys, math
import argparse

from PIL import Image

###### MY PREPROCESS ####
import preprocessing as pre

########## ARGS #########
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 1e-4)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--img_size', type = int, default = 64)
parser.add_argument('--activate', type = int, default = 0)

parser.add_argument('--train_path', type = str, default = './cartoon')
args = parser.parse_args()

########## GLOBAL DEF ###
N_IMG_SIZE = 0
N_FC1_SIZE = 64
N_FC2_SIZE = 16

N_EPOCH_LIMIT = 6000
N_LEARN_RATE = args.lr
N_BATCH_SIZE = args.batch_size
N_IMG_SIZE = args.img_size

TRAIN_PATH = args.train_path

# save the model
model_path = 'my_vae.pt'
loss_path = 'best_loss.txt'
best_loss = 0.0

############# USE CUDA ################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print('Train with CUDA ')

############# FOR GRAPHING ############
epoch_list = []
learning_curve = []
train_acc_list = []
test_acc_list = []

def make_graph():
    # plot the learning curve
    plt.clf()
    title_str = 'BAT= ' + str(N_BATCH_SIZE) + ' ETA= ' + str(N_LEARN_RATE)
    plt.title(title_str)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.plot(epoch_list, learning_curve, color = 'blue', label = 'norm 0.5')
    plt.legend()
    plt.savefig(str(N_LEARN_RATE) + '_' + str(N_BATCH_SIZE) + '_' + 'LC_cnn' + '.png', dpi = 150)

########## VAE ##########
class flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class unflatten(nn.Module):
    def forward(self, input, size = 1024):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels = 3, h_dim = 1024, z_dim = 32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(image_channels, 32, kernel_size = 4, stride = 2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size = 4, stride = 2),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size = 4, stride = 2),
                nn.ReLU(),
                flatten()
                )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
                unflatten(),
                nn.ConvTranspose2d(h_dim, 128, kernel_size = 5, stride = 2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size = 5, stride = 2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size = 6, stride = 2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, image_channels, kernel_size = 6, stride = 2),
                nn.ReLU()
                )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size()).to(device)
        z = mu + std * eps
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduce = 'sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

def train(train_loader, model, optimizer, cur_epoch):
    train_loss = 0
    inputs = None
    recon_imgs = None

    for i, (imgs, _) in enumerate(train_loader):
        x = Variable(imgs).to(device)
        recon_imgs, mu, logvar = model(x) # return: batch reconstructed vector, mean and stdev

        loss, mse, kld = loss_function(recon_imgs, x, mu, logvar) # data is the ground truth
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('Epoch %5d loss: %.3f' %(cur_epoch, float(train_loss)))
    if cur_epoch != 0 and cur_epoch % 5 == 0 and recon_imgs is not None:
        ##### RECONSTRUCT #####
        #save_image(inputs.cpu(), str(N_LEARN_RATE) + '_' + str(N_BATCH_SIZE) + '_' + str(cur_epoch) + '_' + 'x_cnn' + '.png')
        #save_image(recon_batch.cpu(), str(N_LEARN_RATE) + '_' + str(N_BATCH_SIZE) + '_' + str(cur_epoch) + '_' + 'recon_x_cnn' + '.png')

        ##### RANDOM GEN ######
        randn_noise = torch.zeros([32, 32]).to(device)
        randn_noise.normal_()
        generated_imgs = model.decode(randn_noise)
        print('zzz_my ', randn_noise)
        save_image(generated_imgs.cpu(), str(N_LEARN_RATE) + '_' + str(N_BATCH_SIZE) + '_' + str(cur_epoch) + '_' + 'gen_cnn' + '.png')

    return float(train_loss)

########## MAIN #########
if __name__ == '__main__':

    ##### LOAD DATASET ######
    train_loader = pre.load_dataset(False, N_BATCH_SIZE, N_IMG_SIZE, TRAIN_PATH)
    print('Train loader type %s with length %d ' %(type(train_loader), len(train_loader)))

    ##### LOAD PRETRAINED ###
    has_pretrained = False
    if os.path.isfile(model_path) and os.path.isfile(loss_path):
        model = torch.load(model_path)
        f = open(loss_path, 'r')
        best_loss = float(f.read())
        has_pretrained = True
        print('Has my own pretrained model, directly load it!')
        print('Current best acc ', best_acc)
        print(model)
    else:
        print('No my own pretrained model')
        model = VAE()

    ##### OPTIMIZER #########
    optimizer = optim.Adam(model.parameters(), lr = N_LEARN_RATE, weight_decay = 8e-4)

    ##### CUDA ##############
    model = model.to(device)

    print('Start training, N_BATCH_SIZE = %4d, N_EPOCH_LIMIT = %4d, N_LEARN_RATE %f\n' %(N_BATCH_SIZE, N_EPOCH_LIMIT, N_LEARN_RATE))

    ##### MAIN TRAIN #########
    for cur_epoch in range(1, N_EPOCH_LIMIT + 1):
        if has_pretrained is False:
            cur_loss = train(train_loader, model, optimizer, cur_epoch)
            learning_curve.append(cur_loss)

        epoch_list.append(cur_epoch)

        if cur_epoch == N_EPOCH_LIMIT - 1 and cur_loss < best_loss:
            print('Last epoch, better model with cur_loss %.3f over best_loss %.3f, save model and acc'%(cur_acc, best_acc))
            torch.save(model, model_path)
            f = open(loss_path, 'w')
            f.write(str(cur_loss))
            f.close()

    torch.cuda.empty_cache()
    make_graph()

