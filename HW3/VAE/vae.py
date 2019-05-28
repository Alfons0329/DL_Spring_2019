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
parser.add_argument('--lr', type = float, default = 1e-4)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--img_size', type = int, default = 128)

parser.add_argument('--train_path', type = str, default = 'cartoon/')
args = parser.parse_args()

########## GLOBAL DEF ###
N_IMG_SIZE = 0
N_FC1_SIZE = 400
N_FC2_SIZE = 20

N_EPOCH_LIMIT = 200
N_LEARN_RATE = args.lr
N_BATCH_SIZE = args.batch_size
N_IMG_SIZE = args.img_size

TRAIN_PATH = args.train_path

# save the model
model_path = 'my_vae.pt'
loss_path = 'best_loss.txt'
best_loss = 0.0

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
    plt.savefig(str(N_LEARN_RATE) + '_' + str(N_BATCH_SIZE) + '_' + 'LC' + '.png', dpi = 150)

########## VAE ##########
class VAE(nn.Module):
    def __init__(self): # init the vae layers
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(N_IMG_SIZE * N_IMG_SIZE, N_FC1_SIZE)
        self.fc21 = nn.Linear(N_FC1_SIZE, N_FC2_SIZE) # mean vector
        self.fc22 = nn.Linear(N_FC1_SIZE, N_FC2_SIZE) # standard deviation vector
        self.fc3 = nn.Linear(N_FC2_SIZE, N_FC1_SIZE) # sampled latent vector sapce
        self.fc4 = nn.Linear(N_FC1_SIZE, N_IMG_SIZE * N_IMG_SIZE) # final output result

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

def show_reconstructed(recon_x, x, cur_epoch):
    recon_x = recon_x / 2 + 0.5
    x = x / 2 + 0.5

    npimg_recon_x = recon_x.cpu().detach().numpy()
    npimg_x = x.cpu().detach().numpy()

    plt.imshow(np.transpose(npimg_recon_x, (1, 2, 0)))
    plt.title('Reconstructed')
    #plt.show()
    plt.savefig(str(N_LEARN_RATE) + '_' + str(N_BATCH_SIZE) + '_' + str(cur_epoch) + '_' + 'recon_x' + '.png')

    plt.imshow(np.transpose(npimg_x, (1, 2, 0)))
    plt.title('Ground Truth')
    #plt.show()
    plt.savefig(str(N_LEARN_RATE) + '_' + str(N_BATCH_SIZE) + '_' + str(cur_epoch) + '_' + 'x' + '.png')

def show_generated(x, cur_epoch):
    x = x / 2 + 0.5

    npimg_x = x.cpu().detach().numpy()

    plt.imshow(np.transpose(npimg_x, (1, 2, 0)))
    #plt.show()
    plt.savefig(str(N_LEARN_RATE) + '_' + str(N_BATCH_SIZE) + '_' + str(cur_epoch) + '_' + 'gen' + '.png')

def loss_function(recon_x, x, mu, logvar):
    # recon_x is the reconstructed tensor(or image)
    # sum up BCE
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, N_IMG_SIZE * N_IMG_SIZE), reduction = 'sum')
    mse_loss = nn.MSELoss(reduction = 'mean')
    num_channel, img_dim = recon_x.shape
    x = x.view(num_channel, img_dim) # fit the shape to be (N_BATCH_SIZE * channel, image dimension)
    MSE = mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def train(train_loader, model, optimizer, cur_epoch, device):
    model.train()
    train_loss = 0
    inputs = None
    recon_batch = None

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(inputs) # return: batch reconstructed vector, mean and stdev
        loss = loss_function(recon_batch, inputs, mu, logvar) # data is the ground truth
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print('Epoch %5d loss: %.3f' %(cur_epoch, float(train_loss)))

    if cur_epoch != 0 and cur_epoch % 50 == 0 and inputs is not None and recon_batch is not None:
        ##### RECONSTRUCTED ######
        show_reconstructed(torchvision.utils.make_grid(recon_batch) \
                , torchvision.utils.make_grid(inputs), cur_epoch)

        ##### RANDOM GEN ######
        randn_noise = torch.randn(N_BATCH_SIZE, 3, N_IMG_SIZE, N_IMG_SIZE)
        generated_imgs, _, _ = model(randn_noise)
        show_generated(generated_imgs, cur_epoch)

    return float(train_loss) / float(N_BATCH_SIZE)

########## MAIN #########
if __name__ == '__main__':

    ##### LOAD DATASET ######
    train_loader = pre.load_dataset(True, N_BATCH_SIZE, N_IMG_SIZE, TRAIN_PATH)
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print('Train with CUDA ')
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = torch.nn.DataParallel(model)
        model.cuda()

    print('Start training, N_BATCH_SIZE = %4d, N_EPOCH_LIMIT = %4d, N_LEARN_RATE %f\n' %(N_BATCH_SIZE, N_EPOCH_LIMIT, N_LEARN_RATE))

    ##### MAIN TRAIN #########
    for cur_epoch in range(N_EPOCH_LIMIT):
        if has_pretrained is False:
            cur_loss = train(train_loader, model, optimizer, cur_epoch, device)
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
