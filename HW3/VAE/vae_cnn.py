########## IMPORT #######
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

from torch.autograd import Variable

import argparse, os, sys, numpy
import matplotlib.pyplot as plt

########## ARGS #########
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 1e-4)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--img_size', type = int, default = 64)
parser.add_argument('--epochs', type = int, default = 200)
parser.add_argument('--activate', type = int, default = 0)

parser.add_argument('--train_path', type = str, default = './cartoon')
args = parser.parse_args()

############# USE CUDA ################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    print('Train with CUDA ')

############# FOR GRAPHING ############
epoch_list = []
learning_curve = []

def make_graph():
    # plot the learning curve
    plt.clf()
    title_str = 'BAT= ' + str(args.batch_size) + ' ETA= ' + str(args.lr)
    plt.title(title_str)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.plot(epoch_list, learning_curve, color = 'blue', label = 'VAE')
    plt.legend()
    plt.savefig(str(args.lr) + '_' + str(args.batch_size) + '_' + 'LC_cnn' + '.png', dpi = 150)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
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


def loss_fn(recon_x, x, mu, logvar, kla):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD * kla, BCE, KLD * kla

##### LOAD DATASET #######
dataset = datasets.ImageFolder(root='./cartoon', transform=transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.ToTensor(),
]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
print("Image count: %d, Batch count: %d" % (len(dataset.imgs), len(dataloader)))

# Check if the shape is correct
fixed_x, _ = next(iter(dataloader))
print("Batch shape: {}".format(fixed_x.shape))

image_channels = fixed_x.size(1)

##### INSTANTIATE #######
model = VAE(image_channels=image_channels).to(device)

##### OPTIMIZER #########
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

##### RANDOM NOISE ######
zzz = torch.zeros([32, 32]).to(device)
zzz.normal_()

##### KL ANNEALING ######
kla_t = 200
kla_i = 0

##### TRAINING ##########
print('Start training, lr: %s' %(args.lr))
for epoch in range(args.epochs):
    train_loss = 0.0
    for idx, (img, _) in enumerate(dataloader):
        kla_i = (kla_i + 1) % kla_t
        kla = kla_i / kla_t

        x = Variable(img).to(device)
        recon_img, mu, logvar = model(x)

        loss, bce, kld = loss_fn(recon_img, x, mu, logvar, kla)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if idx == len(dataloader) - 1 and epoch != 0 and epoch % 5 == 0:
            to_print = "Epoch[{}/{}] Idx: {} Loss: {:.3f} BCE: {:.3f} KLD: {:.3f} kla: {}".format(epoch + 1,args.epochs,idx, loss.item()/args.batch_size, bce.item()/args.batch_size, kld.item()/args.batch_size, kla)
            print(to_print)

            save_image(img.data.cpu(), '%d_%f_%d_x_cnn.png' %(args.batch_size, args.lr, epoch + 1))
            save_image(recon_img.data.cpu(), '%d_%f_%d_recon_x_cnn.png' %(args.batch_size, args.lr, epoch + 1))

            learning_curve.append(train_loss)
            epoch_list.append(epoch)

    gen_img = model.decode(zzz)
    save_image(gen_img.data.cpu(), '%d_%f_%d_gen_cnn.png' %(args.batch_size, args.lr, epoch + 1))
    torch.save(model.state_dict(), 'dict')

make_graph()

