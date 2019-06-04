import itertools
import argparse

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torchvision
import numpy as np

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import weights_init_normal

import time

import os, sys
start_time = time.time()

if not os.path.exists('ckpt'):
    os.makedirs('output/animation')

# parameters
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--animation_root', type=str, default='', help='root directory of the dataset')
parser.add_argument('--cartoon_dataroot', type=str, default='', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--img_size', type=int, default=64, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()

print(opt)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

netG_A2B.to(device)
netG_B2A.to(device)
netD_A.to(device)
netD_B.to(device)

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

# TODO: Lr sheduler if needed (tuning after the whole architecture is finished)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
input_A = Tensor(opt.batch_size, opt.input_nc, opt.img_size, opt.img_size)
input_B = Tensor(opt.batch_size, opt.output_nc, opt.img_size, opt.img_size)
target_real = Variable(Tensor(opt.batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batch_size).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer ReplayBuffer()

# Dataset loader
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
animation_set = torchvision.datasets.ImageFolder(opt.animation_root, transform)
cartoon_set = torchvision.datasets.ImageFolder(opt.cartoon_root, transform)
animation_loader = torch.utils.data.DataLoader(dataset=animation_set,batch_size=opt.batch_size,shuffle=True)
cartoon_loader = torch.utils.data.DataLoader(dataset=cartoon_set,batch_size=opt.batch_size,shuffle=True)
###################################
G_loss  = []
DA_loss  = []
DB_loss  = []
###### Training ######
for epoch in range(1, epochs):
    i = 1 # index
    print('epoch %5d'%(epoch))
    for batch in zip(animation_loader, cartoon_loader):
        # Set model input
        A = torch.FloatTensor(batch[0][0])
        B = torch.FloatTensor(batch[1][0])
        real_A = Variable(input_A.copy_(A))
        real_B = Variable(input_B.copy_(B))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # Referencing to: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/322
        # G_A2B(B) should equal B if real B is fed
        # TODO : calculate the loss for the generators, and assign to loss_G
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        # G_B2A(B) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        loss_G = 1 + 1
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # TODO : calculate the loss for a discriminator, and assign to loss_D_A
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # TODO : calculate the loss for the other discriminator, and assign to loss_D_B
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        G_loss.append(loss_G.data[0])
        DA_loss.append(loss_D_A.data[0])
        DB_loss.append(loss_D_B.data.data[0])
        # Progress report
        if i % 100 == 0:
            print("loss_G : ", loss_G.data.cpu().numpy() ,",loss_D:", (loss_D_A.data.cpu().numpy() + loss_D_B.data.cpu().numpy()))
            i = 0
        i = i + 1
    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'ckpt/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'ckpt/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'ckpt/netD_A.pth')
    torch.save(netD_B.state_dict(), 'ckpt/netD_B.pth')

end_time = time.time()
print('Total cost time',time.strftime("%H hr %M min %S sec", time.gmtime(end_time - start_time)))

# TODO : plot the figure
