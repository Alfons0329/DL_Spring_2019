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
import matplotlib.pyplot as plt
start_time = time.time()

if not os.path.exists('ckpt'):
    os.makedirs('ckpt')
    os.makedirs('output/old')
    os.makedirs('output/young')
    print('Done mkdir of output/old output/young ckpt')

###### Argparse option ######
# parameters
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--old_root', type=str, default='old/', help='root directory of the dataset')
parser.add_argument('--young_root', type=str, default='young/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--img_size', type=int, default=32, help='size of the data crop (squared assumed)')
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
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr * 0.5, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr * 0.5, betas=(0.5, 0.999))
print('optimizer_G: ', optimizer_G)
print('optimizer_D_A: ', optimizer_D_A)
print('optimizer_D_B: ', optimizer_D_B)

# TODO: Lr sheduler if needed (tuning after the whole architecture is finished)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
input_A = Tensor(opt.batch_size, opt.input_nc, opt.img_size, opt.img_size)
input_B = Tensor(opt.batch_size, opt.output_nc, opt.img_size, opt.img_size)
target_real = Variable(Tensor(opt.batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batch_size).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transform = transforms.Compose([transforms.Resize((opt.img_size, opt.img_size)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
old_set = torchvision.datasets.ImageFolder(opt.old_root, transform)
young_set = torchvision.datasets.ImageFolder(opt.young_root, transform)
old_loader = torch.utils.data.DataLoader(dataset=old_set,batch_size=opt.batch_size,shuffle=True, num_workers=opt.n_cpu)
young_loader = torch.utils.data.DataLoader(dataset=young_set,batch_size=opt.batch_size,shuffle=True, num_workers=opt.n_cpu)

###################################
# List to be used for collecting number for graphing
G_loss  = []
DA_loss  = []
DB_loss  = []
epoch_list = []

def make_graph():
    # plot the loss of both discriminators
    plt.clf()
    title_str = 'LR= ' + str(opt.lr) + 'BAT= ' + str(opt.batch_size)
    plt.title(title_str)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.plot(epoch_list, DA_loss, color = 'blue', label = 'Discriminator A loss')
    plt.plot(epoch_list, DB_loss, color = 'red', label = 'Discriminator B loss')
    plt.legend()
    plt.savefig(str(opt.lr) + '_' + str(opt.batch_size) + '_dis.png')

    # plot the loss of the generator
    plt.clf()
    title_str = 'LR= ' + str(opt.lr) + 'BAT= ' + str(opt.batch_size)
    plt.title(title_str)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.plot(epoch_list, G_loss, color = 'blue', label = 'Generator loss')
    plt.legend()
    plt.savefig(str(opt.lr) + '_' + str(opt.batch_size) + '_gen.png')

###### Training ######
for epoch in range(1, opt.epochs):
    i = 1 # batch_index
    print('Epoch %5d'%(epoch))

    batch_loss_G = 0.0
    batch_loss_DA = 0.0
    batch_loss_DB = 0.0
    for batch in zip(old_loader, young_loader):
        # Set model input
        A = torch.FloatTensor(batch[0][0])
        B = torch.FloatTensor(batch[1][0])
        real_A = Variable(input_A.copy_(A))
        real_B = Variable(input_B.copy_(B))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # TODO : calculate the loss for the generators, and assign to loss_G
        # Identity loss
        # Referencing to: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/322
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        # G_B2A(B) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        optimizer_G.step()

        ###################################
        # TODO : calculate the loss for a discriminator, and assign to loss_DA
        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_DA = (loss_D_real + loss_D_fake) * 0.5
        loss_DA.backward()
        optimizer_D_A.step()

        ###################################
        # TODO : calculate the loss for the other discriminator, and assign to loss_DB
        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_DB = (loss_D_real + loss_D_fake) * 0.5
        loss_DB.backward()
        optimizer_D_B.step()

        ###################################
        ''' Graphing method 1
        G_loss.append(loss_G.item())
        DA_loss.append(loss_DA.item())
        DB_loss.append(loss_DB.item())
        '''
        batch_loss_G += loss_G.item()
        batch_loss_DA += loss_DA.item()
        batch_loss_DB += loss_DB.item()

        # Progress report for every 100 batches
        if i % 100 == 0:
            print('Batch %4d' %(i), " loss_G: ", loss_G.data.cpu().numpy() ,", loss_D: ", (loss_DA.data.cpu().numpy() + loss_DB.data.cpu().numpy()))
        i = i + 1

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'ckpt/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'ckpt/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'ckpt/netD_A.pth')
    torch.save(netD_B.state_dict(), 'ckpt/netD_B.pth')

    G_loss.append(loss_G)
    DA_loss.append(loss_DA)
    DB_loss.append(loss_DB)
    epoch_list.append(epoch)

end_time = time.time()
print('Total cost time: ',time.strftime("%H hr %M min %S sec", time.gmtime(end_time - start_time)))

# TODO : plot the figure
make_graph()
