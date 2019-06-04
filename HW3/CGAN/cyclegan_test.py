from torchvision.utils import save_image
import os
import sys

import itertools

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
###### Definition of variables ######
# TODO : assign input_nc and output_nc

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

# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

# Load state dicts
netG_A2B.load_state_dict(torch.load('ckpt/netG_A2B.pth'))
netG_B2A.load_state_dict(torch.load('ckpt/netG_B2A.pth'))

netG_A2B.to(device)
netG_B2A.to(device)

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Dataset loader
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
animation_set = torchvision.datasets.ImageFolder(animation_root, transform)
cartoon_set = torchvision.datasets.ImageFolder(cartoon_root, transform)
animation_loader = torch.utils.data.DataLoader(dataset=animation_set,batch_size=batchsize,shuffle=True)
cartoon_loader = torch.utils.data.DataLoader(dataset=cartoon_set,batch_size=batchsize,shuffle=True)


if not os.path.exists('output/animation'):
    os.makedirs('output/animation')
if not os.path.exists('output/cartoon'):
    os.makedirs('output/cartoon')

i = 0 # iteration count
for batch in zip(animation_loader, cartoon_loader):
    # Set model input
    A = torch.FloatTensor(batch[0][0])
    B = torch.FloatTensor(batch[1][0])
    real_A = Variable(input_A.copy_(A))
    real_B = Variable(input_B.copy_(B))

    # Generate output
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

    # Save image files
    save_image(real_A, 'output/animation/real%04d.png' % (i+1))
    save_image(real_B, 'output/cartoon/real%04d.png' % (i+1))
    save_image(fake_A, 'output/animation/fake%04d.png' % (i+1))
    save_image(fake_B, 'output/cartoon/fake%04d.png' % (i+1))

    sys.stdout.write('\rGenerated images %04d' % (i+1))
    i = i+1
    if (i == 10):
        break

sys.stdout.write('\n')
