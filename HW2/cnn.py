import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import regex as re


############# GLOBAL DEF ####### #####
classes = ('dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel')
train_path = 'animal/train/'
valid_path = 'animal/val/'


############# FILE IO ################
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show

def file_IO():
    train_input = torchvision.datasets.ImageFolder(root = train_path, transform = transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_input, batch_size = 128, num_workers = 0, shuffle = False)
    dataiter = iter(train_loader)
    images = dataiter.next()
    print(type(dataiter), type(train_loader))
    imshow(images)
    return train_loader

if __name__ == '__main__':
    train_loader = file_IO()
