import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import os, sys
import argparse


########### GLOBAL DEF ############

########### ARGPARSE  #############
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type = str, default = 'cartoon/')
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--img_size', type = int, default = 224)
args = parser.parse_args()

TRAIN_PATH = args.train_path
N_BATCH_SIZE = args.batch_size
N_IMG_SIZE = args.img_size
N_CPU_THREADS = 12
torch.multiprocessing.set_sharing_strategy('file_system')  # prevent multithread data error

########### IO PREPROCESS ##########
def load_dataset(shuffle_or_not):
    # my_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    my_transform = transforms.Compose([transforms.CenterCrop((350, 350)), transforms.Resize((N_IMG_SIZE, N_IMG_SIZE)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_input = torchvision.datasets.ImageFolder(root = TRAIN_PATH, transform = my_transform)
    train_loader = torch.utils.data.DataLoader(train_input, batch_size = N_BATCH_SIZE, num_workers = N_CPU_THREADS, shuffle = True)

    return train_loader

if __name__ == '__main__':
    train_loader = load_dataset(True)
    print('len train_loader: ', len(train_loader))
    # try solving with enumerate
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        print('len inputs ', len(inputs), 'len labels ', len(labels), 'counter i ', i)


