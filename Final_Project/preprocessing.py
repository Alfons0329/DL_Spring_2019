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
parser.add_argument('--train_path', type = str, default = 'alfons/DL_Final_Train_Data/imdb_crop')
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--img_num', type = int, default = 32000)
args = parser.parse_args()

TRAIN_PATH = args.train_path
N_BATCH_LIMIT = int(args.img_num / args.batch_size)
N_BATCH_SIZE = args.batch_size

N_CPU_THREADS = 6
torch.multiprocessing.set_sharing_strategy('file_system')  # prevent multithread data error

########### IO PREPROCESS ##########
def IO_preprocess(shuffle_or_not):
    # image rgb range [0, 255] -> [0.0, 1.0] and -> [-1, 1]
    # 0430 delete random crop to see if better and force the shuffle to be true
    my_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # NOTE: The order of tr
    train_input = torchvision.datasets.ImageFolder(root = TRAIN_PATH, transform = my_transform)
    train_loader = torch.utils.data.DataLoader(train_input, batch_size = N_BATCH_SIZE, num_workers = N_CPU_THREADS, shuffle = True)

    # test_input = torchvision.datasets.ImageFolder(root = valid_path, transform = my_transform)
    # test_loader = torch.utils.data.DataLoader(test_input, batch_size = N_BATCH, num_workers = N_CPU_THREADS, shuffle = False
    return train_loader

if __name__ == '__main__':
    train_loader = IO_preprocess(True)
    train_loader = train_loader[:N_BATCH_LIMIT]
    print('len train_loader: ', len(train_loader))


