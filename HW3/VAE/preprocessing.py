import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import os, sys
import argparse

########### GLOBAL DEF ############
N_CPU_THREADS = 12
torch.multiprocessing.set_sharing_strategy('file_system')  # prevent multithread data error

########### IO PREPROCESS ##########
def load_dataset(shuffle_or_not, N_BATCH_SIZE, N_IMG_SIZE, TRAIN_PATH):
    my_transform = transforms.Compose([transforms.Resize(N_IMG_SIZE), transforms.ToTensor()])
    train_input = torchvision.datasets.ImageFolder(root = TRAIN_PATH, transform = my_transform)
    train_loader = torch.utils.data.DataLoader(train_input, batch_size = N_BATCH_SIZE, num_workers = N_CPU_THREADS, shuffle = shuffle_or_not)

    return train_loader
