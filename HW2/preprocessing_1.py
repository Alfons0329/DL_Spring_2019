import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import os, sys

########### GLOBAL DEF ############

train_path = 'animal/train/'
valid_path = 'animal/val/'
N_BATCH_SIZE = int(sys.argv[2])

########### IO PREPROCESS ##########
def IO_preprocess():
    # image rgb range [0, 255] -> [0.0, 1.0] and -> [-1, 1]
    my_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_input = torchvision.datasets.ImageFolder(root = train_path, transform = my_transform)
    train_loader = torch.utils.data.DataLoader(train_input, batch_size = N_BATCH_SIZE, num_workers = 0, shuffle = False)

    test_input = torchvision.datasets.ImageFolder(root = valid_path, transform = my_transform)
    test_loader = torch.utils.data.DataLoader(train_input, batch_size = N_BATCH_SIZE, num_workers = 0, shuffle = False)
    return train_loader, test_loader

