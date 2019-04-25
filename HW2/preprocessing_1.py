import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import os, sys

########### GLOBAL DEF ############
classes = ('dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel') ### represent in [0. 9]
train_path = 'animal/train/'
valid_path = 'animal/val/'
N_CPU_THREADS = 12

########### IO PREPROCESS ##########
def IO_preprocess(b_size):
    # image rgb range [0, 255] -> [0.0, 1.0] and -> [-1, 1]
    my_transform = transforms.Compose([transforms.Resize((256, 256)),transforms.RandomCrop((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # NOTE: The order of tr
    train_input = torchvision.datasets.ImageFolder(root = train_path, transform = my_transform)
    train_loader = torch.utils.data.DataLoader(train_input, batch_size = b_size, num_workers = N_CPU_THREADS, shuffle = False)

    test_input = torchvision.datasets.ImageFolder(root = valid_path, transform = my_transform)
    test_loader = torch.utils.data.DataLoader(test_input, batch_size = b_size, num_workers = N_CPU_THREADS, shuffle = False)
    return train_loader, test_loader

def add_label(input_data, input_len):
    output_data = np.zeros((input_len, 2))
    class_idx = 0
    for j in range(len(input_data)):
        if j % 1000 == 0 and j != 0:
            class_idx += 1
        output_data[class_idx][0] = input_data[j]
        output_data[class_idx][1] = class_idx

    return output_data



