import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import os, sys

########### GLOBAL DEF ############
classes = ('butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel')
train_path = 'animal/train/'
valid_path = 'animal/val/'
N_CPU_THREADS = 1
torch.multiprocessing.set_sharing_strategy('file_system')  # prevent multithread data error
N_TRAIN_DATA = 10000
N_TEST_DATA = 4000

########### IO PREPROCESS ##########
def IO_preprocess(b_size, shuffle_or_not):
    # image rgb range [0, 255] -> [0.0, 1.0] and -> [-1, 1]
    my_transform = transforms.Compose([transforms.Resize((256, 256)),transforms.RandomCrop((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # NOTE: The order of tr
    train_input = torchvision.datasets.ImageFolder(root = train_path, transform = my_transform)
    train_loader = torch.utils.data.DataLoader(train_input, batch_size = b_size, num_workers = N_CPU_THREADS, shuffle = shuffle_or_not)

    test_input = torchvision.datasets.ImageFolder(root = valid_path, transform = my_transform)
    test_loader = torch.utils.data.DataLoader(test_input, batch_size = b_size, num_workers = N_CPU_THREADS, shuffle = shuffle_or_not)
    return train_loader, test_loader

########### ADD LABEL FOR DATA ##########
def add_label():
    train_label = []
    class_idx = 0
    for i in range(N_TRAIN_DATA):
        if i % 1000 == 0 and i != 0:
            class_idx = class_idx + 1
        train_label.append(class_idx)

    test_label = []
    class_idx = 0
    for i in range(N_TEST_DATA):
        if i % 400 == 0 and i != 0:
            class_idx = class_idx + 1
        test_label.append(class_idx)

    return train_label, test_label


