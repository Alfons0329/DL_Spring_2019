############# IMPORT MODULE #########
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import regex as re

############# MY PROPROCESSS #########
import preprocessing_1
############# GLOBAL DEF ####### #####
classes = ('dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel')
train_path = 'animal/train/'
valid_path = 'animal/val/'

############# FILE IO ################


if __name__ == '__main__':
	train_input, test_input = preprocessing_1.IO_preprocess()
