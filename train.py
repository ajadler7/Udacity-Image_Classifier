# PROGRAMMER: Aaron J. Adler
# DATE CREATED: September 14, 2023                                
# REVISED DATE: 
# PURPOSE:To create an image classifier program that will take in
# the necessary parameters to train a model, then run a photo through that model
# and give the topk probabilities for the classes

#import necessary packages/functions
from get_input_args import get_input_args
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
from train_model import train_model
from workspace_utils import keep_awake


def main():
    in_arg = get_input_args()
    #check_command_line_arguments(in_arg)
    
    train_model(in_arg.data_directory, in_arg.save_dir, in_arg.arch, in_arg.learning_rate,
               in_arg.hidden_units, in_arg.epochs, in_arg.gpu)
    

# Call to main function to run the program
if __name__ == "__main__":
    main()