from collections import OrderedDict
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
from get_input_args import get_input_args

def load_checkpoint(save_dir):
    in_arg = get_input_args()
    checkpoint = torch.load(save_dir+'/checkpoint.pth')
    if checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
        model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, checkpoint['hidden'])),
                              ('relu1', nn.ReLU()),
                              ('do1', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(checkpoint['hidden'], 102)),
                              ('output', nn.LogSoftmax(dim=1))
                               ]))
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(1024, checkpoint['hidden'])),
                            ('relu1', nn.ReLU()),
                            ('do1', nn.Dropout(0.2)),
                            ('fc2', nn.Linear(checkpoint['hidden'], 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    else:
        raise ValueError("Unsupported architecture: {}".format(in_arg.arch))

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = (checkpoint['class_to_idx'])
    epochs = (checkpoint['epochs'])
    
    return model