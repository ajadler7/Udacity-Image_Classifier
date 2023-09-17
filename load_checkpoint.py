from collections import OrderedDict
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def load_checkpoint(save_dir):
    checkpoint = torch.load(save_dir+'/checkpoint.pth')
    model = models.arch(pretrained=True)
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hiden_units),
                          ('relu1', nn.ReLU()),
                          ('do1', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = (checkpoint['class_to_idx'])
    epochs = (checkpoint['epochs'])
    
    return model