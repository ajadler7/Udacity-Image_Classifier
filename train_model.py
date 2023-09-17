#  Aaron J. Adler

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
from get_input_args import get_input_args

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg19 = models.vgg16(pretrained=True)

models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg19}

def TrainModel(data_directory, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
        #self.data_directory = data_directory
        #self.save_dir = save_dir
        #self.arch = arch
        #self.learning_rate = learning_rate
        #self.hidden_units = hidden_units
        #self.epochs = epochs
        #self.gpu = gpu

    in_arg = get_input_args()
    #check_command_line_arguments(in_arg)   
    
        
    train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    test_data_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    val_data_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    train_dir = in_arg.data_directory+'/train'
    valid_dir = in_arg.data_directory+'/valid'
    test_dir = in_arg.data_directory+'/test'
    
    train_data = datasets.ImageFolder(train_dir, transform=train_data_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_data_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=val_data_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    
    model = models[in_arg.arch]
    
    if in_arg.gpu==True:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu" )  
    else:
        device = "cpu"
    
    for param in model.parameters():
        param.requires_grad = False


    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, in_arg.hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('do1', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(in_arg.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)
    
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(in_arg.epochs):
        for inputs, labels in trainloader:
            steps += 1
        # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                    # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{in_arg.epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(validloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': 25088,
              'output_size': 102,
              'learning rate': in_arg.learning_rate,
              'epochs': in_arg.epochs,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx':model.class_to_idx}

    torch.save(checkpoint, self.save_dir+'/checkpoint.pth')