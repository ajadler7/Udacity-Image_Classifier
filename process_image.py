from PIL import Image
import numpy as np
import torch

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    with Image.open(image) as im:
       # im_resized = im.thumbnail((256,256))
        im.thumbnail((256,256))
    (width, height)= im.size
    left = (width - 240)/2
    top = (height - 240)/2
    right = (width + 240)/2
    bottom = (height + 240)/2
    im.crop((left, top, right, bottom))
    
    np_image = np.array(im)
    np_image = np_image/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image-mean)/std
    
    np_image = np_image.transpose((2, 0, 1))
    
    return torch.Tensor(np_image)