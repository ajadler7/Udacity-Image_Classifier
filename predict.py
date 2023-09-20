# PROGRAMMER: Aaron J. Adler
# DATE CREATED: September 14, 2023                                
# REVISED DATE: 
# PURPOSE:To create an image classfifer program that will take in
# the necessary paramaters to train a model, then run a photo through that model
# and give the topk probabilities for the classes

from load_checkpoint import load_checkpoint
from process_image import process_image
import numpy
import matplotlib.pyplot as plt
import json
from im_show import im_show
import argparse
from get_input_args import get_input_args
import torch



def main():
    in_arg = get_input_args()
    #check_command_line_arguments(in_arg)

    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
     
    # TODO: Implement the code to predict the class from an image file
    loaded_model = load_checkpoint(in_arg.save_dir)
    image = process_image(in_arg.image_path)
    image = torch.unsqueeze(image,0)
    map_dict =  {v: k for k, v in loaded_model.class_to_idx.items()}

    if in_arg.gpu == True and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    loaded_model.to(device)
    classes = []
    with torch.no_grad():
            log_ps = loaded_model.forward(image)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(in_arg.top_k, dim=1)
            probs = top_p
            
            probs = probs.flatten().tolist()
            top_indices = top_class[0].numpy().flatten()
           
            classes = [map_dict[x] for x in top_indices]
            
            
    #return probs, classes

    #%matplotlib inline
    #%config InlineBackend.figure_format = 'retina'



    print(probs,classes)

    flower_names = []

    for i in range(len(classes)):
        flower_names.append(cat_to_name[classes[i]])
    
    

    im_show(process_image(in_arg.image_path))

    plt.figure(figsize = [20, 5])
#plt.subplot(2,1,1)
    plt.barh(flower_names, probs )
    plt.xlabel('probability')
    plt.ylabel('flower name')

    plt.show()

if __name__ == "__main__":
    main()