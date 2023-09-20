# Create Parse using ArgumentParser
import argparse

def get_input_args():
    parser = argparse.ArgumentParser()
    
    # Create 3 command line arguments as mentioned above using add_argument() from ArgumentParser method
    parser.add_argument('--data_directory')
    parser.add_argument('--save_dir', default = '')
    parser.add_argument('--arch', default = 'vgg19')
    parser.add_argument('--learning_rate', default = .001, type=float)
    parser.add_argument('--hidden_units', default = '5000', type=int)
    parser.add_argument('--epochs', default = '3', type=int)
    parser.add_argument('--image_path')
    parser.add_argument('--top_k', default='5', type=int)
    parser.add_argument('--category_names')
    parser.add_argument('--gpu', default = True, type=bool)
    
    
    return parser.parse_args()