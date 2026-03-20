import torch
import argparse
import cv2
import numpy as np
import torch.nn as nn
from model import build_model
from torchvision import transforms
import torch.nn.functional as F

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model', default='../outputs/model.pth',
    help='path to the trained model'
)
parser.add_argument(
    '-i', '--input', 
    help='path to the input image'
)
args = vars(parser.parse_args())

# computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# build the model
model = build_model(pretrained=False, fine_tune=False).to(device)
# load the trained model weights
model.load_state_dict(torch.load(args['model']))

# load the image
image = cv2.imread(args['input'])
# define the transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
])

# get the image label
# convert to list
class_names = open('outputs/class_names.txt', 'r').read().splitlines()

# prediction function
def predict(image, model):
    model.eval()
    with torch.no_grad():
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        
        # forward pass
        outputs = model(image)
        # get the prediction
        # _, pred = torch.max(outputs.data, 1)

        # get the probabilities for each class
        probabilities = F.softmax(outputs, dim=1)
        
    print(f"Class Probabilities:")
    for i, prob in enumerate(probabilities[0]):
        print(f" {class_names[i]}: {prob.item():.4f}")
        
    # print(probabilities)

    return probabilities

if __name__ == '__main__':
    predict(image, model)
