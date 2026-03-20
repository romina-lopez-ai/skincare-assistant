import torch
import argparse
import torch.nn as nn
from tqdm.auto import tqdm
from model import build_model # Assuming this is defined in an external file (model.py)
from dataset import get_valid_transform # Assuming this is defined in an external file (dataset.py)
from torch.utils.data import DataLoader
from torchvision import datasets

# Evaluation function (No changes to logic)
def evaluate(model, testloader, device):
    model.eval()
    print('Evaluating')
    running_correct = 0
    with torch.no_grad():
        # The tqdm display is confirmed to show '1/1' for the current small dataset
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            
            # forward pass
            outputs = model(image)
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            running_correct += (preds == labels).sum().item()
            
    accuracy = 100. * (running_correct / len(testloader.dataset))
    print(f"Test accuracy: {accuracy:.3f}")

if __name__ == '__main__':
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', default='../outputs/model.pth',
        help='path to the trained model'
    )
    parser.add_argument(
        '-d', '--data', default='../data/test',
        help='path to the test data'
    )
    args = vars(parser.parse_args())

    # computation device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # Move model building and loading inside the if __name__ == '__main__': block.
    # This prevents worker processes from re-executing this section, solving the repeated log issue.
    print("[INFO]: Building and loading model...")
    model = build_model(pretrained=False, fine_tune=False).to(device)
    model.load_state_dict(torch.load(args['model']))
    print("[INFO]: Model loaded successfully.")

    # load the test dataset
    test_dataset = datasets.ImageFolder(
        args['data'],
        transform=get_valid_transform((224, 224))
    )
    # To avoid the process fork issue on Windows, either set num_workers=0 (single process) 
    # or ensure the main logic is within if __name__ == '__main__': (which we did above).
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    
    # Run the evaluation function
    evaluate(model, test_loader, device)
