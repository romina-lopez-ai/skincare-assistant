import torch
import argparse
import torch.nn as nn
from tqdm.auto import tqdm
from model import build_model
from dataset import get_valid_transform
from torch.utils.data import DataLoader
from torchvision import datasets

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

# build the model
model = build_model(pretrained=False, fine_tune=False).to(device)
# load the trained model weights
model.load_state_dict(torch.load(args['model']))

# load the test dataset
test_dataset = datasets.ImageFolder(
    args['data'],
    transform=get_valid_transform((224, 224))
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4
)

# evaluation function
def evaluate(model, testloader):
    model.eval()
    print('Evaluating')
    running_correct = 0

    # Collect labels
    all_labels = []
    all_preds = []  
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            
            # forward pass
            outputs = model(image)
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)

            # Collect labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            running_correct += (preds == labels).sum().item()

            # Confusion Matrix
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())   
        
    accuracy = 100. * (running_correct / len(testloader.dataset))
    print(f"Test accuracy: {accuracy:.3f}")

    # Confusion matrix - Plotting
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=test_dataset.classes)
    disp.plot(xticks_rotation=45, cmap='Blues', colorbar=False)
    plt.title('Confusion Matrix (Test)')
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    evaluate(model, test_loader)
