import torch.nn as nn
from torchvision import models

def build_model(pretrained=True, fine_tune=True, num_classes=3):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
        model = models.densenet121(pretrained=True)
    else:
        print('[INFO]: Not loading pre-trained weights')
        model = models.densenet121(pretrained=False)

    # Replace the classifier layer with a new one
    # The input feature size for the final layer of DenseNet-121 is 1024
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        # Enable training for all layers
        for params in model.parameters():
            params.requires_grad = True
    else:
        print('[INFO]: Freezing hidden layers and fine-tuning only the classifier...')
        # Disable training for the feature extraction layers
        for params in model.parameters():
            params.requires_grad = False

        # Enable training for the new classifier layer
        for param in model.classifier.parameters():
            param.requires_grad = True

    print('[INFO]: Finished model building.')
    # print(model)

    return model
