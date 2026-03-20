from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGE_SIZE = (224, 224)

# train transforms
def get_train_transform(image_size):
    train_transform = transforms.Compose([
        # zoom and resize image
        transforms.RandomResizedCrop(IMAGE_SIZE[0], scale=(0.8, 1.2)),
        # horizontal flipping
        transforms.RandomHorizontalFlip(),
        # random rotation +/- 20 degrees
        transforms.RandomRotation(20),
        # brightness adjustments
        transforms.ColorJitter(brightness=0.2),
        # conversion to tensor
        transforms.ToTensor(),
        # nomalisation
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    return train_transform

# validation transforms
def get_valid_transform(image_size):
    valid_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    return valid_transform

def get_datasets(train_dir, valid_dir):
    train_transform = get_train_transform(IMAGE_SIZE)
    valid_transform = get_valid_transform(IMAGE_SIZE)
    train_dataset = datasets.ImageFolder(
        train_dir, 
        transform=train_transform
    )
    valid_dataset = datasets.ImageFolder(
        valid_dir,
        transform=valid_transform
    )
    return train_dataset, valid_dataset

def get_dataloaders(train_dataset, valid_dataset, batch_size):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=4
    )
    return train_loader, valid_loader
