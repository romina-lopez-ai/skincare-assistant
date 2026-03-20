import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
import os
import json
from tqdm.auto import tqdm

# NEW: plotting & metrics
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
    classification_report,
)

from model import build_model
from dataset import get_datasets, get_dataloaders


# Argument parser

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=20,
    help="number of epochs to train our network for",
)
parser.add_argument(
    "-lr",
    "--learning-rate",
    type=float,
    dest="learning_rate",
    default=0.001,
    help="learning rate for training the model",
)
parser.add_argument(
    "-ft",
    "--fine-tune",
    dest="fine_tune",
    action="store_true",
    help="whether to fine-tune all layers or not",
)
parser.add_argument(
    "-p",
    "--pretrained",
    action="store_true",
    help="whether to use pretrained model or not",
)
parser.add_argument(
    "-s",
    "--save-model",
    dest="save_model",
    action="store_true",
    help="whether to save the model or not",
)
parser.add_argument(
    "-o",
    "--out-dir",
    dest="out_dir",
    default="outputs",
    help="directory to save models, plots, and metrics",
)
args = vars(parser.parse_args())


def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print("Training")
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation.
        loss.backward()
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100.0 * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


def validate(model, testloader, criterion, device):
    model.eval()
    print("Validation")
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100.0 * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

# Main loop

if __name__ == "__main__":
    # Prepare output directory
    out_dir = args["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Load the training and validation datasets.
    train_dataset, valid_dataset = get_datasets("data/train", "data/validation")
    print(f"[INFO]: Number of training images: {len(train_dataset)}")
    print(f"[INFO]: Number of validation images: {len(valid_dataset)}")

    # Load the training and validation data loaders.
    train_loader, valid_loader = get_dataloaders(
        train_dataset, valid_dataset, batch_size=32
    )

    # Learning_parameters.
    lr = args["learning_rate"]
    epochs = args["epochs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")

    # Build the model
    model = build_model(
        pretrained=args["pretrained"],
        fine_tune=args["fine_tune"],
        num_classes=len(train_dataset.classes),
    ).to(device)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")

    # Optimizer and loss.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model, train_loader, optimizer, criterion, device
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model, valid_loader, criterion, device
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(
            f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}"
        )
        print(
            f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}"
        )
        print("-" * 50)

    # Plots & Evaluation Artifacts

    # 1) Curves: loss & accuracy
    try:
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label="Train Loss")
        plt.plot(valid_loss, label="Valid Loss")
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_acc, label="Train Acc")
        plt.plot(valid_acc, label="Valid Acc")
        plt.title("Training vs Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()

        plt.tight_layout()
        curves_path = os.path.join(out_dir, "training_curves.png")
        plt.savefig(curves_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] {curves_path}")
    except Exception as e:
        print(f"[WARN] Failed to save training curves: {e}")

    # 2) Validation metrics, confusion matrix, classification report
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # Compute metrics
    try:
        acc = accuracy_score(all_labels, all_preds)
        f1_w = f1_score(all_labels, all_preds, average="weighted")

        # Save compact metrics JSON
        metrics = {
            "val_accuracy": round(acc * 100.0, 4),
            "val_f1_weighted": round(float(f1_w), 6),
            "num_val_samples": len(all_labels),
            "classes": valid_dataset.classes,
        }
        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[SAVED] {metrics_path}")
        print(f"[RESULTS] Validation Accuracy: {acc*100:.2f}%")
        print(f"[RESULTS] Weighted F1 Score: {f1_w:.4f}")
    except Exception as e:
        print(f"[WARN] Failed to compute/save metrics.json: {e}")

    # Classification report (per-class precision/recall/F1)
    try:
        report = classification_report(
            all_labels, all_preds, target_names=valid_dataset.classes, digits=4
        )
        report_path = os.path.join(out_dir, "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[SAVED] {report_path}")
        print("\n" + report)
    except Exception as e:
        print(f"[WARN] Failed to save classification report: {e}")

    # Confusion matrix
    try:
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=valid_dataset.classes
        )
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("Confusion Matrix (Validation)")
        cm_path = os.path.join(out_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] {cm_path}")
    except Exception as e:
        print(f"[WARN] Failed to save confusion matrix: {e}")

    # Save the trained model (optional)
    if args["save_model"]:
        model_path = os.path.join(out_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"[SAVED] {model_path}")
