import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm
import os

def train(model, train_loader, criterion, optimizer, device):
    """
    Function to train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc='Training loop'):
        # Move inputs and labels to the device
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    return train_loss


def evaluate(model, val_loader, criterion, device):
    """
    Function to evaluate the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation loop'):
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)
         
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
    val_loss = running_loss / len(val_loader.dataset)
    return val_loss

def save_model(model, path="models/saved_model/card_classifier.pth"):

    # Ensure the path is relative to the main project directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    absolute_path = os.path.join(project_root, path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    torch.save(model.state_dict(), absolute_path)
    print(f"Model saved to {path}")


def load_model(model, path="models/saved_model/card_classifier.pth", device='cpu'):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    absolute_path = os.path.join(project_root, path)
    if os.path.exists(absolute_path):
        model.load_state_dict(torch.load(absolute_path, map_location=device))
        print(f"Model loaded from {absolute_path}")
    else:
        print("No saved model found. Starting from scratch.")


def train_model(model, train_loader, val_loader, num_epochs, device, save_path="models/saved_model/card_classifier.pth"):
    """
    Main function to handle the training loop.
    """
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Track losses
    train_losses, val_losses = [], []

    # Move model to device
    model.to(device)

    # Load existing model if available
    load_model(model, save_path, device)

    for epoch in range(num_epochs):
        # Training phase
        train_loss = train(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validation phase
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

        # Save the model after each epoch
        save_model(model, save_path)

    return train_losses, val_losses

