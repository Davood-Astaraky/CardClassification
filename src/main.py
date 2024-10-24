import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.load_data import download_dataset, load_data
from src.data.preprocess import get_default_transform
from src.data.utils import print_versions
from src.models.simple_card_classifier import SimpleCardClassifer
from src.models.train import train_model
import torch

def main():
    # Print version information
    print_versions()
    
    # Download the dataset
    data_path = download_dataset()
    

    # Set the directory where the dataset is located
    train_folder = f'{data_path}/2/train'
    valid_folder = f'{data_path}/2/valid'
    test_folder  = f'{data_path}/2/test'

    # Set up the transformation
    transform = get_default_transform()

    # Load the data
    train_dataset = load_data(train_folder, transform=transform)[1]
    val_dataset   = load_data(valid_folder, transform=transform)[1]
    test_dataset   = load_data(valid_folder, transform=transform)[1]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


    # Initialize the model
    model = SimpleCardClassifer(num_classes=53)

    # Set the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Define the path to save the model
    save_path = "models/saved_model/card_classifier.pth"

    # Train the model
    num_epochs = 5
    train_model(model, train_loader, val_loader, num_epochs, device, save_path)


if __name__ == "__main__":
    main()