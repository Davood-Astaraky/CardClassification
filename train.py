import torch
from src.data.load_data import download_dataset, load_data
from src.data.preprocess import get_default_transform
from src.models.simple_card_classifier import SimpleCardClassifer
from src.models.train import train_model
import matplotlib.pyplot as plt

def start_training():

    # Device setup with MPS support for Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    # Download dataset if not already available
    data_path = download_dataset()

    # Define train, validation, and test directories
    train_folder = f'{data_path}/train'
    valid_folder = f'{data_path}/valid'

    # Set up the transformation
    transform = get_default_transform()


    # Load datasets
    train_dataset = load_data(train_folder, transform=transform)[1]
    val_dataset   = load_data(valid_folder, transform=transform)[1]


     # Define DataLoader objects
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)


    # Initialize model
    model = SimpleCardClassifer(num_classes=53)
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)


    # Train the model using train_model from src/models/train.py
    num_epochs = 6
    save_path = "models/saved_model/card_classifier.pth"
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs, device, save_path=save_path)
 
    # Inform if training was skipped
    if train_losses is None and val_losses is None:
        print("Training skipped. Existing model is loaded and ready for use.")
    else:
        print("Training completed and model saved.")

        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.legend()
        plt.title("Loss over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()


if __name__ == "__main__":
    start_training()