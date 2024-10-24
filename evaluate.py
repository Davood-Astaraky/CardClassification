import sys
import os

# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.models.simple_card_classifier import SimpleCardClassifer
from src.models.train import load_model
from src.models.evaluate import preprocess_image, predict, visualize_predictions
from src.data.load_data import load_data
import torch
import torchvision.transforms as transforms

def main():
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Initialize and load the model
    model = SimpleCardClassifer(num_classes=53)
    save_path = "models/saved_model/card_classifier.pth"
    load_model(model, save_path, device)
    model.to(device)

    # Example image for testing
    test_image = 'data/raw/2/test/five of diamonds/2.jpg'
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    # Preprocess the image and get predictions
    original_image, image_tensor = preprocess_image(test_image, transform)
    probabilities = predict(model, image_tensor, device)

    # Assuming dataset.classes gives the class names
    train_folder = 'data/raw/2/train'
    dataset = load_data(train_folder, transform=transform)[1]
    class_names = dataset.classes

    # Visualize predictions
    visualize_predictions(original_image, probabilities, class_names)

if __name__ == "__main__":
    main()
