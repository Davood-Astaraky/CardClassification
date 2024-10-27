import sys
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from glob import glob

# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import necessary modules
from src.models.simple_card_classifier import SimpleCardClassifer
from src.models.train import load_model
from src.models.evaluate import preprocess_image, predict, visualize_predictions
from src.data.load_data import load_data

def main():
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize and load the model
    model = SimpleCardClassifer(num_classes=53)
    save_path = "models/saved_model/card_classifier.pth"
    load_model(model, save_path, device)
    model.to(device)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    # Load class names from training data
    train_folder = 'data/raw/2/train'
    dataset = load_data(train_folder, transform=transform)[1]
    class_names = dataset.classes

    # Select random examples from test images
    test_images = glob('data/raw/2/test/*/*')
    test_examples = np.random.choice(test_images, 10, replace=False)  # 10 random examples

    # Loop through selected examples and visualize predictions
    for example in test_examples:
        original_image, image_tensor = preprocess_image(example, transform)
        probabilities = predict(model, image_tensor, device)
        visualize_predictions(original_image, probabilities, class_names)

if __name__ == "__main__":
    main()
