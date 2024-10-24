import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.load_data import download_dataset, load_data
from src.data.preprocess import get_default_transform
from src.data.utils import print_versions

def main():
    # Print version information
    print_versions()
    
    # Download the dataset
    data_path = download_dataset()
    

    # Set the directory where the dataset is located
    data_dir = f'{data_path}/2/train'

    # Set up the transformation
    transform = get_default_transform()

    # Load the data
    dataloader, dataset = load_data(data_dir, batch_size=32, transform=transform)

    # Example usage of the dataset
    image, label = dataset[6000]
    print("Label:", label)
    print("Classes:", dataset.classes)

    # Display mapping of target index to class name
    target_to_class = {v: k for k, v in dataset.data.class_to_idx.items()}
    print("Target to Class Mapping:", target_to_class)

if __name__ == "__main__":
    main()