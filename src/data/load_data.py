import kagglehub
import os
import shutil
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

class PlayingCardDataset(Dataset):
    
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes

def download_dataset(dataset_name="gpiosenka/cards-image-datasetclassification", base_dir="../data/raw"):
    
    # Check if the dataset already exists
    if os.path.exists(base_dir) and os.listdir(base_dir):
        print(f"Dataset already exists in {base_dir}. Skipping download.")
        return base_dir
    
    # Ensure the base directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Download dataset using kagglehub (without specifying `path`)
    try:
        path = kagglehub.dataset_download(dataset_name)
        print("Dataset downloaded to:", path)
        
        # Move the downloaded dataset to the desired `base_dir` if necessary
        if os.path.exists(path):
            destination = os.path.join(base_dir, os.path.basename(path))
            if not os.path.exists(destination):
                shutil.move(path, destination)
            print("Dataset moved to:", destination)
            return destination
        else:
            print("Dataset path not found:", path)
            return None
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None



def load_data(data_dir, batch_size=32, transform=None, shuffle=True):
    # Load the dataset using the custom PlayingCardDataset class
    dataset = PlayingCardDataset(data_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset

