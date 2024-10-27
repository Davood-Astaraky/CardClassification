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

def download_dataset(dataset_name="gpiosenka/cards-image-datasetclassification", base_dir="data/raw/2"):
    # Expected subdirectories
    expected_dirs = ['train', 'valid', 'test']
    all_dirs_exist = all(os.path.exists(os.path.join(base_dir, sub_dir)) and os.listdir(os.path.join(base_dir, sub_dir)) for sub_dir in expected_dirs)
    
    # If data is already downloaded in the expected structure, skip download
    if all_dirs_exist:
        print(f"Dataset already exists in {base_dir}. Skipping download.")
        return base_dir

    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)

    # Download dataset using kagglehub
    try:
        path = kagglehub.dataset_download(dataset_name)
        print("Dataset downloaded to:", path)
        
        # Move contents of the downloaded dataset into `base_dir`
        if os.path.exists(path):
            for item in os.listdir(path):
                source = os.path.join(path, item)
                destination = os.path.join(base_dir, item)
                if os.path.isdir(source):
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                else:
                    shutil.move(source, destination)
            print("Dataset contents moved to:", base_dir)
            shutil.rmtree(path)
            return base_dir
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

