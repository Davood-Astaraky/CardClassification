from torchvision import transforms

def get_default_transform():
    
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])