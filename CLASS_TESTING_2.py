from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_names = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image


# Example transformations: Resize and Normalize images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize image to 128x128
    transforms.ToTensor(),  # Convert image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Normalize (optional, depending on model)
])

dataset = ImageDataset(image_dir='path_to_downloaded_images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
