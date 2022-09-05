import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image
import os

class BaselineDataset(Dataset):
    def __init__(self, data, data_path, lazy=True, sample_transform=None):
        self.data = data
        self.sample_transform = transforms.Compose([transforms.ToTensor()]) if sample_transform is None else sample_transform
        self.lazy = lazy
        self.data_path = data_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target = None
        if self.lazy:
            target = Image.open(os.path.join(self.data_path, self.data[index]['filename']))
        else:
            target = self.data[index]["file"]

        item = {
            'target': self.sample_transform(target),
            'target_label': self.data[index]["target_label"],
            'target_text': self.data[index]["target_text"]
        }
        return item