from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image
import os
import glob
import random

class BaselineDataset(Dataset):
    def __init__(self, data, data_path, lazy=True, sample_transform=None):
        self.data = data
        self.sample_transform = transforms.Compose([transforms.ToTensor()]) if sample_transform is None else sample_transform
        self.lazy = lazy
        self.data_path = data_path
        self.class_list = os.listdir(self.data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target = None
        positive = None
        negative = None
        if self.lazy:
            class_name = str(self.data[index]['target_text'])
            target = Image.open(os.path.join(self.data_path, class_name, self.data[index]['filename']))
            positive_list = glob.glob(os.path.join(self.data_path, class_name, "*.jpg"))
            random.shuffle(positive_list)
            positive = Image.open(positive_list[0])

            random.shuffle(self.class_list)
            negative_class = self.class_list[0]
            if class_name == negative_class: negative_class = self.class_list[1]
            negative_list = glob.glob(os.path.join(self.data_path, negative_class, "*.jpg"))
            random.shuffle(negative_list)
            negative = Image.open(negative_list[0])
        else:
            target = self.data[index]["file"]

        item = {
            'filename': self.data[index]["filename"],
            'target': self.sample_transform(target),
            'positive': self.sample_transform(positive),
            'negative': self.sample_transform(negative),
            'target_label': self.data[index]["target_label"],
            'target_text': self.data[index]["target_text"]
        }
        return item