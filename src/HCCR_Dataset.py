import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import sys
sys.path.append('./src')
import os
import random
import numpy as np
from PIL import Image

from util import read_json, read_data


class HCCR_Dataset(Dataset):
    def __init__(self, data, dictionary_path, data_path = "", template_path= "", compare_num=128, img_size=64, transform=None, template_transform=None, with_label=True):
        label_dict = read_json(dictionary_path)
        self.with_label = with_label
        if self.with_label:
            self.data = [{'filename': img["filename"], 'target_label': label_dict[img["label_hex"]], 'target_text': img["target_text"]} for img in data]
        else:
            self.data = [{'filename': img["filename"], 'target_text': img["target_text"]} for img in data]
        self.data_path = data_path
        self.transform = transforms.Compose([transforms.ToTensor()]) if transform is None else transform
        self.template_transform = transforms.Compose([transforms.ToTensor()]) if template_transform is None else template_transform
        self.norm_templates = None
        self.template_labels = None
        self.compare_num = compare_num
        self.img_size = img_size
        self.vocab_size = len(label_dict.items())
        if template_path != "":
            self.norm_templates, self.template_labels = self.template_normalization(label_dict, template_path)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target = Image.open(os.path.join(self.data_path, self.data[index]['filename']))
            
        item = {
            'id': self.data[index]['filename'],
            'target': self.transform(target),
            'target_label': self.data[index]['target_label'] if self.with_label else 0,
            'target_text': self.data[index]['target_text'],
        }
        return item
    
    def template_normalization(self, label_list, template_path):
        templates_arr, templates_label = [], []
        for key, val in label_list.items():
            templates_label.append(val)
            template_image = Image.open(os.path.join(template_path, f"{key}_{self.img_size}x{self.img_size}.jpg"))
            templates_arr.append(self.template_transform(template_image))
        return templates_arr, templates_label

    def get_compare_template(self, batch_of_data, compare_num):
    
        if compare_num < self.vocab_size: 
            data_indice = batch_of_data['target_label']
            data_indice = list(set(data_indice.tolist())) # remove duplicates

            sample_indice = np.arange(self.vocab_size)
            sample_indice = np.delete(sample_indice, data_indice)

            sample_indice = random.sample(sample_indice.tolist(), compare_num-len(data_indice))
            sample_indice = sample_indice+data_indice

            sampled_templates = [self.norm_templates[i] for i in sample_indice] 
            sampled_data = list(zip(sampled_templates, sample_indice))
            random.shuffle(sampled_data)

            sampled_templates, sampled_label = list(zip(*sampled_data))
            sampled_templates, sampled_label = list(sampled_templates), list(sampled_label)
        else:
            sampled_templates = self.norm_templates
            sampled_label = self.template_labels
            
        # create match indice
        templates_ans_list = []
        for target_label in batch_of_data['target_label']:
            x = sampled_label.index(target_label)
            templates_ans_list.append(x)

        return torch.stack(sampled_templates, dim=0), torch.Tensor(list(sampled_label)), torch.Tensor(templates_ans_list)

    def collate_fn(self, samples):
        batch = {}
        # key_1 = ['id', 'target_text', 'candidates']
        key_1 = ['id', 'target_text']
        key2tensor = []
        if self.with_label: 
            key2tensor = ['target_label']
            
        key_tensor_cat = ['target']

        for key in key_1:
            batch[key] = [sample[key] for sample in samples]

        for key in key2tensor:
            batch[key] = [sample[key] for sample in samples]
            batch[key] = torch.tensor(batch[key])

        for key in key_tensor_cat:
            batch[key] = [sample[key].unsqueeze(0) for sample in samples]
            batch[key] = torch.cat(batch[key], dim=0)
        return batch
    
    def collate_fn_with_templates(self, samples):
        batch = self.collate_fn(samples)
        batch["templates"], batch["template_label"], batch["template_ans"] = self.get_compare_template(batch, self.compare_num)
        return batch
    
if __name__ == '__main__':
    label_path = "/home/tingwang/ChineseCharacterRecognize/src/casia_competition_label.json"
    valid_datapath = "/home/tingwang/casia_data/train_test_split/valid_96.pkl"
    data_folder = "/home/tingwang/casia_data/data_bg0_96"
    template_path = "/home/tingwang/casia_data/template_bg0"
    img_list = read_data(valid_datapath)
    
    batch_num = 4

    # dataset = HCCR_Dataset(data=img_list, dictionary_path=label_path, data_path=data_folder, template_path=template_path)
    # dataloader = DataLoader(dataset, batch_num, collate_fn=dataset.collate_fn_with_templates, shuffle=False)
    # for batch in dataloader:
    #     print(batch["target_text"])
    #     print(batch["templates"].size())
    #     print(batch["template_ans"])
    #     break
    
    dataset = HCCR_Dataset(data=img_list, dictionary_path=label_path, data_path=data_folder, img_size=96)
    dataloader = DataLoader(dataset, batch_num, collate_fn=dataset.collate_fn, shuffle=False)
    for batch in dataloader:
        print(batch["target_text"])
        print(batch["target_label"])
        break