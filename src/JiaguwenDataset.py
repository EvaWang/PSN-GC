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
import glob

from util import read_json, read_data


class JiaguwenDataset(Dataset):
    def __init__(self, data, dictionary_path, data_path = "", template_path= "", compare_num=-1, img_size=64, transform=None, template_transform=None, with_label=True):
        label_dict = read_json(dictionary_path)
        print(dictionary_path)
        self.with_label = with_label
        self.data = data
        self.data_path = data_path
        self.transform = transforms.Compose([transforms.ToTensor()]) if transform is None else transform
        self.template_transform = transforms.Compose([transforms.ToTensor()]) if template_transform is None else template_transform
        self.norm_templates = None
        self.template_labels = None
        self.img_size = img_size
        self.vocab_size = len(label_dict.items())
        self.compare_num = compare_num if compare_num>0 else self.vocab_size
        self.lookup = {v: k for k, v in label_dict.items()}
        if template_path != "":
            self.norm_templates, self.template_labels = self.template_normalization(label_dict, template_path)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target = Image.open(os.path.join(self.data_path, str(self.data[index]['target_text']), self.data[index]['filename']))
            
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
            template_image = Image.open(os.path.join(template_path, f"{key}.jpg"))
            templates_arr.append(self.template_transform(template_image))
        return templates_arr, templates_label

    def get_compare_template(self, batch_of_data, compare_num):
        
        sampled_templates, sampled_label = [], []
        
        if compare_num < self.vocab_size: 
            data_indice = batch_of_data['target_label']
            data_indice = list(set(data_indice.tolist())) # remove duplicates  
            
            # 取沒有選到的類別
            sample_indice = np.arange(self.vocab_size) 
            sample_indice = np.delete(sample_indice, data_indice)
            
            # 補成每個batch需要的數量
            sample_indice = random.sample(sample_indice.tolist(), compare_num-len(data_indice))
            sample_indice = sample_indice+data_indice
            
            for sample in sample_indice:
                sub_folder = self.lookup[sample].replace('id_','')
                file_list = glob.glob(os.path.join(self.data_path, sub_folder)+"/*")
                random_idx = random.randint(0, len(file_list)-1)
                template_image = Image.open(file_list[random_idx])
                sampled_templates.append(self.template_transform(template_image))
                
            sampled_data = list(zip(sampled_templates, sample_indice))
            random.shuffle(sampled_data)
            sampled_templates, sampled_label = list(zip(*sampled_data))
            sampled_templates, sampled_label = list(sampled_templates), list(sampled_label)
            
        else: 
            # print("all templates")    
            for cls in self.lookup:
                sub_folder = self.lookup[cls].replace('id_','')
                file_list = glob.glob(os.path.join(self.data_path, sub_folder)+"/*")
                random_idx = random.randint(0, len(file_list)-1)
                template_image = Image.open(file_list[random_idx])
                sampled_templates.append(self.template_transform(template_image))
                sampled_label.append(cls)
                
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

    data_label = "/home/tingwang/ChineseCharacterRecognize/preprocess/jiaguwen_selected_train.pkl"
    label_filepath = "/home/tingwang/ChineseCharacterRecognize/preprocess/jiaguwen_selected.json"
    data_folder = "/home/tingwang/jiaguwen/img_jpg_selected"
    img_list = read_data(data_label)
    print(img_list[0])
    label_dict = read_json(label_filepath)
    # dataset = JiaguwenDataset(data=img_list, dictionary_path=label_filepath, data_path=data_folder, compare_num=8)
    # dataloader = DataLoader(dataset, 4, collate_fn=dataset.collate_fn_with_templates, shuffle=False)
    # for batch in dataloader:
    #     print(batch['template_ans'])
    #     print(batch['template_label'])
    #     print(batch['target_label'])
    #     break
