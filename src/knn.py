# extract features
import glob
import sys
sys.path.append('../nets')
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from train_facenet import Baseline as TripletModel
from baseline import Baseline as SoftmaxModel
from BaselineDataset import BaselineDataset
from util import read_data, read_json


class Params():
    def __init__(self):
        self.model_path = './lightning_logs/version_9/checkpoints/epoch=22-val_loss=0.02-val_acc=0.00.ckpt'
        self.train_data_folder = "/home/tingwang/Oracle-20k/train"
        self.train_data_label = "/home/tingwang/Oracle-20k/train.pkl"
        self.test_data_label = "/home/tingwang/Oracle-20k/test_all.pkl"
        self.test_data_folder = "/home/tingwang/Oracle-20k/test"
        self.label_filepath = "/home/tingwang/Oracle-20k/jiaguwen_hanziyuan.json"
        self.device = "cuda"

args = Params()


img_list = read_data(args.test_data_label)
train_dataset = BaselineDataset(img_list, data_path=args.test_data_folder)
print(f"total file: {len(img_list)}")

dataloader = DataLoader(train_dataset, 64, shuffle=False)

pretrained_model = TripletModel.load_from_checkpoint(checkpoint_path=args.model_path)
pretrained_model.to(args.device)
pretrained_model.freeze()

all_encodings = []

for batch in tqdm(dataloader):
    encodings = pretrained_model.get_encodings(batch['target'].to(args.device))
    all_encodings.append(encodings)
        
all_encodings = torch.cat(all_encodings, dim=0)
print(all_encodings.shape)
torch.save(all_encodings, 'oracle-20k-test-triplet.pt')