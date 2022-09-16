# extract features
import glob
import sys
sys.path.append('../nets')
from tqdm import tqdm
import argparse


import torch
from torch.utils.data import DataLoader

from train_facenet import BaselineTriplet as TripletModel
from baseline import Baseline as SoftmaxModel
from BaselineDataset import BaselineDataset
from util import read_data, read_json



def _parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_path', default='', type=str, help='saved weights')
    parser.add_argument('--data_folder', default="/home/tingwang/Oracle-20k/train", type=str, help='img path')
    parser.add_argument('--data_label', default="/home/tingwang/Oracle-20k/train.pkl", type=str, help='label path')
    # parser.add_argument('--test_data_label', default="/home/tingwang/Oracle-20k/test_all.pkl", type=str, help='label path')
    # parser.add_argument('--test_data_folder', default="/home/tingwang/Oracle-20k/test", type=str, help='label path')
    parser.add_argument('--label_filepath', default="/home/tingwang/Oracle-20k/jiaguwen_hanziyuan.json", type=str, help='dictionary path')
    parser.add_argument('--batch_size', default=64, type=int, help='8G GPU Memory')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')
    parser.add_argument('--filename', default="oracle-20k-test-triplet.pt", type=str, help='')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()

    img_list = read_data(args.data_label)
    train_dataset = BaselineDataset(img_list, data_path=args.data_folder)
    print(f"total file: {len(img_list)}")

    dataloader = DataLoader(train_dataset, args.batch_size, shuffle=False)

    pretrained_model = TripletModel.load_from_checkpoint(checkpoint_path=args.model_path)
    pretrained_model.to(args.device)
    pretrained_model.freeze()

    all_encodings = []

    for batch in tqdm(dataloader):
        encodings = pretrained_model.get_encodings(batch['target'].to(args.device))
        all_encodings.append(encodings)
            
    all_encodings = torch.cat(all_encodings, dim=0)
    print(all_encodings.shape)
    torch.save(all_encodings, args.filename)