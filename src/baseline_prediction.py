import argparse
from tqdm import tqdm
import logging
import os

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

torch.use_deterministic_algorithms(False)
pl.seed_everything(42, workers=True)

from util import *
from BaselineDataset import BaselineDataset
from baseline import Baseline

def prediction(model_path, test_data, label_filepath, batch_size=32, device='cuda'):
    print('start prediction...')
    device = torch.device(device)

    label_dict = read_json(label_filepath)
    vocab_size = len(label_dict)
    label_list = ['X']*vocab_size
    for key, val in label_dict.items(): label_list[val]=key 

    # load model
    print('loading model...')
    pretrained_model = Baseline.load_from_checkpoint(checkpoint_path=model_path).to(device)
    print('model loaded...')
    pretrained_model.freeze()

    print('loading data...')
    dataloader = DataLoader(test_data, batch_size, num_workers=8)
    print('data loaded...')

    stat = []
    counter, correct = 0, 0
    for batch in tqdm(dataloader):
        logging.debug(f"start batch:{counter}")
        target = batch['target'].to(device)
        target_label = batch['target_label']
        filename = batch["filename"]

        logit = pretrained_model(target)
        predict_ans = torch.argmax(logit, dim=1)
        batch_size = target.size(0)
        for batch_num in range(batch_size):
            ans_idx = predict_ans[batch_num]
            pred_text = label_list[ans_idx]

            gt_text = label_list[target_label[batch_num]]
            counter = counter+1
            pred_correct = pred_text == gt_text
            if pred_correct: correct=correct+1
            logging.info(f"filename: {filename[batch_num]}, pred:{pred_text}, ground truth:{gt_text}, correct:{pred_correct}")
        
        logging.debug(f"end batch:{counter}")

    logging.info(f"acc: {correct/counter}")
    total_acc = f"acc: {correct/counter}"
    return stat, total_acc

def _parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_path', default='', type=str, help='saved weights')
    parser.add_argument('--data_folder', default="/home/tingwang/Oracle-20k/test", type=str, help='test set file path')
    parser.add_argument('--test_datapath', default="/home/tingwang/Oracle-20k/test_all.pkl", type=str, help='test set file path')
    parser.add_argument('--label_filepath', default="/home/tingwang/ChineseCharacterRecognize/preprocess/jiaguwen_hanziyuan.json", type=str, help='output file path')
    parser.add_argument('--output_filepath', default="", type=str, help='output file path')
    parser.add_argument('--batch_size', default=256, type=int, help='8G GPU Memory')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    
    logging.basicConfig(filename=args.output_filepath, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=20)

    img_list = read_data(args.test_datapath)
    dataset = BaselineDataset(img_list, data_path=args.data_folder)

    prediction_stat, acc = prediction(model_path=args.model_path, test_data=dataset, label_filepath=args.label_filepath, batch_size=args.batch_size, device=args.device)
    print(acc)

