import argparse
from this import s
import numpy
import logging
import sys
sys.path.append('./src')
from PIL import Image
import glob
import os

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy

from util import *
from HCCR_Dataset import HCCR_Dataset
from duo_global import DeepMatchNet
from tqdm import tqdm


def label2text(label_list, idx):
    text = label_list[idx]
    text = chr(int(text,16))
    return text

def prediction(pretrained_model, dataset, batch_size, label_dict, device='cuda'):
    print('start prediction...')
    device = torch.device(device)

    vocab_size = len(label_dict.items())
    label_list = ['X']*vocab_size
    for key, val in label_dict.items(): label_list[val]=key 
    
    pretrained_model.to(device)
    pretrained_model.freeze()
    
    dataloader = DataLoader(dataset, batch_size, collate_fn=dataset.collate_fn, shuffle=False)
    
    print('create template encode')
    templates, templates_text = dataset.norm_templates, dataset.template_labels
    templates = torch.stack(templates, dim=0).to(device) #torch.Size([3755, 1, 64, 64])
    templates_encode = pretrained_model.get_template_encode(templates)
    print('template encode created.')

    stat = []
    counter, correct = 0, 0
    for batch in tqdm(dataloader):
        logging.debug(f"start batch:{counter}")
        target = batch['target'].to(device)
        target_label = batch['target_label']

        logit = pretrained_model.predict(templates_encode, target)
        logit = F.softmax(logit, dim=1)
        pred_prob, predict_ans = torch.topk(logit, dim=1, k=5)
        batch_size = target.size(0)
        for batch_num in range(batch_size):
            pred_idx = predict_ans[batch_num]
            gt_idx = target_label[batch_num]
            gt_text = label2text(label_list, gt_idx)
            pred_list = [batch['id'][batch_num], gt_text]
            
            for i in range(5):
                label = templates_text[pred_idx[i]]
                pred_text = label2text(label_list, label)
                pred_list.append(pred_text)
            pred_list += pred_prob[batch_num].tolist()
            
            counter = counter+1
            pred_correct = pred_list[2] == gt_text
            if pred_correct: correct=correct+1
            logging.info(f"filename:{batch['id'][batch_num]}, pred:{pred_list[2]}, ground truth:{gt_text}, correct:{pred_correct}")
            stat.append(pred_list)
            
        logging.debug(f"end batch:{counter}")

    logging.info(f"acc: {correct/counter}")
    total_acc = f"acc: {correct/counter}"
    return stat, total_acc


def _parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_path', default='', type=str, help='saved weights')
    parser.add_argument('--data_folder', default="/home/tingwang/casia_data/competition/", type=str, help='img path')
    parser.add_argument('--data_label', default="/home/tingwang/casia_data/competition/test.pkl", type=str, help='label path')
    parser.add_argument('--template_path', default="/home/tingwang/casia_data/template_bg0_skeletonize/", type=str, help='label path')
    parser.add_argument('--label_filepath', default="/home/tingwang/ChineseCharacterRecognize/src/casia_competition_label.json", type=str, help='dictionary path')
    parser.add_argument('--output_filepath', default="", type=str, help='output file path')
    parser.add_argument('--batch_size', default=128, type=int, help='8G GPU Memory')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')
    parser.add_argument('--model_type', default="duo", type=str, help='duo;dmn')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()

    logging.basicConfig(filename=args.output_filepath, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=20)
    
    pretrained_model = None
    if args.model_type == 'duo':
        pretrained_model = DeepMatchNet.load_from_checkpoint(checkpoint_path=args.model_path)
    else: 
        raise ValueError(f"No model type: {args.model_type}")

    print(f"loading data... path: {args.data_label}")
    img_list = read_data(args.data_label)
    print(img_list[0])
    print('data loaded...')

    label_dict = read_json(args.label_filepath)
    
    dataset = HCCR_Dataset(data=img_list, dictionary_path=args.label_filepath, data_path=args.data_folder, compare_num=len(label_dict) , template_path=args.template_path,)
    prediction_stat, acc = prediction(pretrained_model, dataset, args.batch_size, label_dict, device=args.device)

    csvfile = numpy.asarray(prediction_stat)
    numpy.savetxt(args.output_filepath.replace(".log", ".csv"), csvfile, delimiter=",", fmt="%s")
    print(acc)

