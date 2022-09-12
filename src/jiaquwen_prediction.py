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
from JiaguwenDataset import JiaguwenDataset
from duo_global_jiaguwen import DeepMatchNetJiaguwen
from dmn import DeepMatchNetJiaguwen as dmn
from tqdm import tqdm


def prediction(pretrained_model, dataset, batch_size, template_folder, template_label, device='cuda'):
    print('start prediction...')
    device = torch.device(device)
    
    pretrained_model.to(device)
    pretrained_model.freeze()
    
    dataloader = DataLoader(dataset, batch_size, collate_fn=dataset.collate_fn, shuffle=False)
    
    # print('create template encode')
    template_transform = transforms.Compose([transforms.ToTensor()])
    template_list = read_data(template_label)
    print(template_list[0])
    templates, templates_text = [], []
    for temp in template_list:
        # img = Image.open(os.path.join(template_folder, temp["ans"],os.path.basename(temp["filename"])))
        img = Image.open(os.path.join(template_folder, temp["target_text"],temp["filename"]))
        templates.append(template_transform(img))
        templates_text.append(temp["target_text"])
    templates = torch.stack(templates, dim=0).to(device) #torch.Size([3755, 1, 64, 64])
    templates_encode = pretrained_model.get_template_encode(templates)
    print('template encode created.')

    stat = []
    counter, correct = 0, 0
    for batch in tqdm(dataloader):
        logging.debug(f"start batch:{counter}")
        target = batch['target'].to(device)

        logit = pretrained_model.predict(templates_encode, target)
        logit = F.softmax(logit, dim=1)
        pred_prob, predict_ans = torch.topk(logit, dim=1, k=5)
        batch_size = target.size(0)
        for batch_num in range(batch_size):
            pred_idx = predict_ans[batch_num]
            gt_text = batch['target_text'][batch_num]
            pred_list = [batch['id'][batch_num], gt_text]
            
            for i in range(5):
                label = templates_text[pred_idx[i]]
                pred_list.append(label)
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
    parser.add_argument('--data_folder', default="/home/tingwang/Oracle-20k/test", type=str, help='img path')
    parser.add_argument('--data_label', default="/home/tingwang/Oracle-20k/test_all.pkl", type=str, help='label path')
    parser.add_argument('--template_label', default="/home/tingwang/Oracle-20k/template.pkl", type=str, help='label path')
    parser.add_argument('--template_folder', default="/home/tingwang/Oracle-20k/train", type=str, help='label path')
    parser.add_argument('--label_filepath', default="/home/tingwang/Oracle-20k/jiaguwen_hanziyuan.json", type=str, help='dictionary path')
    parser.add_argument('--output_filepath', default="", type=str, help='output file path')
    parser.add_argument('--batch_size', default=128, type=int, help='8G GPU Memory')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')
    parser.add_argument('--model_type', default="duo", type=str, help='duo;dmn')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()

    logging.basicConfig(filename=args.output_filepath, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=20)

    print(f"loading data... path: {args.data_label}")
    img_list = read_data(args.data_label)
    print(img_list[0])
    # img_list = [{'target_text': img['target_text'].replace('id_',''), 'filename': os.path.basename(img['filename']), 'target_label': img['target_label']} for img in img_list]
    img_list = [{'target_text': img['target_text'].replace('id_',''), 'filename': os.path.basename(img['filename']), 'target_label': 0} for img in img_list]
    print(img_list[0])
    print('data loaded...')

    pretrained_model = None
    if args.model_type == 'duo':
        pretrained_model = DeepMatchNetJiaguwen.load_from_checkpoint(checkpoint_path=args.model_path)
    elif args.model_type == 'dmn':
        pretrained_model = dmn.load_from_checkpoint(checkpoint_path=args.model_path)
    else: 
        raise ValueError(f"No model type: {args.model_type}")
        
    dataset = JiaguwenDataset(data=img_list, dictionary_path=args.label_filepath, data_path=args.data_folder)
    prediction_stat, acc = prediction(pretrained_model, dataset, args.batch_size, args.template_folder, args.template_label, device=args.device)

    csvfile = numpy.asarray(prediction_stat)
    numpy.savetxt(args.output_filepath.replace(".log", ".csv"), csvfile, delimiter=",", fmt="%s")
    print(acc)

