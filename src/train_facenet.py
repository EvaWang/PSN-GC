import imp
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from torchvision.models import vgg16_bn, resnet18

import sys
sys.path.append('./nets')
from util import read_data, read_json
from BaselineDataset import BaselineDataset
from melnyk_net import MelnykNet
from HardTripletLoss import HardTripletLoss

import argparse
from argparse import Namespace
import math

torch.use_deterministic_algorithms(True)
pl.seed_everything(42, workers=True)

# input_size: 64x64

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

class BaselineTriplet(pl.LightningModule):
    def __init__(self, hparams):
        super(BaselineTriplet, self).__init__()

        self.save_hyperparameters(hparams)
        self.output_size = self.hparams.output_size

        if self.hparams.model_type=="vgg16": 
            self.encoder = vgg16_bn()
            self.encoder.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.encoder.classifier[-1] = nn.Linear(4096, self.output_size)
            nn.init.kaiming_normal_(self.encoder.features[0].weight, mode='fan_out', nonlinearity='relu')
            if self.encoder.features[0].bias is not None:
                nn.init.constant_(self.encoder.features[0].bias, 0)
            nn.init.normal_(self.encoder.classifier[-1].weight, 0, 0.01)
            nn.init.constant_(self.encoder.classifier[-1].bias, 0)
            
        if self.hparams.model_type=="resnet18": 
            self.encoder = resnet18()
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            nn.init.kaiming_normal_(self.encoder.conv1.weight, mode='fan_out', nonlinearity='relu')
            self.encoder.fc = nn.Sequential(nn.Dropout(self.hparams.dropout_rate), nn.Linear(512, self.output_size))
            
        if self.hparams.model_type=="melnyknet": 
            self.encoder = MelnykNet(include_top=True, vocab_size=self.output_size, input_size=64)

        if self.hparams.pretrained_weight!="":
            pretrained_net = BaselineTriplet.load_from_checkpoint(self.hparams.pretrained_weight)
            self.encoder.load_state_dict(pretrained_net.encoder.state_dict()) 

        self.loss = HardTripletLoss(margin=0.2, hardest=self.hparams.triplet_hard)

    def forward(self, targets):
        logits = self.encoder(targets)
        logits_norm = torch.nn.functional.normalize(logits, p=2, dim=1)

        return logits_norm

    def get_encodings(self, targets):
        encodings = self.encoder(targets)
        encodings_norm = torch.nn.functional.normalize(encodings, p=2, dim=1)
        return encodings_norm

    def _unpack_batch(self, batch):
        return batch['target'], batch['target_label'].int()

    def training_step(self, batch, batch_nb):
        target, target_label = self._unpack_batch(batch)
        logit_anchor = self.forward(target)
        triplet_loss = self.loss(logit_anchor, target_label)

        self.log('train_loss', triplet_loss)
        return {'loss': triplet_loss}

    def validation_step(self, batch, batch_nb):
        target, target_label = self._unpack_batch(batch)
        logit_anchor = self.forward(target)
        triplet_loss = self.loss(logit_anchor, target_label)

        self.log('val_loss', triplet_loss, prog_bar=True)
        return {'val_loss': triplet_loss, 'progress_bar':{'val_loss': triplet_loss}}

    def validation_epoch_end(self, outputs) -> None:
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_val_loss)

    def validation_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_val_loss)
        return {'avg_val_loss': avg_val_loss, 'progress_bar':{'avg_val_loss': avg_val_loss}}

    def configure_optimizers(self):
        optimizer = None
        if self.hparams.optimzer_type == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
            return {
                'optimizer': optimizer,
                'monitor': 'avg_val_loss'
            }
        elif self.hparams.optimzer_type == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=0.1, verbose=True, mode='min', min_lr=1e-7)
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'avg_val_loss'
            }

    def _load_dataset(self, dataset_path: str, data_path: str):
        print('loading data...')
        img_list = read_data(dataset_path)
        print(img_list[0])
        dataset = BaselineDataset(img_list, data_path=data_path)
    
        return dataset

    def train_dataloader(self):
        dataset = self._load_dataset(self.hparams.train_dataset_path, self.hparams.data_folder)
        # dataset.data = dataset.data[:320]
        return DataLoader(dataset, 
                          self.hparams.batch_size, 
                          shuffle=True,
                        #   collate_fn=dataset.collate_fn,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=True, pin_memory=True,)

    def val_dataloader(self):
        dataset = self._load_dataset(self.hparams.valid_dataset_path, self.hparams.data_folder)
        # dataset.data = dataset.data[:32]
        return DataLoader(dataset, 
                          self.hparams.batch_size, 
                        #   collate_fn=dataset.collate_fn,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=True, pin_memory=True,)

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline Triplet"
    )
    parser.add_argument('--max_epochs', default=200, type=int, help='max_epochs')
    parser.add_argument('--pretrained_weight', default="", type=str, help='ckpt path')
    parser.add_argument('--resume_from_checkpoint', default="", type=str, help='ckpt path')
    parser.add_argument('--gpu', default="0", type=str, help='number of gpu(s)')
    parser.add_argument('--train_datapath', default="/home/tingwang/Oracle-20k/train.pkl", type=str, help='')
    parser.add_argument('--valid_datapath', default="/home/tingwang/Oracle-20k/valid.pkl", type=str, help='')
    parser.add_argument('--data_folder', default="/home/tingwang/Oracle-20k/train", type=str, help='')
    parser.add_argument('--label_list', default="/home/tingwang/Oracle-20k/jiaguwen_hanziyuan.json", type=str, help='')
    parser.add_argument('--batch_size', default=256, type=int, help='')
    parser.add_argument('--model_type', default="melnyknet", type=str, help='name')
    parser.add_argument('--optimzer_type', default="SGD", type=str, help='name, SGD;ADAM')
    parser.add_argument('--lr', default=0.1, type=float, help='')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='')
    parser.add_argument('--output_size', default=128, type=int, help='')
    parser.add_argument('--triplet_hard', default=0, type=int, help='true:1, false:0')
    
    args = parser.parse_args()
    return args

def main(args):
    
    hparams = Namespace(**{
        'train_dataset_path': args.train_datapath,
        'valid_dataset_path': args.valid_datapath,
        'data_folder': args.data_folder,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'optimzer_type': args.optimzer_type,
        'weight_decay': args.weight_decay,
        'dropout_rate': args.dropout_rate,
        'model_type': args.model_type,
        'num_workers': 4,
        'output_size': args.output_size,
        'description': 'hard triplet loss',
        'pretrained_weight': args.pretrained_weight,
        'triplet_hard': args.triplet_hard == 1
    })

    print("create trainer")
    
    gpu_list = args.gpu.split(',')
    gpu_list = [int(i) for i in gpu_list]
    trainer = pl.Trainer(gpus=gpu_list, max_epochs=args.max_epochs, gradient_clip_val=1, callbacks=[
        EarlyStopping(monitor='avg_val_loss', patience=10, mode='min', verbose=True), 
        ModelCheckpoint(monitor="avg_val_loss", mode='min', save_top_k=-1, filename="{epoch:02d}-{val_loss:.4f}"),
        LearningRateMonitor(logging_interval='epoch')]) 
    baseline = BaselineTriplet(hparams)
    print(baseline)
    
    if args.resume_from_checkpoint != "":
        trainer.fit(baseline, ckpt_path=args.resume_from_checkpoint)
    else:
        trainer.fit(baseline)

if __name__ == '__main__':
    args = _parse_args()
    main(args)