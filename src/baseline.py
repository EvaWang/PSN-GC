import imp
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torchvision.transforms as transforms

from torchvision.models import vgg16_bn, resnet18

import sys
sys.path.append('./nets')
from util import read_data, read_json
from BaselineDataset import BaselineDataset
# from resnet12 import resnet12
from melnyk_net import MelnykNet

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


class Baseline(pl.LightningModule):
    def __init__(self, hparams):
        super(Baseline, self).__init__()

        self.save_hyperparameters(hparams)
        self.vocab_size = self.hparams.vocab_size

        if self.hparams.model_type=="vgg16": 
            self.encoder = vgg16_bn()
            self.encoder.classifier[-1] = nn.Linear(4096, self.vocab_size)
            self.encoder.apply(weights_init)
            
        if self.hparams.model_type=="resnet18": 
            self.encoder = resnet18()
            self.encoder.fc = nn.Sequential(nn.Dropout(self.hparams.dropout_rate), nn.Linear(512, self.vocab_size))
            self.encoder.apply(weights_init)
            
        if self.hparams.model_type=="melnyknet": 
            self.encoder = MelnykNet(include_top=True, vocab_size=self.vocab_size, input_size=64)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, target):
        logit_global = self.encoder(target) #target_encode: [batch_size, embedding_dim=128]

        return logit_global

    def _unpack_batch(self, batch):
        return batch['target'], batch['target_label'].long()

    def training_step(self, batch, batch_nb):
        target, target_label = self._unpack_batch(batch)
        logit_global = self.forward(target)
        loss_global = self.loss(logit_global, target_label)

        self.log('train_loss', loss_global)
        return {'loss': loss_global}

    def validation_step(self, batch, batch_nb):
        target, target_label = self._unpack_batch(batch)
        logit_global = self.forward(target)
        loss_global = self.loss(logit_global, target_label)
        self.log('val_loss', loss_global)

        pred = torch.argmax(logit_global, dim=1)
        correct = torch.sum(pred==target_label)
        acc = correct/len(pred)
        self.log('acc', acc, prog_bar=True)
        return {'val_loss': loss_global, 'correct':correct.float(), 'progress_bar':{'val_loss': loss_global, 'acc':acc}}

    def validation_epoch_end(self, outputs) -> None:
        acc = torch.stack([x['correct'] for x in outputs]).mean()
        self.log('val_acc', acc)

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_loss', avg_loss)
        return {'val_loss': avg_loss, 'progress_bar':{'val_loss': avg_loss}}

    def configure_optimizers(self):
        optimizer = None
        if self.hparams.optimzer_type == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimzer_type == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=0.9)
            
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=0.1, verbose=True, mode='max', min_lr=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_acc'
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
        description="Baseline"
    )
    parser.add_argument('--max_epochs', default=100, type=int, help='max_epochs')
    parser.add_argument('--resume_from_checkpoint', default="", type=str, help='ckpt path')
    parser.add_argument('--gpu', default="0", type=str, help='number of gpu(s)')
    parser.add_argument('--train_datapath', default="/home/tingwang/Oracle-20k/train.pkl", type=str, help='')
    parser.add_argument('--valid_datapath', default="/home/tingwang/Oracle-20k/valid.pkl", type=str, help='')
    parser.add_argument('--data_folder', default="/home/tingwang/Oracle-20k/resized", type=str, help='')
    parser.add_argument('--label_list', default="/home/tingwang/ChineseCharacterRecognize/preprocess/jiaguwen_hanziyuan.json", type=str, help='')
    parser.add_argument('--batch_size', default=256, type=int, help='')
    parser.add_argument('--model_type', default="melnyknet", type=str, help='name')
    parser.add_argument('--optimzer_type', default="SGD", type=str, help='name, SGD;ADAM')
    parser.add_argument('--lr', default=0.1, type=int, help='')
    parser.add_argument('--weight_decay', default=1e-3, type=int, help='')
    parser.add_argument('--dropout_rate', default=0.5, type=int, help='')
    args = parser.parse_args()
    return args

def main(args):
    
    label_list = read_json(args.label_list)

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
        'vocab_size': len(label_list)
    })

    print("create trainer")
    
    gpu_list = args.gpu.split(',')
    gpu_list = [int(i) for i in gpu_list]
    trainer = pl.Trainer(gpus=gpu_list, max_epochs=args.max_epochs, gradient_clip_val=1, callbacks=[
        EarlyStopping(monitor='val_acc', patience=10, mode='max', verbose=True), 
        ModelCheckpoint(monitor="val_acc", mode='max', save_top_k=-1, filename="{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}"),
        LearningRateMonitor(logging_interval='epoch')]) 
    baseline = Baseline(hparams)
    print(baseline)
    
    if args.resume_from_checkpoint != "":
        trainer.fit(baseline, ckpt_path=args.resume_from_checkpoint)
    else:
        trainer.fit(baseline)

if __name__ == '__main__':
    args = _parse_args()
    main(args)