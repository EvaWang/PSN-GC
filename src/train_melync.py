import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
pl.seed_everything(42, workers=True)

import sys
sys.path.append('./nets')
from util import read_data

from HCCR_Dataset import HCCR_Dataset
from melync_net import Melync

import argparse
from argparse import Namespace



# input_size: 64x64

class MelyncNet(pl.LightningModule):
    def __init__(self, hparams):
        super(MelyncNet, self).__init__()

        self.save_hyperparameters(hparams)
        self.encoder_dim = 256
        self.vocab_size = self.hparams.vocab_size
        self.img_size = 96
        if 'img_size' in self.hparams:
            self.img_size = self.hparams.img_size

        self.encoder = Melync(include_top=True, vocab_size=self.vocab_size, input_size=self.img_size)
        
        self.loss = nn.CrossEntropyLoss()

    def forward(self, target):
        logit_global = self.encoder(target)
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
        return {'val_loss': loss_global, 'correct':correct.float(), 'progress_bar':{'val_loss': loss_global}}

    def validation_epoch_end(self, outputs):
        avg_acc = torch.stack([x['correct'] for x in outputs]).mean()
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_loss', avg_loss, prog_bar=True)
        self.log('val_acc', avg_acc,  prog_bar=True)

    def configure_optimizers(self):
        # LR scheduler
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=0.1, verbose=True, mode='max', min_lr=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_acc',
        }

    def _load_dataset(self, dataset_path: str, data_folder):
        print('loading data...')
        img_list = read_data(dataset_path)
        print(img_list[0])
        # total = len(img_list)
        # print(total)
        # img_list = img_list[: int(total/1000)]
        dataset = HCCR_Dataset(data=img_list, dictionary_path=self.hparams.dictionary_path, data_path = data_folder, img_size=96)
        return dataset

    def train_dataloader(self):
        dataset = self._load_dataset(self.hparams.train_dataset_path, self.hparams.train_data_folder)
        return DataLoader(dataset, 
                          self.hparams.batch_size, 
                          shuffle=True,
                          collate_fn=dataset.collate_fn,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=True, pin_memory=True,)

    def val_dataloader(self):
        dataset = self._load_dataset(self.hparams.valid_dataset_path, self.hparams.valid_data_folder)
        return DataLoader(dataset, 
                          self.hparams.batch_size, 
                          collate_fn=dataset.collate_fn,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=True, pin_memory=True,)

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Melync"
    )
    parser.add_argument('--max_epochs', default=50, type=int, help='max_epochs')
    parser.add_argument('--resume_from_checkpoint', default="", type=str, help='ckpt path')
    parser.add_argument('--gpu', default="0,1", type=str, help='number of gpu(s)')
    parser.add_argument('--train_datapath', default="/home/tingwang/casia_data/train_test_split/train_exclude1_2.pkl", type=str, help='')
    parser.add_argument('--valid_datapath', default="/home/tingwang/casia_data/train_test_split/valid_exclude1_2.pkl", type=str, help='')
    parser.add_argument('--train_data_folder', default="/home/tingwang/casia_data/data_bg0_96_norm", type=str, help='')
    parser.add_argument('--valid_data_folder', default="/home/tingwang/casia_data/data_bg0_96_norm", type=str, help='')
    parser.add_argument('--batch_size', default=256, type=int, help='')
    parser.add_argument('--ckpt_path', default="", type=str, help='ckpt path')
    parser.add_argument('--dictionary_path', default="/home/tingwang/ChineseCharacterRecognize/src/casia_competition_label.json", type=str, help='ckpt path')
    parser.add_argument('--lr', default=0.1, type=float, help='')
    parser.add_argument('--img_size', default=96, type=int, help='')
    args = parser.parse_args()
    return args

def main(args):

    hparams = Namespace(**{
        'train_dataset_path': args.train_datapath,
        'valid_dataset_path': args.valid_datapath,
        'train_data_folder': args.train_data_folder,
        'valid_data_folder': args.valid_data_folder,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'dropout_rate': 0.5,
        'num_workers': 8,
        'vocab_size': 6375,
        'model_name': "melnyk ahcdb",
        'dictionary_path': args.dictionary_path,
        'img_size': args.img_size
    })

    print("create trainer")

    gpu_list = args.gpu.split(',')
    gpu_list = [int(i) for i in gpu_list]

    trainer = pl.Trainer(gpus=gpu_list, gradient_clip_val=1, max_epochs=args.max_epochs, callbacks=[EarlyStopping(monitor='val_acc', patience=10, mode='max', verbose=True), ModelCheckpoint(monitor="val_acc", mode='max', save_top_k=-1, filename="{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}")]) 

    melyncNet = MelyncNet(hparams)
    print(melyncNet)
    if args.ckpt_path != "":
        trainer.fit(melyncNet, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(melyncNet)

if __name__ == '__main__':
    args = _parse_args()
    main(args)