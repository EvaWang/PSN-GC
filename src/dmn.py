import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
pl.seed_everything(42, workers=True)

import sys
sys.path.append('./nets')
import argparse
from argparse import Namespace
from melnyk_net import MelnykNet
from JiaguwenDataset import JiaguwenDataset
from util import read_data, read_json


# input_size: 64x64
class DeepMatchNetJiaguwen(pl.LightningModule):
    def __init__(self, hparams):
        super(DeepMatchNetJiaguwen, self).__init__()
        
        print(hparams)
        self.save_hyperparameters(hparams)
        self.emb_dim = self.hparams.emb_dim
        self.vocab_size = self.hparams.vocab_size
        
        self.encoder = MelnykNet(include_top=False, vocab_size=self.vocab_size, input_size=64)
        self.encoder_dim = 448
            
        self.template_fc = nn.Linear(self.encoder_dim, self.emb_dim)
        self.target_fc = nn.Sequential(nn.Dropout(self.hparams.dropout_rate), nn.Linear(self.encoder_dim, self.emb_dim))
                        
        self.loss = nn.CrossEntropyLoss()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
        
    def forward(self, templates, target):
        # templates:[train_batch=128, 1, 64, 64]
        template_encode = self.encoder(templates)  # [train_batch=128, embedding_dim=128]
        template_encode = self.template_fc(template_encode) # [train_batch=128, embedding_dim=128]

        target_encode = self.encoder(target) #target_encode: [batch_size, embedding_dim=128]

        target_encode = self.target_fc(target_encode)
        logit = torch.mm(target_encode, template_encode.t()) # logit:[batch_size, compared words=128, 1]
        logit = logit.squeeze(1)
        # print(f"logit:{logit.shape}")
        return logit

    def get_template_encode(self, template):
        # template:[vocab_size, 1, 64, 64]
        template_encode = self.encoder(template) # template_encode:[batch_size*128, embedding_dim=128]
        # print(f"template_encode:{template_encode.shape}")
        # template_encode:[vocab_size, 256]
        template_encode = self.template_fc(template_encode)
        # [vocab_size, embedding_dim=128]
        return template_encode

    def predict(self, template_encode, target):
        # print(f"template_encode:{template_encode.shape}")
        target_encode = self.encoder(target)
        # print(f"target_encode:{target_encode.shape}")
        # target_encode: [batch_size, embedding_dim=256]
        target_encode = self.target_fc(target_encode)
        # target_encode: [batch_size, embedding_dim=128]
        logit = torch.mm(target_encode, template_encode.t())
        # [batch_size, 128]*[vocab_size, embedding_dim=128]T = [batch_size, vocab_size]
        logit = logit.squeeze(1)
        return logit

    def _unpack_batch(self, batch):
        return batch["templates"], batch["target"], batch["template_ans"].long(), batch["target_label"].long()
    
    def training_step(self, batch, batch_nb):
        templates, target, templates_ans, target_label = self._unpack_batch(batch)
        logit = self.forward(templates, target)

        loss = self.loss(logit, templates_ans)
        
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        templates, target, templates_ans, target_label = self._unpack_batch(batch)
        logit = self.forward(templates, target)

        loss = self.loss(logit, templates_ans)
        self.log('val_loss', loss)

        pred = torch.argmax(logit, dim=1)
        correct = torch.sum(pred==templates_ans)
        acc = correct/len(pred)

        self.log('acc', acc, prog_bar=True)
        return {'val_loss': loss, 'correct':correct.float(), 'progress_bar':{'val_loss': loss, 'acc':acc}}

    def validation_epoch_end(self, outputs) -> None:
        acc = torch.stack([x['correct'] for x in outputs]).mean()
        self.log('val_acc', acc)

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['correct'] for x in outputs]).mean()
        self.log('avg_loss', avg_loss)
        return {'val_avg_loss': avg_loss, 'val_acc': avg_acc, 'progress_bar':{'val_avg_loss': avg_loss}}

    def configure_optimizers(self):
        print(f"=========lr:{self.hparams.learning_rate}============")
        # LR scheduler
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=0.1, verbose=True, mode='max', min_lr=1e-07)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_acc'
        }

    def _load_dataset(self, dataset_path: str, data_folder):
        print('loading data...')
        img_list = read_data(dataset_path)
        print(img_list[0])
        dataset = JiaguwenDataset(data=img_list, dictionary_path=self.hparams.label_path, data_path=data_folder, compare_num=self.hparams.train_batch)
        
        return dataset

    def train_dataloader(self):
        print(f"==========train_dataset_path:{self.hparams.train_dataset_path}=============")
        dataset = self._load_dataset(self.hparams.train_dataset_path, self.hparams.data_folder)
        return DataLoader(dataset, 
                          self.hparams.batch_size, 
                          shuffle=True,
                          collate_fn=dataset.collate_fn_with_templates,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=True, pin_memory=True,)

    def val_dataloader(self):
        print(f"==========valid_dataset_path:{self.hparams.valid_dataset_path}=============")
        dataset = self._load_dataset(self.hparams.valid_dataset_path, self.hparams.data_folder)
        return DataLoader(dataset, 
                          self.hparams.batch_size, 
                          collate_fn=dataset.collate_fn_with_templates,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=True, pin_memory=True,)

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Deep Matching Net"
    )
    parser.add_argument('--max_epochs', default=50, type=int, help='max_epochs')
    parser.add_argument('--ckpt_path', default="", type=str, help='ckpt path')
    parser.add_argument('--gpu', default="0,1", type=str, help='number of gpu(s)')
    parser.add_argument('--train_datapath', default="/home/tingwang/Oracle-20k/train.pkl", type=str, help='')
    parser.add_argument('--valid_datapath', default="/home/tingwang/Oracle-20k/valid.pkl", type=str, help='')
    parser.add_argument('--data_folder', default="/home/tingwang/Oracle-20k/train", type=str, help='')
    parser.add_argument('--label_path', default="/home/tingwang/Oracle-20k/jiaguwen_hanziyuan.json", type=str, help='')
    parser.add_argument('--batch_size', default=128, type=int, help='')
    parser.add_argument('--emb_dim', default=128, type=int, help='best practice=128')
    parser.add_argument('--train_batch', default=128, type=int, help='first 128, then full vocab')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    args = parser.parse_args()
    return args

def main(args):
    
    label_dict = read_json(args.label_path)

    hparams = Namespace(**{
        'train_dataset_path': args.train_datapath,
        'valid_dataset_path': args.valid_datapath,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'dropout_rate':0.5,
        'num_workers':4,
        'train_batch': args.train_batch,
        'emb_dim': args.emb_dim,
        'data_folder': args.data_folder,
        'name':'dmn jiaguwen',
        'label_path': args.label_path,
        'ckpt_path': args.ckpt_path,
        'vocab_size': len(label_dict)
    })
    print("create trainer")

    gpu_list = args.gpu.split(',')
    gpu_list = [int(i) for i in gpu_list]
    trainer = pl.Trainer(gpus=gpu_list, max_epochs=args.max_epochs, gradient_clip_val=1, callbacks=[
        EarlyStopping(monitor='val_acc', patience=10, mode='max', verbose=True), 
        ModelCheckpoint(monitor="val_acc", mode='max', save_top_k=-1, filename="{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}"),
        LearningRateMonitor(logging_interval='epoch')]) 
    deepMatchNetJiaguwen = DeepMatchNetJiaguwen(hparams)
    print(deepMatchNetJiaguwen)
    if args.ckpt_path != "":
        trainer.fit(deepMatchNetJiaguwen, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(deepMatchNetJiaguwen)
        
if __name__ == '__main__':
    args = _parse_args()
    main(args)