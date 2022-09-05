import torch
import torch.nn as nn


class MelnykNet(nn.Module):
    def __init__(self, include_top, vocab_size, input_size=96):
        super(MelnykNet, self).__init__()
        self.include_top = include_top
        self.vocab_size = vocab_size
        
        self.layer1 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3,padding='same', stride=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU())
        self.layer2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3,padding='same', stride=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AvgPool2d((3, 3), stride=(2, 2), padding=1))
        self.layer3 = nn.Sequential(
                nn.Conv2d(64, 96, kernel_size=3, padding='same', stride=1, bias=False),
                nn.BatchNorm2d(96),
                nn.ReLU())
        self.layer4 = nn.Sequential(
                nn.Conv2d(96, 64, kernel_size=3, padding='same', stride=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU())
        self.layer5 = nn.Sequential(
                nn.Conv2d(64, 96, kernel_size=3,padding='same', stride=1, bias=False),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.AvgPool2d((3, 3), stride=(2, 2), padding=1))
        self.layer6 = nn.Sequential(
                nn.Conv2d(96, 128, kernel_size=3,padding='same', stride=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU())
        self.layer7 = nn.Sequential(
                nn.Conv2d(128, 96, kernel_size=3,padding='same', stride=1, bias=False),
                nn.BatchNorm2d(96),
                nn.ReLU())
        self.layer8 = nn.Sequential(
                nn.Conv2d(96, 128, kernel_size=3,padding='same', stride=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AvgPool2d((3, 3), stride=(2, 2), padding=1))
        self.layer9 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3,padding='same', stride=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU())
        self.layer10 = nn.Sequential(
                nn.Conv2d(256, 192, kernel_size=3,padding='same', stride=1, bias=False),
                nn.BatchNorm2d(192),
                nn.ReLU())
        self.layer11 = nn.Sequential(
                nn.Conv2d(192, 256, kernel_size=3,padding='same', stride=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AvgPool2d((3, 3), stride=(2, 2), padding=1))
        self.layer12 = nn.Sequential(
                nn.Conv2d(256, 448, kernel_size=3,padding='same', stride=1, bias=False),
                nn.BatchNorm2d(448),
                nn.ReLU())
        self.layer13 = nn.Sequential(
                nn.Conv2d(448, 256, kernel_size=3,padding='same', stride=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU())
        self.layer14 = nn.Sequential(
                nn.Conv2d(256, 448, kernel_size=3,padding='same', stride=1, bias=False),
                nn.BatchNorm2d(448),
                nn.ReLU())
        
        if input_size==96:
            self.W = torch.nn.Parameter(torch.ones(448, 6, 6))
        elif input_size==64:
            self.W = torch.nn.Parameter(torch.ones(448, 4, 4))
            
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(448, vocab_size, bias=False))

        # L2: optimizer weight_decay 0.01
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                # n = m.weight.size(1)
                m.weight.data.normal_(0, 0.001) 
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        
        x = x*self.W
        x = x.sum(dim=(-2, -1))
        if self.include_top: 
            x = self.fc(x)
        
        # layer1: torch.Size([256, 64, 96, 96])                                                                                                                                        
        # layer2: torch.Size([256, 64, 48, 48])                                                                                                                                        
        # layer3: torch.Size([256, 96, 48, 48])                                                                                                                                        
        # layer4: torch.Size([256, 64, 48, 48])                                                                                                                                        
        # layer5: torch.Size([256, 96, 24, 24])                                                                                                                                        
        # layer6: torch.Size([256, 128, 24, 24])                                                                                                                                       
        # layer7: torch.Size([256, 96, 24, 24])                                                                                                                                        
        # layer8: torch.Size([256, 128, 12, 12])                                                                                                                                       
        # layer9: torch.Size([256, 256, 12, 12])                                                                                                                                       
        # layer10: torch.Size([256, 192, 12, 12])                                                                                                                                      
        # layer11: torch.Size([256, 256, 6, 6])                                                                                                                                        
        # layer12: torch.Size([256, 448, 6, 6])                                                                                                                                        
        # layer13: torch.Size([256, 256, 6, 6])                                                                                                                                        
        # layer14: torch.Size([256, 448, 6, 6])                                                                                                                                        
        # torch.Size([256, 448, 6, 6]) 

        return x

