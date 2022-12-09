'''Loss Prediction Module in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np

class LossNet_Linear(nn.Module):
    def __init__(self,  num_channels=[2399, 512], interm_dim=128):
        super(LossNet_Linear, self).__init__()
        
        # self.FC1 = nn.Linear(num_channels[0], interm_dim)
        # self.FC2 = nn.Linear(num_channels[1], interm_dim)

        self.FC1 = nn.Linear(num_channels[0]//2, interm_dim)
        self.FC2 = nn.Linear(num_channels[1]//2, interm_dim)
        self.GAP1 = nn.AvgPool1d(kernel_size=2)
        self.GAP2 = nn.AvgPool1d(kernel_size=2)

        self.linear1 = nn.Linear(2 * interm_dim, 64)
        self.linear2 = nn.Linear(64, 1)

    
    def forward(self, features):

        # out1 = F.relu(self.FC1(features[0]))
        # out2 = F.relu(self.FC2(features[1]))

        out1 = features[0].unsqueeze(1)
        out1 = self.GAP1(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = features[1].unsqueeze(1)
        out2 = self.GAP1(out2)
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out = self.linear1(torch.cat((out1, out2), 1))

        out = self.linear2(out)

        return out

class LossNet_v1005(nn.Module):
    def __init__(self,  num_channels=[2399, 512], interm_dim=128):
        super(LossNet_v1005, self).__init__()
        
        # self.FC1 = nn.Linear(num_channels[0], interm_dim)
        # self.FC2 = nn.Linear(num_channels[1], interm_dim)

        self.FC1 = nn.Linear(num_channels[0]//2, interm_dim)
        self.FC2 = nn.Linear(num_channels[1]//2, interm_dim)
        
        self.GAP1 = nn.AvgPool1d(kernel_size=2)
        self.GAP2 = nn.AvgPool1d(kernel_size=2)

        self.linear1 = nn.Linear(interm_dim, 1)

    def forward(self, features):

        # out1 = F.relu(self.FC1(features[0]))
        # out2 = F.relu(self.FC2(features[1]))

        # out1 = features[0].unsqueeze(1)
        # out1 = self.GAP1(out1)
        # out1 = out1.view(out1.size(0), -1)
        # out1 = F.relu(self.FC1(out1))

        out2 = features[1].unsqueeze(1)
        out2 = self.GAP2(out2)
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))


        out = self.linear1(out2)

        return out