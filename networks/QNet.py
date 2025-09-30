import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_net(nn.Module):
    def __init__(self, Dim_in, act_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=Dim_in,out_channels=32,kernel_size=(8,8),stride=(4,4)) #(84-8)/4 +1 =76/4 +1 =20
        self.maxpool1 =nn.MaxPool2d(2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4,4), stride=(2,2)) # (20-4)/2 +1 =9
        self.maxpool2 = nn.MaxPool2d(2, stride=2,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1)) # (9-3)/1 +1 =7  7*7*64
        self.fc1   = nn.Linear(in_features=576,out_features=576)
        self.fc2   = nn.Linear(in_features=576,out_features=act_dim)
        self.Relu =nn.ReLU()
    def forward(self,x): # inshape (batch,channel,x,y,channel)
        x = self.conv1(x)
        x=  self.Relu(x)
        #
        x = self.Relu(self.conv2(x))
        x = self.maxpool1(x)
        x = self.Relu(self.conv3(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0),-1)
        x = self.Relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DuelingQNet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(DuelingQNet, self).__init__()
        C, H, W = obs_dim  # 通道数、高、宽

        # 卷积特征提取
        self.feature = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )

        # 计算卷积输出维度
        with torch.no_grad():
            test_input = torch.zeros(1, C, H, W)
            feat_dim = self.feature(test_input).shape[1]

        # 状态价值流 V(s)
        self.value = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )

        # 优势流 A(s, a)
        self.advantage = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        """前向传播: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))"""
        x = self.feature(x)
        V = self.value(x)
        A = self.advantage(x)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q