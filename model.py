import torch.nn as nn
import torchvision
import torch


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),  # n, h, w, c
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class FeatureExtractor2(nn.Module):
    def __init__(self, n_channels=1024):
        super(FeatureExtractor2, self).__init__()
        self.n_channels = n_channels

        self.conv_g = DoubleConv(n_channels, 128, 512)
        self.conv_l = DoubleConv(n_channels, 128, 512)

    def forward(self, gx, lx):
        gx = self.conv_g(gx)
        lx = self.conv_l(lx)
        x = torch.cat([gx, lx], dim=1)
        return x


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        ResNet50 = torchvision.models.resnet50(pretrained=True)  # pretrained=True 既要加载网络模型结构，又要加载模型参数
        ResNet50.eval()  # to not do dropout

        self.features = nn.Sequential(*list(ResNet50.children())[0:-3])  # ResNet50取前6部分

    def forward(self, x):
        x = self.features(x)
        return x


"""
    文章所属Q-Network结构
"""


class DQN(nn.Module):
    def __init__(self, t, n_actions):
        super(DQN, self).__init__()
        self.g_dim = 200704
        self.l_dim = 200704
        self.h_dim=t*n_actions
        self.conv_g = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, bias=False),  # n, h, w, c
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.conv_l = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, bias=False),  # n, h, w, c
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=110 + 128*14*14*2, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=11)
        )

    def forward(self, x):
        B = x.size(0)
        start = 0
        g = x[:, start:start + self.g_dim]  # [B, 200704]
        start += self.g_dim
        l = x[:, start:start + self.l_dim]  # [B, 200704]
        start += self.l_dim
        h = x[:, start:start + self.h_dim]  # [B, t * n_actions]
        start += self.h_dim

        # reshape g, l 为 [B, 1024, 14, 14]
        g = g.view(B, 1024, 14, 14)
        l = l.view(B, 1024, 14, 14)

        # 分别卷积压缩并 flatten
        g_feat = self.conv_g(g).view(B, -1)  # [B, 1568]
        l_feat = self.conv_l(l).view(B, -1)  # [B, 1568]

        # 拼接所有特征

        feat = torch.cat([g_feat, l_feat, h], dim=1)

        return self.classifier(feat)

