import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, num_channels, use_conv1=False, strides=1):
        super(Residual, self).__init__()
        self.use_conv1 = use_conv1
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.LazyBatchNorm2d()
        self.relu1 = nn.ReLU()
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.LazyBatchNorm2d()
        self.relu2 = nn.ReLU()
        if self.use_conv1:
            self.shortcut = nn.Sequential(nn.LazyConv2d(num_channels, kernel_size=1, stride=strides),
                                          nn.LazyBatchNorm2d())
        else:
            self.conv3 = None

    def forward(self, X):
        identity = X
        Y = self.conv1(X)
        Y = self.bn1(Y)
        Y = self.relu1(Y)
        Y = self.conv2(Y)
        Y = self.bn2(Y)
        # Y = self.relu2(Y)
        if self.use_conv1:
            identity = self.shortcut(identity)
        Y += identity
        return F.relu(Y)


class ResNet(nn.Module):
    def __init__(self, out_channels, num_classes):
        super(ResNet, self).__init__()
        self.net = nn.Sequential()
        self.fist_blk = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.residual_blk = self.residual_creation(out_channels)
        self.output_blk = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )
        self.net.add_module('FirstBlk', self.fist_blk)

    def forward(self, X):
        X = self.fist_blk(X)
        X = self.residual_blk(X)
        X = self.output_blk(X)
        return X

    @staticmethod
    def residual_creation(out_channels):
        blk = []
        for i, num_channel in enumerate(out_channels):
            if i != 0 and i % 2 == 0:
                blk.append(Residual(num_channel, use_conv1=True, strides=2))
            else:
                blk.append(Residual(num_channel))
        return nn.Sequential(*blk)


class RestNet18(ResNet):
    def __init__(self):
        super(RestNet18, self).__init__((64, 64, 128, 128, 256, 256, 512, 512), 10)
