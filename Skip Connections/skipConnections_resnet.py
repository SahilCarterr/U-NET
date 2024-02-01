import torch
from torch import nn

# Basic residual block of resnet
# This is generic in sense , it could be used for downsampling of features

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=[1,1]):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], 
                               padding = 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], 
                               padding = 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)

        out = nn.ReLU()(self.bn(self.conv1(x)))
        out = self.bn(self.conv2(out))
        out = out + residual
        out = nn.ReLU()(out)
        return out

downsample = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride = 2,bias = False),
                           nn.BatchNorm2d(128))
resnet_block = nn.Sequential(nn.Conv2d(3, 64, kernel_size=2, stride=2, padding=3,
                                       bias = False),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            ResidualBlock(64, 64),
                            ResidualBlock(64, 64),
                            ResidualBlock(64, 128, stride=[2,1], downsample = downsample))
inputs=torch.randn(1,3,100,100)
output=resnet_block(inputs)
print(f"Input Shape: {inputs.shape}")
print(f"Output Shape: {output.shape}")