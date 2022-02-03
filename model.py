import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
            groups=in_channels, bias=bias, padding='same')
        self.pointwise = nn.Conv2d(in_channels, out_channels,
            kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class EEGNet(nn.Module):
    def __init__(self, F1=8, F2=16, sample_freq=128, num_channels=128, depth=2):
        super(EEGNet, self).__init__()

        self.T = 1153
        self.last_layer_shape = F2 * self.T // 32

        # Block 1
        self.conv2d = nn.Conv2d(1, F1, (1, sample_freq//2), padding='same')
        self.batchnorm1 = nn.BatchNorm2d(F1, False)

        self.depth_wise_conv2d = nn.Conv2d(
            F1, F1*depth, (num_channels, 1), groups=F1, padding='valid'
        )
        self.batchnorm2 = nn.BatchNorm2d(depth*F1, False)
        self.pooling2 = nn.MaxPool2d(1, 4)

        # Block 2
        self.separable_conv_2d = SeparableConv2d(depth*F1, F2, (1 , 16))
        self.batchnorm3 = nn.BatchNorm2d(F2, False)
        self.pooling3 = nn.MaxPool2d((1, 8))

        # Classifier
        self.fc = nn.Linear(self.last_layer_shape, 4)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # Block 1
        x = torch.unsqueeze(x, 1)
        x = self.conv2d(x)
        x = self.batchnorm1(x)

        x = self.depth_wise_conv2d(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = F.dropout(x, 0.25)

        # Block 2
        x = self.separable_conv_2d(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling3(x)
        x = F.dropout(x, 0.25)

        # Classifier
        x = x.reshape(-1, self.last_layer_shape)
        x = self.softmax(self.fc(x))

        return x
