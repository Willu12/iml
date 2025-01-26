"""
Module containing Convolutional Neural Network (CNN) models for audio 
classification tasks using spectrograms.
"""

import torch
from torch import nn
import torch.nn.functional as F


class TutorialCNN(nn.Module):
    """
    A tutorial CNN model for educational purposes, suitable for general
    image classification tasks.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        pool (nn.MaxPool2d): Max pooling layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Output fully connected layer.
    """

    def __init__(self):
        super(TutorialCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        """
        Defines the forward pass of the CNN.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class OriginalSizeCNN(nn.Module):
    """
    CNN for 94x128 grayscale images
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flattened_size = 64 * (94 // 8) * (128 // 8)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(-1, self.flattened_size)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# This is generic in the sense, it could be used for downsampling of features.
# https://www.analyticsvidhya.com/blog/2021/08/all-you-need-to-know-about-skip-connections/
class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=[1, 1],
        kernel_size=3,
        downsample=None,
        batch_normalization=True,
        residual_add=True,
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride[0],
            padding=kernel_size // 2,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride[1],
            padding=kernel_size // 2,
            bias=False,
        )

        self.batch_normalization = batch_normalization
        self.residual_add = residual_add
        if batch_normalization:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)

        out = F.relu(self.conv1(x))
        if self.batch_normalization:
            out = self.bn1(out)
        out = self.conv2(out)
        if self.batch_normalization:
            out = self.bn2(out)

        if self.residual_add:
            out += residual

        out = F.relu(out)
        return out


class OurResNet(nn.Module):
    """
    Residual CNN for 94x128 grayscale images
    """

    def __init__(self, residual_connections=True, batch_normalization=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, bias=False
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        downsample = nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False)

        self.res1 = ResidualBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            residual_add=residual_connections,
            batch_normalization=batch_normalization,
        )
        self.res2 = ResidualBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            residual_add=residual_connections,
            batch_normalization=batch_normalization,
        )
        self.res3 = ResidualBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=[2, 1],
            downsample=downsample,
            residual_add=residual_connections,
            batch_normalization=batch_normalization,
        )

        self.flattened_size = 64 * (94 // 4) * (128 // 4)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.res1(x))
        x = F.relu(self.res2(x))
        x = F.relu(self.res3(x))

        x = x.view(-1, self.flattened_size)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
