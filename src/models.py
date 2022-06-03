import torch
import torch.utils.data
from torch.nn import functional as F
from torch import nn
import numpy as np


class MyModel(nn.Module):

    def __init__(self,
                 input_size=25,
                 num_out_channels=1,
                 conv_channels=8,
                 num_dense=64,
                 num_in_channels=1,
                 kernel_size=3,
                 stride=1):
        super(MyModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(num_in_channels, conv_channels, kernel_size, stride),
            nn.BatchNorm2d(conv_channels), nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size, stride),
            nn.BatchNorm2d(conv_channels), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten())
        self.num_dense_input = self.features.forward(
            torch.Tensor(np.zeros(
                (2, num_in_channels, input_size, input_size)))).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(self.num_dense_input, num_dense),
            nn.BatchNorm1d(num_dense), nn.ReLU(),
            nn.Linear(num_dense, num_dense), nn.BatchNorm1d(num_dense),
            nn.ReLU(), nn.Linear(
                num_dense,
                num_out_channels,
            ))

    def forward_repr(self, x):
        x = self.features(x)
        # print(x.shape)
        x = self.classifier[0](x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    test_model = MyModel()
