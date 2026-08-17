from abc import ABC
from torch import nn


class LeNet5(nn.Module, ABC):
    def __init__(self, in_dim, n_class):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, 6, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        fc_in = 256 if in_dim == 1 else 400

        self.fc = nn.Sequential(
            nn.Linear(fc_in, 120),
            nn.Linear(120, 84),
            nn.Linear(84, n_class)
        )

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv = out_conv2.view(out_conv2.size(0), -1)
        out = self.fc(out_conv)
        return out


def lenet5(dataset='cifar10'):
    in_dim = 1 if dataset == 'mnist' else 3
    return LeNet5(in_dim, 10)
