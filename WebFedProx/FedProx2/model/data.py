import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class loader(object):
    def __init__(self, data_path='./dataset/', batch_size=64):
        self.data_path = data_path
        self.batch_size = batch_size
        self.__load_dataset()

    def __load_dataset(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_dataset = datasets.MNIST(
            self.data_path, train=True, download=False, transform=transform)
        self.test_dataset = datasets.MNIST(
            self.data_path, train=False, download=False, transform=transform)

    def get_train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def get_test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_loader(self):
        return self.get_train_loader(), self.get_test_loader()
