import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class loader(object):
    def __init__(self, cmd='mnist', batch_size=64):
        self.cmd = cmd
        self.batch_size = batch_size
        self.__load_dataset()
        self.__get_index()

    def __load_dataset(self):
        # ✅ MNIST만 다운로드
        self.train_mnist = datasets.MNIST('./dataset/',
                                          train=True, download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                          ]))
        self.test_mnist = datasets.MNIST('./dataset/',
                                         train=False, download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))
                                         ]))

    def __get_index(self):
        self.train_dataset = self.train_mnist
        self.test_dataset = self.test_mnist

        self.indices = [[], [], [], [], [], [], [], [], [], []]
        for index, data in enumerate(self.train_dataset):
            self.indices[data[1]].append(index)

    def get_loader(self, rank):
        dataset_indices = []
        difference = list(set(range(10)).difference(set(rank)))
        for i in difference:
            dataset_indices.extend(self.indices[i])

        dataset = torch.utils.data.Subset(self.train_dataset, dataset_indices)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader