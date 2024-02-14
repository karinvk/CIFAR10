import torchvision
from torch.utils.data import DataLoader
import argparse

def data_loader(dataset,batch_size):
    dataset_func = getattr(torchvision.datasets, dataset)
    train_data = dataset_func('./', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = dataset_func('./', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return train_data,test_data,train_loader,test_loader
