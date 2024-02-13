import torchvision
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10('../data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_loder = DataLoader(train_data, batch_size=100, shuffle=True)
test_data = torchvision.datasets.CIFAR10('../data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=True, num_workers=0, drop_last=True)

#####with_transform