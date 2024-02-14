import torchvision
from torch.utils.data import DataLoader
import argparse

def data_loader(dataset,batch_size):
    train_data = torchvision.datasets.dataset('../data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    train_loder = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = torchvision.datasets.dataset('../data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=100, help="batch size")
    parser.add_argument("--dataset", type=str, default=None, help="dataset")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    data_loader(args)