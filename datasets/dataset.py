import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import FashionMNIST
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import random

class CustomDataLoader:
    def __init__(self, batch_size: int, resolution: int, dataset_name: str, num_workers:int =1, gray_fashion=False):
        self.batch_size = batch_size
        self.res = resolution
        self.dataset_name = dataset_name
        self.num_workers = num_workers
        self.supported_datasets = {
            "celeba": self._load_celeba,
            "fashionmnist": self._load_fashion_mnist,
            "stl10": self._load_stl10
        }
        self.gray_fashion = gray_fashion

    def get_loader(self) -> DataLoader:
        loader_fn = self.supported_datasets.get(self.dataset_name)
        if loader_fn is None:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported.")
        return loader_fn()
    
    def _load_fashion_mnist(self):
        if self.gray_fashion:
            transform = transforms.Compose([
                transforms.Resize(self.res),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]) # gray scale
            ])
        else:
            transform = transforms.Compose([
            transforms.Resize(self.res),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # gray to RGB
            transforms.Normalize([0.5]*3, [0.5]*3),
            ])
        dataset = FashionMNIST(root="./data", train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, drop_last=True, shuffle=True, num_workers=self.num_workers)
    
    def _load_stl10(self):
        transform = transforms.Compose([
            transforms.Resize(self.res),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3) 
        ])
        dataset = STL10(root="./data", split="unlabeled", download=True, transform=transform)
        indices = random.sample(range(len(dataset)), 1000)
        subset = Subset(dataset, indices)
        return DataLoader(dataset, batch_size=self.batch_size, drop_last=True, shuffle=True, num_workers=self.num_workers)
    
    def _load_celeba(self):
        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(self.res),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3) 
        ])
        dataset = datasets.ImageFolder(root="./data/celeba", transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, drop_last=True, shuffle=True, num_workers=self.num_workers)

