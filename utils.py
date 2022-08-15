from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torch


def image_augmentation():
    train_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])
    test_aug = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])
    return train_aug, test_aug


def get_FashionMNIST(train_aug, test_aug, root):
    train_set = FashionMNIST(root=root, train=True, transform=train_aug, download=True)
    test_set = FashionMNIST(root=root, train=True, transform=test_aug, download=True)
    return train_set, test_set


def create_dataloader(train_set, val_set, batch_size):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader, val_loader


def get_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device

