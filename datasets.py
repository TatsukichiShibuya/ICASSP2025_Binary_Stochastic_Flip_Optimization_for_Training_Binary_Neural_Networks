import os
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms


with open(".strage_path.txt") as f:
    STRAGE_PATH = f.read()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        feature = self.X[index]
        label = self.y[index]
        return feature, label


class Subset(Dataset):
    def __init__(self, dataset, indices, transform=False):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return self.transform(image), label

    def __len__(self):
        return len(self.indices)


class OtherDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.classes.sort()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls in self.classes:
            class_path = os.path.join(self.root_dir, cls)
            image_path = glob(os.path.join(class_path, "*.jpg"))
            image_path.sort()
            for img_name in image_path:
                images.append((img_name, self.class_to_idx[cls]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        return image, label



def get_dataset(dataset_name, test=True, resize=256):
    
    resize2d = (resize, resize)
    valid_ratio = 0.15
    
    if dataset_name=="MNIST":
        transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = tv.datasets.MNIST(root=os.path.join(STRAGE_PATH, 'dataset'), train=True, download=True, transform=transform)
        test_set = tv.datasets.MNIST(root=os.path.join(STRAGE_PATH, 'dataset'), train=False, download=True, transform=transform)

    elif dataset_name=="CIFAR10":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.Resize(resize2d), transforms.ToTensor(), normalize])
        train_set = tv.datasets.CIFAR10(root=os.path.join(STRAGE_PATH, 'dataset'), train=True, download=True, transform=transform)
        test_set = tv.datasets.CIFAR10(root=os.path.join(STRAGE_PATH, 'dataset'), train=False, download=True, transform=transform)
    
    elif dataset_name=="CIFAR100":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.Resize(resize2d), transforms.ToTensor(), normalize])
        train_set = tv.datasets.CIFAR100(root=os.path.join(STRAGE_PATH, 'dataset'), train=True, download=True, transform=transform)
        test_set = tv.datasets.CIFAR100(root=os.path.join(STRAGE_PATH, 'dataset'), train=False, download=True, transform=transform)
    
    elif dataset_name=="Caltech101":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.Resize(resize2d), transforms.ToTensor(), normalize])
        data = OtherDataset(root_dir=os.path.join(STRAGE_PATH, 'dataset/caltech-101'))
        train_indices, test_indices = train_test_split(range(len(data)), test_size=valid_ratio, random_state=42)
        train_set = Subset(data, train_indices, transform=transform)
        test_set = Subset(data, test_indices, transform=transform)
    
    elif dataset_name=="Caltech256":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.Resize(resize2d), transforms.ToTensor(), normalize])
        data = OtherDataset(root_dir=os.path.join(STRAGE_PATH, 'dataset/caltech-256'))
        train_indices, test_indices = train_test_split(range(len(data)), test_size=valid_ratio, random_state=42)
        train_set = Subset(data, train_indices, transform=transform)
        test_set = Subset(data, test_indices, transform=transform)

    elif dataset_name=="OxfordPets":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.Resize(resize2d), transforms.ToTensor(), normalize])
        data = OtherDataset(root_dir=os.path.join(STRAGE_PATH, 'dataset/oxford-pets'))
        train_indices, test_indices = train_test_split(range(len(data)), test_size=valid_ratio, random_state=42)
        train_set = Subset(data, train_indices, transform=transform)
        test_set = Subset(data, test_indices, transform=transform_)

    elif dataset_name=="OxfordFlowers":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.Resize(resize2d), transforms.ToTensor(), normalize])
        data = OtherDataset(root_dir=os.path.join(STRAGE_PATH, 'dataset/oxford-flowers'))
        train_indices, test_indices = train_test_split(range(len(data)), test_size=valid_ratio, random_state=42)
        train_set = Subset(data, train_indices, transform=transform)
        test_set = Subset(data, test_indices, transform=transform)

    else:
        raise NotImplementedError()
    
    loss_function = nn.CrossEntropyLoss(reduction="mean")
    
    return train_set, test_set, loss_function
