import os
import torch

import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.io import decode_image

# 自定义Dataset实现
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    # 下载FashionMNIST数据集
    train_data = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    # 数据集遍历与可视化
    labels_map = {
        0 : "T-Shirt",
        1 : "Trouser",
        2 : "Pullover",
        3 : "Dress",
        4 : "Coat",
        5 : "Sandal",
        6 : "Shirt",
        7 : "Sneaker",
        8 : "Bag",
        9 : "Ankle Boot"
    }

    figsize = (8, 8)
    fig = plt.figure(figsize=figsize)
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]
        fig.add_subplot(rows, cols, i)
        
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    
    plt.show()



