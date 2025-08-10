import os
import torch

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

def one_shot_encode_lambda():
    return Lambda(
        lambda y :
        torch.zeros(10, dtype=torch.float)
        .scatter_(0, torch.tensor(y), value=1)
    )

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    ds = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=one_shot_encode_lambda()
    )
