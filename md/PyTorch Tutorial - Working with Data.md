源码地址：https://github.com/wallace-lai/pytorch-tutorial-notes.git

# 1. Tensor
Tensor的定义：
- Tensors are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.

Tensor和numpy中ndarray的区别：
- Tensors are similar to NumPy’s ndarrays, except that tensors can run on GPUs or other hardware accelerators. 【PyTorch中的Tensor可以使用硬件加速】
- In fact, tensors and NumPy arrays can often share the same underlying memory, eliminating the need to copy data【PyTorch中的Tensor往往和Numpy中的ndarray共享同一个底层存储，所以修改其中任何一个会影响另一个】

## 1.1 Tensor初始化
（1）从原始数据中初始化tensor
- `torch.tensor()`：从原始数据中初始化tensor
```py
>>> d1 = [[1, 2], [3, 4]]
>>> d2 = (1, 2, 3, 4)
>>> d3 = ((1, 2), (3, 4))
>>> d1_t = torch.tensor(d1)
>>> d2_t = torch.tensor(d2)
>>> d3_t = torch.tensor(d3)
>>> d1_t
tensor([[1, 2],
        [3, 4]])
>>> d2_t
tensor([1, 2, 3, 4])
>>> d3_t
tensor([[1, 2],
        [3, 4]])
>>>
```

（2）从numpy中的array初始化tensor
- `np.array()`：从原始数据转换成numpy中的array
- `torch.from_numpy()`：从numpy中的array转换成tensor

```py
>>> d1 = [[1, 2], [3, 4]]
>>> d1_np = np.array(d1)
>>> d1_np
array([[1, 2],
       [3, 4]])
>>> d1_t = torch.from_numpy(d1_np)
>>> d1_t
tensor([[1, 2],
        [3, 4]])
>>>
```

（3）从tensor中初始化另一个tensor
- `torch.ones_like(x_data)`：新的tensor属性（shape和datatype）将会保留不变；
- `torch.rand_like(x_data, dtype=torch.float)`：新tensor属性中的datatype将会变成float类型；

```py
>>> d1_t
tensor([[1, 2],
        [3, 4]])
>>> x_ones = torch.ones_like(d1_t)
>>> x_rand = torch.rand_like(d1_t, dtype=torch.float)
>>>
>>> x_ones
tensor([[1, 1],
        [1, 1]])
>>> x_rand
tensor([[0.9793, 0.4337],
        [0.0333, 0.7139]])
>>>
```

（4）以随机值或者常量构造tensor
- `shape = (2, 3)`：指定tensor的维度为2行3列；
- `torch.rand()`：随机值填充tensor；
- `torch.ones()`：全1值填充tensor；
- `torch.zeros()`：全0值填充tensor；

```py
>>> shape = (2, 3)
>>> x_rand = torch.rand(shape)
>>> x_ones = torch.ones(shape)
>>> x_zero = torch.zeros(shape)
>>>
>>> x_rand
tensor([[0.6677, 0.8990, 0.8063],
        [0.7190, 0.1690, 0.6588]])
>>> x_ones
tensor([[1., 1., 1.],
        [1., 1., 1.]])
>>> x_zero
tensor([[0., 0., 0.],
        [0., 0., 0.]])
>>>
```

## 1.2 Tensor属性

```py
>>> t = torch.rand(3, 4)
>>> t
tensor([[0.6075, 0.3464, 0.6556, 0.4764],
        [0.2198, 0.8475, 0.1372, 0.6179],
        [0.1765, 0.6938, 0.7785, 0.7702]])
>>> t.shape
torch.Size([3, 4])
>>> t.dtype
torch.float32
>>> t.device
device(type='cpu')
>>>
```

- `t.shape`：tensor的维度信息；
- `t.dtype`：tensor中值的类型；
- `t.device`：tensor所存储在哪个设备中；

## 1.3 Tensor操作
（1）将tensor移动到加速器上（CUDA、MPS等）
```py
>>> if torch.accelerator.is_available():
...     t = t.to(torch.accelerator.current_accelerator())
...
>>> torch.accelerator.is_available()
False
>>> t.device
device(type='cpu')
>>>
```

- `tensor.to()`：tensor默认在CPU上存储和运行，使用该方法可以将tensor移动到加速器上运行；

（2）索引和切片
```py
>>> t = torch.rand(2, 3)
>>> t
tensor([[0.4233, 0.6658, 0.5413],
        [0.3009, 0.4306, 0.8417]])
>>>
```

- 第一行
```py
>>> t[0]
tensor([0.4233, 0.6658, 0.5413])
>>>
```
- 第一列
```py
>>> t[:,0]
tensor([0.4233, 0.3009])
>>>
```

- 最后一列
```py
>>> t[...,-1]
tensor([0.5413, 0.8417])
>>> t[:, -1]
tensor([0.5413, 0.8417])
>>>
```

- 第二行第二列的元素
```py
>>> t[1, 1]
tensor(0.4306)
>>>
```

- 第二行的值全部修改为1
```py
>>> t[1] = 1
>>> t
tensor([[0.4233, 0.6658, 0.5413],
        [1.0000, 1.0000, 1.0000]])
>>>
```

（3）拼接
```python
>>> t
tensor([[0.6137, 0.2641, 0.9753],
        [0.4740, 0.1279, 0.8192]])
>>> t1 = torch.cat([t, t, t], dim=0)
>>> t2 = torch.cat([t, t, t], dim=1)
>>> t1
tensor([[0.6137, 0.2641, 0.9753],
        [0.4740, 0.1279, 0.8192],
        [0.6137, 0.2641, 0.9753],
        [0.4740, 0.1279, 0.8192],
        [0.6137, 0.2641, 0.9753],
        [0.4740, 0.1279, 0.8192]])
>>> t2
tensor([[0.6137, 0.2641, 0.9753, 0.6137, 0.2641, 0.9753, 0.6137, 0.2641, 0.9753],
        [0.4740, 0.1279, 0.8192, 0.4740, 0.1279, 0.8192, 0.4740, 0.1279, 0.8192]])
>>>
```

- `dim=0`：按行拼接；
- `dim=1`：按列拼接；

（4）数学操作

```py
>>> t1 = torch.ones(2, 3)
>>> t2 = torch.zeros(3, 2)
>>> t1
tensor([[1., 1., 1.],
        [1., 1., 1.]])
>>> t2
tensor([[0., 0.],
        [0., 0.],
        [0., 0.]])
```

- 矩阵乘法（要求第一个张量的列数等于第二个张量的行数）
```py
>>> y1 = t1 @ t2
>>> y2 = t1.matmul(t2)
>>> y1
tensor([[0., 0.],
        [0., 0.]])
>>> y2
tensor([[0., 0.],
        [0., 0.]])
>>>
```

- 哈达码乘积（要求两个张量维度相等）
```py
>>> y3 = t1 * t2
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1
>>> t3 = torch.tensor([[1, 2, 3], [4, 5, 6]])
>>> y3 = t1 * t3
>>> y4 = t1.mul(t3)
>>> y3
tensor([[1., 2., 3.],
        [4., 5., 6.]])
>>> y4
tensor([[1., 2., 3.],
        [4., 5., 6.]])
>>>
```

- 矩阵转置
```py
>>> t1, t1.shape
(tensor([[1., 1., 1.],
        [1., 1., 1.]]), torch.Size([2, 3]))
>>> t1.T, t1.T.shape
(tensor([[1., 1.],
        [1., 1.],
        [1., 1.]]), torch.Size([3, 2]))
>>> t1.t(), t1.t().shape
(tensor([[1., 1.],
        [1., 1.],
        [1., 1.]]), torch.Size([3, 2]))
>>>
```

（5）单元素tensor

```py
>>> t1
tensor([[1., 1., 1.],
        [1., 1., 1.]])
>>> sum = t1.sum()
>>> sum
tensor(6.)
>>> sum.item()
6.0
>>> type(sum.item())
<class 'float'>
>>> t1.item()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: a Tensor with 6 elements cannot be converted to Scalar
>>>
```

- `t1.sum()`：对所有元素进行求和，结果是一个单元素的tensor；
- `sum.item()`：取出单元素tensor中的值，非单元素tensor不可转换成标量；

（6）就地操作
```py
>>> t1
tensor([[1., 1., 1.],
        [1., 1., 1.]])
>>> t1.t()
tensor([[1., 1.],
        [1., 1.],
        [1., 1.]])
>>> t1
tensor([[1., 1., 1.],
        [1., 1., 1.]])
>>> t1.t_()
tensor([[1., 1.],
        [1., 1.],
        [1., 1.]])
>>> t1
tensor([[1., 1.],
        [1., 1.],
        [1., 1.]])
>>>
```

- 所有的就地操作函数都以下划线结尾；
- 不赞成使用就地操作函数；

## 1.4 Tensor与numpy之间的联系
PyTorch中的tensor和numpy之间的ndarray之间的联系：
- Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.

（1）对tensor的修改会影响array
```py
>>> t = torch.ones(2, 3)
>>> n = t.numpy()
>>> t
tensor([[1., 1., 1.],
        [1., 1., 1.]])
>>> n
array([[1., 1., 1.],
       [1., 1., 1.]], dtype=float32)
>>>
>>> t.add_(1)
tensor([[2., 2., 2.],
        [2., 2., 2.]])
>>> n
array([[2., 2., 2.],
       [2., 2., 2.]], dtype=float32)
>>>
```

（2）对array的修改会影响tensor
```py
>>> n = np.ones([2, 3])
>>> t = torch.from_numpy(n)
>>> n
array([[1., 1., 1.],
       [1., 1., 1.]])
>>> t
tensor([[1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
>>>
>>> np.add(n, 1, out=n)
array([[2., 2., 2.],
       [2., 2., 2.]])
>>> t
tensor([[2., 2., 2.],
        [2., 2., 2.]], dtype=torch.float64)
>>>
```

## 附：Tensor的常见操作
https://docs.pytorch.org/docs/stable/torch.html

# 2. Datasets and DataLoaders

（1）定义
- `torch.utils.data.Dataset`：stores the samples and their corresponding labels;

- `torch.utils.data.DataLoader`：wraps an iterable around the Dataset to enable easy access to the samples

（2）PyTorch预定义数据集
- 视觉
```py
imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=args.nThreads)
```

https://docs.pytorch.org/vision/stable/datasets.html

- 文本
```py
# import datasets
from torchtext.datasets import IMDB

train_iter = IMDB(split='train')

def tokenize(label, line):
    return line.split()

tokens = []
for label, line in train_iter:
    tokens += tokenize(label, line)
```
https://docs.pytorch.org/text/stable/datasets.html

- 语音

```py
yesno_data = torchaudio.datasets.YESNO('.', download=True)
data_loader = torch.utils.data.DataLoader(
    yesno_data,
    batch_size=1,
    shuffle=True,
    num_workers=args.nThreads)
```

https://docs.pytorch.org/audio/stable/datasets.html

## 2.1 加载数据集
（1）加载Fashion-MNIST数据集
```py
import os
import torch

import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

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
```

## 2.2 数据集遍历和可视化
（1）可视化Fashion-MNIST数据集

```py
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
```

## 2.3 创建自定义的Dataset
假设你所有的图片都保存在`img_dir`路径下，有一个`annotations_file`文件保存了图片名和对应的标签值，如下所示。

```
tshirt1.jpg, 0
tshirt2.jpg, 0
...
ankleboot999.jpg, 9
```

针对上述假设，你可以用下面的代码实现自定义的Dataset类。

```py
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
```

自定义Dataset类时，在继承Dataset父类以后，必须实现下面这三个方法：

- `__init__`：初始化
- `__len__`：返回数据集中的样本个数
- `__getitem__`：根据索引返回对应的样本和标签

## 2.4 创建DataLoader
（1）使用DataLoader遍历Dataset时，可以指定每个小批量数据的大小和是否打乱样本顺序
```py
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

## 2.5 使用DataLoader遍历数据集

```py
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

# 3. Transforms
All TorchVision datasets have two parameters:
- transform：用于转换样本
- target_transform：用于转换标签

```py
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
```

（1）`ToTensor()`的作用
ToTensor converts a PIL image or NumPy ndarray into a FloatTensor. and scales the image’s pixel intensity values in the range `[0., 1.]`

- 将图像的PIL对象转换成向量；
- 将图像像素值归一化到0到1之间；

（2）`lambda`的作用
```py
torch.zeros(10, dtype=torch.float)
```
- 先创建一个全0值的10个元素的向量；

```py
.scatter_(0, torch.tensor(y), value=1)
```
- 将位置`y`下的0赋值成1，以此完成one-shot编码；

