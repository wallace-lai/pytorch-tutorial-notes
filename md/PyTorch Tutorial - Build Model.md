源码地址：https://github.com/wallace-lai/pytorch-tutorial-notes.git

# 1. Build the Neural Network

## 1.1 获取训练加速器
在PyTorch中想要获取CUDA加速（如果有的话）是非常简单的，只需要下面的代码获取设备。
```py
    # use accelerator if available
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Use {device} device for traininng.")
```
## 1.2 自定义网络模型
想要自定义网络模型，只需要以下步骤：
- 通过继承`nn.Module`自定义网络模型子类；
- 在`__init__`方法中初始化网络结构；
- 同时实现网络结构的前向传播方法，即`forward`方法；
```py
# 自定义网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

（1）`nn.Flatten`
`nn.Flatten`的功能：We initialize the nn.Flatten layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values ( the minibatch dimension (at dim=0) is maintained)【将每个28 * 28大小的二维矩阵铺开成一个784元素的一维向量】

```py
>>> input = torch.rand(3, 28, 28)
>>> input.shape
torch.Size([3, 28, 28])
>>> flat_input = flatten(input)
>>> flat_input.shape
torch.Size([3, 784])
>>>
```

（2）`nn.Linear`
`nn.Linear`是一个线性变换，默认是带有bias的，原型如下所示：
```py
class torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
```

- `in_features`：每个输入样本维度大小；
- `out_features`：每个输出的维度大小；
- `bias=True`：默认带偏置bias；

```py
>>> flat_input.shape
torch.Size([3, 784])
>>> layer1 = nn.Linear(in_features=28*28, out_features=20)
>>> hidden1 = layer1(flat_input)
>>> hidden1.shape
torch.Size([3, 20])
>>>
```

（3）`nn.ReLU`

非线性激活函数的功能：
Non-linear activations are what create the complex mappings between the model’s inputs and outputs. They are applied after linear transformations to introduce nonlinearity, helping neural networks learn a wide variety of phenomena.

（4）`nn.Sequential`

`nn.Sequential`is an ordered container of modules. The data is passed through all the modules in the same order as defined.

使用`nn.Sequential`可以很方便地定义神经网络结构，比如：

```py
seq_modules = nn.Sequential(
	flatten,
	layer1,
	nn.ReLU(),
	nn.Linear(20, 10)
)

input = torch.rand(3, 28, 28)
logits = seq_modules(input)
```

（5）`nn.Softmax`

```py
>>> linear_rule_stack = nn.Sequential(
...     nn.Linear(28 * 28, 512),
...     nn.ReLU(),
...     nn.Linear(512, 512),
...     nn.ReLU(),
...     nn.Linear(512, 10),
... )
>>> input = torch.rand(3, 28, 28)
>>> flat_input = flatten(input)
>>> logits = linear_rule_stack(flat_input)
>>> logits
tensor([[ 0.0778,  0.0793,  0.0109,  0.0366,  0.0094, -0.0439, -0.0808,  0.0680,
          0.0935, -0.0372],
        [ 0.0371,  0.1048, -0.0136, -0.0095,  0.0110, -0.0562, -0.0784,  0.0839,
          0.1305, -0.0069],
        [ 0.0550,  0.1158,  0.0341, -0.0486,  0.0174, -0.0490, -0.1170,  0.0198,
          0.0626,  0.0084]], grad_fn=<AddmmBackward0>)
>>> logits.shape
torch.Size([3, 10])
>>> softmax = nn.Softmax(dim=1)
>>> pred_prob = softmax(logits)
>>> pred_prob
tensor([[0.1056, 0.1058, 0.0988, 0.1014, 0.0987, 0.0935, 0.0901, 0.1046, 0.1073,
         0.0942],
        [0.1015, 0.1086, 0.0965, 0.0969, 0.0989, 0.0924, 0.0904, 0.1063, 0.1114,
         0.0971],
        [0.1044, 0.1110, 0.1023, 0.0941, 0.1006, 0.0941, 0.0879, 0.1008, 0.1052,
         0.0997]], grad_fn=<SoftmaxBackward0>)
>>> pred_prob[0].sum()
tensor(1., grad_fn=<SumBackward0>)
>>>
```

- 上面代码中`logits`的值范围在`[-infty, infty]`之间；
- 将`logits`传入`nn.Softmax`模块后，设定`dim=1`表明要求将做softmax回归，每一行值的总和为1；


## 1.3 模型参数
```py
>>> linear_relu_stack = nn.Sequential(
...     nn.Linear(28 * 28, 512),
...     nn.ReLU(),
...     nn.Linear(512, 512),
...     nn.ReLU(),
...     nn.Linear(512, 10),
... )
>>> linear_relu_stack
Sequential(
  (0): Linear(in_features=784, out_features=512, bias=True)
  (1): ReLU()
  (2): Linear(in_features=512, out_features=512, bias=True)
  (3): ReLU()
  (4): Linear(in_features=512, out_features=10, bias=True)
)
>>> linear_relu_stack.parameters()
<generator object Module.parameters at 0x0000024C5CBF0200>
>>> for name, param in linear_relu_stack.named_parameters():
...     print(f"Layer : {name}, Size : {param.size()}, Values : {param[:2]} \n")
...
Layer : 0.weight, Size : torch.Size([512, 784]), Values : tensor([[ 0.0120, -0.0191,  0.0297,  ...,  0.0104, -0.0264,  0.0307],
        [-0.0132,  0.0048, -0.0316,  ..., -0.0307,  0.0298,  0.0223]],
       grad_fn=<SliceBackward0>)

Layer : 0.bias, Size : torch.Size([512]), Values : tensor([-0.0029,  0.0104], grad_fn=<SliceBackward0>)

Layer : 2.weight, Size : torch.Size([512, 512]), Values : tensor([[ 0.0065,  0.0419, -0.0182,  ..., -0.0120, -0.0192,  0.0182],
        [-0.0136,  0.0346,  0.0297,  ...,  0.0159,  0.0224,  0.0431]],
       grad_fn=<SliceBackward0>)

Layer : 2.bias, Size : torch.Size([512]), Values : tensor([0.0328, 0.0072], grad_fn=<SliceBackward0>)

Layer : 4.weight, Size : torch.Size([10, 512]), Values : tensor([[ 0.0232, -0.0035,  0.0028,  ..., -0.0271, -0.0182, -0.0041],
        [-0.0369,  0.0045, -0.0260,  ...,  0.0352,  0.0312, -0.0375]],
       grad_fn=<SliceBackward0>)

Layer : 4.bias, Size : torch.Size([10]), Values : tensor([0.0300, 0.0091], grad_fn=<SliceBackward0>)

>>>
```

PyTorch将会自动跟踪模型中的所有参数，通过`parameters()`或者`named_parameters()`方法可以访问这些参数

