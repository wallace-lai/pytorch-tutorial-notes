# 1. 自动微分
## 1.1 自动微分引擎
To compute those gradients, PyTorch has a built-in differentiation engine called `torch.autograd`. It supports automatic computation of gradient for any computational graph.

```py
>>> x = torch.ones(5)  # input tensor
>>> y = torch.zeros(3)  # expected output
>>> w = torch.randn(5, 3, requires_grad=True)
>>> b = torch.randn(3, requires_grad=True)
>>> z = torch.matmul(x, w)+b
>>> loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
>>> print(f"Gradient function for z = {z.grad_fn}")
Gradient function for z = <AddBackward0 object at 0x000002A714F92830>
>>> print(f"Gradient function for loss = {loss.grad_fn}")
Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0x000002A7151148B0>
>>>
```

- 对于我们需要优化的参数`w`和`b`，我们需要计算`loss`函数对于参数的梯度，所以将它们的`requires_grad`设置为True；
- A function that we apply to tensors to construct computational graph is in fact an object of class Function;
- This object knows how to compute the function in the forward direction, and also how to compute its derivative during the backward propagation step
- A reference to the backward propagation function is stored in grad_fn property of a tensor.

## 1.2 计算梯度

```py
>>> loss.backward()
>>> print(w.grad)
tensor([[0.0144, 0.3300, 0.0004],
        [0.0144, 0.3300, 0.0004],
        [0.0144, 0.3300, 0.0004],
        [0.0144, 0.3300, 0.0004],
        [0.0144, 0.3300, 0.0004]])
>>> print(b.grad)
tensor([0.0144, 0.3300, 0.0004])
>>>
```

- 为了优化神经网络中的参数，我们需要计算损失函数对参数的导数；
- 也即我们需要计算 $\frac{\partial loss}{\partial w}$和$\frac{\partial loss}{\partial b}$表达式在给定`x`和`y`情况下的值；
- 求梯度的过程如上代码所示；

## 1.3 禁用梯度跟踪
存在以下几种情况，我们不需要计算参数的梯度，比如：
- 我们只想做前向推理，不需要使用方向梯度更新参数；
- 使得部分参数冻结，不再更新；

（1）使用`torch.no_grad()`代码块实现
```py
>>> z = torch.matmul(x, w)+b
>>> z.requires_grad
True
>>>
>>> with torch.no_grad():
...     z = torch.matmul(x, w)+b
...
>>> z.requires_grad
False
>>>
```

（2）或者使用tensor的`detach()`方法
```py
>>> z = torch.matmul(x, w)+b
>>> z.requires_grad
True
>>> z_det = z.detach()
>>> z_det.requires_grad
False
>>>
```