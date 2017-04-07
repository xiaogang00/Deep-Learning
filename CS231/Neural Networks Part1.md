多层的神经网络：

A three-layer neural network could analogously look like：
$$
s = W_3 max(0, W_2 max(0,W_1x))
$$
主要还是介绍了neuron的模型
$$
Output = f(\sum\limits_{i}w_i x_i + b)
$$
f为activation function

一个简单的例子:

```python
class Neuron(object):
# ...
def forward(inputs):
""" assume inputs and weights are 1-D numpy arrays and bias is a number """
cell_body_sum = np.sum(inputs * self.weights) + self.bias
firing_rate = 1.0 / (1.0 + math.exp(-cell_body_sum)) # sigmoid activation function
return firing_rate
```

