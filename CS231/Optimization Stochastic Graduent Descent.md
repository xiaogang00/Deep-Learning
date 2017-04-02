Visualizing the loss function:

For a singe example we have :
$$
L_i = \sum\limits_{j \neq y_i} [max(0, w_j^Tx_i - w_{yi}^T x_i +1)]
$$
It is clear from the equation that the data loss for each example is a sum of lieaner functions of $W$

但是在一般的SVM loss中，我们存在不同的类，而且需要将不同类的Loss做平均作为最后的损失函数
$$
L = (L_0+L_1+L2)/3
$$
最后经过相加得到的Loss函数会成为一个凸函数，所以最优化Loss的问题在最后就成为一个凸优化的问题。



Optimization:如何进行优化？

* 随机搜索，包括全局随机搜索，和局部随机搜索
* 梯度下降

由此我们需要知道如何计算梯度

计算梯度的程序(python):

```python
def eval_numerical_gradient(f, x):
    fx = f(x)
    grad = np.zeros(x.shape)
    h = 0.0001
    it = np.nditer(x, flag=['multi-index'], op_flags = ['readwrite'])
    #np.nditer 是一个迭代器
    while not it.finished:
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h 
        fxh = f(x)
        x[ix] = old_value
        
        #compute the partial derivative
        grad[ix] = (fxh - fx) / h
        it.iternext()
    return grad
        
    
```

Lets compute the gradient for the CIFAR-10 loss function at some random point in the weight space:

```python
def CIFAR10_loss_fun(W):
    return L(X_train, Y_train, W)
W = np.random.rand(10, 3073) * 0.001
df = eval_numerical_gradient(CIFAR10_loss_fun, W)
loss_original = CIFAR10_loss_fun(W)
print 'original loss : %f'

for step_size_log in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]:
    step_size = 10 ** step_size_log
    W_new = W - step_size * df
    loss_new = CIFAR10_loss_fun(W_new)
    print 'for step size %f new loss: %f'  %(step_size, loss_new)
```

