Linear classifier:
$$
f(x_i,W,b) = Wx_i +b
$$
Interpreting a linear classifier:

通过将图片转换成一维的向量，我们可以将其作为高维特征空间中的一个点，所以可以按照在低维空间中的做法，用一个或者是多个超平面，来讲我们需要鉴别的物体分开。

对于bias，可以将式子变为线性，也就是可以将1加到我们需要的模型之中：
$$
f(x_i,W) = Wx_i
$$
Loss Function:

* Multiclass Support Vector Machine loss:

  $f(x_i,W)$ ，the score for the j-th class is the j-th element:$s_j = f(x_i,W)_j$ 

  The multiclass SVM loss for the i-th example:
  $$
  L_i= \sum\limits_{j \neq y_i} max(0, s_j-s_{yj} + \Delta)
  $$

* 在线性映射中，我们具有如下的损失函数表达式：
  $$
  L_i = \sum\limits_{j \neq y_i} max(0, w_j^Tx_i - w_{yi}^T x_i + \Delta)
  $$

* 除去$max(0,—)$ 之外，还可以使用平方误差函数$max(0,—)^2$ 

* 可以为防止过拟合等线性找到一个正则项：$R(W) = \sum\limits_{k}  \sum\limits_{l}  W_{k,l}^2$ 

* ​