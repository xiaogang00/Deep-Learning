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