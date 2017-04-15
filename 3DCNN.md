3D-CNN区别于传统的深度神经网络的地方就在于能够考虑时间维度上的信息。

传统的2D的CNN框架
$$
v_{ij}^{xy}=tanh(b_{ij}+\sum\limits{m}\sum\limits_{p=0}^{P_i -1} \sum\limits_{q = 0}^{Q_i-1}w_{ijm}^{pq}v_{(i-1)m}^{(x+p)(y+q)})
$$
改进之后的3D-CNN的计算框架：
$$
v_{ij}^{xyz}=tanh(b_{ij} + \sum\limits_{m} \sum\limits_{p=1}^{P_i-1}\sum\limits_{q=0}^{Q_i-1}\sum\limits_{r=0}^{R_i-1} w_{ijm}^{pqr}v_{(i-1)m}^{(x+p)(y+q)(z+r)})
$$
