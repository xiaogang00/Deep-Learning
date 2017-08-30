#### Generation

什么是生成（generation）？就是模型通过学习一些数据，然后生成类似的数据。让机器看一些动物图片，然后自己来产生动物的图片，这就是生成。

以前就有很多可以用来生成的技术了，比如auto-encoder（自编码器）。你训练一个encoder，把input转换成code，然后训练一个decoder，把code转换成一个image，然后计算得到的image和input之间的MSE（mean square error），训练完这个model之后，取出后半部分NN Decoder，输入一个随机的code，就能generate一个image。

**但是如何来衡量相似？**

上述的这些生成模型，其实有一个非常严重的弊端。比如VAE，它生成的image是希望和input越相似越好，但是model是如何来衡量这个相似呢？model会计算一个loss，采用的大多是MSE，即每一个像素上的均方差。但是loss小并不是真的表示相似。



#### GAN简述

首先大家都知道GAN有两个网络，一个是generator，一个是discriminator，从二人零和博弈中受启发，通过两个网络互相对抗来达到最好的生成效果。首先，有一个一代的generator，它能生成一些很差的图片，然后有一个一代的discriminator，它能准确的把生成的图片，和真实的图片分类，简而言之，这个discriminator就是一个二分类器，对生成的图片输出0，对真实的图片输出1。

接着，开始训练出二代的generator，它能生成稍好一点的图片，能够让一代的discriminator认为这些生成的图片是真实的图片。然后会训练出一个二代的discriminator，它能准确的识别出真实的图片，和二代generator生成的图片。以此类推，会有三代，四代以至n代的generator和discriminator，最后discriminator无法分辨生成的图片和真实图片。



#### 原理

首先我们知道真实图片集的分布$P_{data}(x)$，x是一个真实图片，可以想象成一个向量，这个向量集合的分布就是$P_{data}(x)$。我们现在有的generator生成的分布可以假设为$P_G(x;\theta)$，这是一个由$\theta$控制的分布。由此我们可以推断出生成模型中的似然：
$$
L = \prod \limits_{i=1}^m P_G(x^i;\theta)
$$
我们想要最大化这个似然，等价于让generator生成那些真实图片的概率最大。这就变成了一个最大似然估计的问题了，我们需要找到一个$\theta^*$来最大化这个似然。

