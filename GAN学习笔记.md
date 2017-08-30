#### Generation

什么是生成（generation）？就是模型通过学习一些数据，然后生成类似的数据。让机器看一些动物图片，然后自己来产生动物的图片，这就是生成。

以前就有很多可以用来生成的技术了，比如auto-encoder（自编码器）。你训练一个encoder，把input转换成code，然后训练一个decoder，把code转换成一个image，然后计算得到的image和input之间的MSE（mean square error），训练完这个model之后，取出后半部分NN Decoder，输入一个随机的code，就能generate一个image。

**但是如何来衡量相似？**

上述的这些生成模型，其实有一个非常严重的弊端。比如VAE，它生成的image是希望和input越相似越好，但是model是如何来衡量这个相似呢？model会计算一个loss，采用的大多是MSE，即每一个像素上的均方差。但是loss小并不是真的表示相似。



#### GAN简述

