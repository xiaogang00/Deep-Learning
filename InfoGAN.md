The unsupervised learning can be described as the general problem of extracting value from unlabelled data which exists in vast quantities.

* The important point is **disentangled representation** 

The modification can be described as:

maximizing the mutual information between a fixed small subset of the GAN's noise variables and the observations.

infoGAN的出发点在于，既然GAN的自由度是由于仅仅有一个noise的变量z而无法控制导致的。最大的重点在于如何利用这个z。

开始引入一组隐变量，在这里表示为c，主要是能够在学习生成图像的时候，图像有许多可控的有含义的维度。比如笔画的粗细什么的。而剩下的不知道该怎么描述的才是z

contribution中的语句的意思就是在于拆解了先验。从而能够控制GAN的学习过程，也能够使得学习出来的模型更加具有可描述性。

在这里的原则是c应该要和$G(z,c)$ 保持高度的相关。通过互信息来描述。

如果使用潜在的变量c，其实没有监督让网络去使用c，往往就没有办法起作用。在这里定义了一个熵，度量mutual information 的程度：

$I(X,Y) = entropy(X) -entropy(X|Y) = entropy(Y) - entropy(Y|X)$ 

联系越大的话，值越小。