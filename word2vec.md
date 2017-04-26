* 预备知识

  sigmoid函数：$\sigma(x) = \frac{1}{1+e^{-x}}$，可以使用到逻辑回归中，

  huffman编码



* 统计语言模型

  统计语言模型是用来计算一个句子的概率的模型。在语音识别系统中，对于给定的语音段Voice，需要找到一个使得概率$p(Text|Voice)$ 最大的文本段Text，利用Bayes公式，有：

  $p(Text|Voice) = \frac{p(Voice|Text) \cdot p(Text)}{p(Voice)}$ 

  $p(Voice|Text)$ 为声学模型，$p(Text)$ 为语言模型

  而句子的概率指的是其单词的联合概率$p(W)=p(w_1^T)=p(w_1,w_2,\dots ,w_T)$ 

  $p(w_1^T) = P(w_1)  \cdot p(w_2|w_1)\cdot p(w_3|w_1^2) \dots p(w_T|w_1^{T-1})$

  ​

* n-gram模型

  我们首先需要计算的就是$p(w_k|w_1^{k-1})$ ，在语料库足够大的时候，可以近似地表示为：

  $p(w_k|w_1^{k-1}) =\frac{count(w_1^k )}{count(w_1^{k-1})}$

  n-gram在这里使用了一个n-1阶的Markov假设，$p(w_k|w_1^{k-1}) = p(w_k|w_{k-n+1}^{k-1})$ 

  n-gram的主要工作是语料库中统计各种词串出现的次数以及平滑操作。

  $p(w|Context(w)) = F(w,Context(w),\theta)$

* 神经概率语言模型

  在这里使用了重要的词向量工具。词向量的形式，包括有one-hot。还有是Distributed representation，主要通过将其映射成短向量。把词的信息分布到各个分量中去。

* Word2vec中用到的两个重要模型，CBOW(Continuous Bag-of-Words Model)和Skip-gram(Continuous Skip-gram Model)。两个模型都是神经网络模型， 包含输入层，投影层和输出层。

* ​

  ​

  ​

  ​