Backpropagation 是用来在这里计算梯度的一种方法

其中比较重要的是一种链式法则的计算

This extra multiplication (for each input) due to the chain rule can turn a single and
relatively useless gate into a cog in a complex circuit such as an entire neural network

可以通过梯度向后传播需要的改变量

Backpropagation can thus be thought of as gates communicating to each other (through the
gradient signal) whether they want their outputs to increase or decrease (and how strongly), so
as to make the final output value higher.



主要的例子：Modularity: Sigmoid example

Backprop in practice: Staged computation

