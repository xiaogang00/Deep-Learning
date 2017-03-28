* 网络模型，caffe的网络模型定义了网络的每一层的行为，其文件为*.prototxt

  可以定义输入的训练数据层，测试数据层，池化层，RELU激活函数等

* 参数配置： 在参数配置文件*.solver.prototxt定义了网络模型训练过程中需要设置的参数。训练出来的模型的输出文件为 *.caffemodel

* Google Protobuf结构化数据，其主要用于RPC系统和持续数据存储系统，Protocol Buffers是一种轻便高效的结构化数据存储格式。Caffe的网络模型利用Protocol Buffer语言定义后存放在caffe.proto文件中

* Caffe数据库

  * 支持三种数据库输入：LevelDB，LMDB，HDF5

* Caffe Net表示的是一个完整的CNN模型，是由不同的layers组成的有向无环图。一个典型的Net从数据层开始，从外部载入数据，最后在Loss层计算目标任务。

  Net::Init()对模型进行初始化，实现两个操作：创建blobs和layers来搭建整个DAG网络图，以及调用layers的SetUp函数。

* Caffe Blob，用来存储，交换和处理网络中正向和反向迭代的数据和导数信息。Blob的维度是：
  $$
  图像数量N \times 通道数K \times 图像高度H \times 图像宽度W
  $$

* Caffe Layer是Caffe模型的本质内容和执行计算的基本单元。可以进行多种运算，如convolve,pool,innerproduct等

  * Data Layer:位于网络的最底层，数据可以从高效率的数据库中读取，也可以直接从内存中读取。

    * 数据库，层类型：Data，参数:source  batch_size  rank_skip  backend
    * 内存数据，层类型：MemoryData，参数：batch_size, channels, height, width
    * HDF5数据，层类型：HDF5Data，参数：source，batch_size
    * 图像数据 Images
    * 窗口 Windows
    * Dummy

  * Convolution Layers

  * Pooling Layers

  * InnerProduct Layers  全连接层

  * ReLu Layers 

  * Sigmoid Layers

  * LRN Layers  局部响应值归一化

  * Dropout Layers

  * SoftmaxWithLoss Layers  在这里满足：
    $$
    softmax_loss = softmax + loss\quad regression
    $$

  * Softmax Layers

  * Accuracy Layers

* Solver 方法

  * SGD
  * AdaDelta  鲁棒的学习率方法
  * AdaGrad  自适应性梯度下降方法
  * Adam   AdaGrad的一种泛化形式
  * NAG  加速梯度下降
  * RMSprop  基于梯度的优化方法

  ​
