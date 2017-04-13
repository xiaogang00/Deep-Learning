* Architecture Overview:

​        the layers of a ConvNet have neurons arranged in 3 dimensions: width, height, depth. 

* Layers used to build ConvNets:

  We use three main types of layers to build ConvNet architectures: Convolutional Layer,
  Pooling Layer, and Fully-Connected Layer (exactly as seen in regular Neural Networks). We will
  stack these layers to form a full ConvNet architecture.

* Parameter Sharing:

  Parameter sharing scheme is used in Convolutional Layers to control the
  number of parameters.

* Backpropagation.:

  The backward pass for a convolution operation (for both the data and the
  weights) is also a convolution (but with spatially-¡ipped lters). This is easy to derive in the 1-
  dimensional case with a toy example (not expanded on for now)

  ​

  ​

  ​