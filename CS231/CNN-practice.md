* Data Augmentation

  Random mix/combinations of :

  * translation
  * rotation
  * stretching 
  * shearing
  * lens distortions

* 在训练CNN的时候，一般需要前几层都是more generic,可以直接套用其他人的网络，而后面的几层一般都需要时more specific.

* The power of small filters->less compute, more nonlinearity

* GoogLeNet 的思想， 一个很重要的就是用小的filter去替代大的，减少计算的复杂度

* Can factor N x N convolutions into 1 x N and N x 1