The problem of Image classification:

the task of assigning an input image one label from a fixed set of categories.

Include challenges:

* Viewpoint variation
* illumination(由于光照等现象)
* deformation（由于物体的变形）
* occlusion(由于遮挡等现象)
* Background clutter(由于背景与物体之间的界限小)
* Intraclass variation




Nearest Neighbor Classifier

最主要的就是能够设立标准，如何去衡量两幅图像之间的距离。

```
 Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/') # a magic function we provide
 # flatten out all images to be one-dimensional
 Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
 Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 307
 nn = NearestNeighbor() # create a Nearest Neighbor classifier class
 nn.train(Xtr_rows, Ytr)
 Yte_predict = nn.predict(Xte_rows)
 print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) 
```

​			
​			

##### Validation sets for hyperparameter tuning 

有些情况下，我们只是知道一个大致的需要做的方向，但是如何选择合适的算法和参数都是不确定。例如在选的时候，如何选择距离的度量也是我们需要解决的问题。

可以使用test set 来进行参数的调优

可以从中选取一个验证集来进行validation set

交叉验证：不停更换数据集和验证集，以达到tuning的效果

其中KNN的计算复杂度比较高，所以可以使用其变形：

ANN(Approximate Nearest Neighbor)



