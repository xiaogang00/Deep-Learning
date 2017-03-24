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
​		
​	