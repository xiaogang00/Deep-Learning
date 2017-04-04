import numpy as np 
caffe_root = "../"
val_dir = '/dataset/imagenet/val'
model_name = 'caffenet_train_iter_450000.caffemodel'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe 
import os 
caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt', 
                caffe_root + 'models/bvlc_reference_caffenet/' + model_name, caffe.TEST)
transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.py').mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))
net.blobs['data'].reshape(50, 3, 227, 227)
fh = open(alexnetlog.txt', 'w')
batchsize = net.blobs['data'].shape[0]
#blobs 是存储，交换，处理网络中正向和反向迭代的数据和导数信息
for dirpath, dirnames, filenames inos.walk(val_dir):
    sortedfiles = sorted(filrnames)
n = len(sortedfiles)
nbatch = (n + batchsize -1)
for i in range(nbatch):
    idx = np.arange(i*batchsize, min(n, (i+1)*batchsize))
    for tdx in idx:
        filename = sortedfiles[tdx]
        indexofdata = tdx % batchsize
        net.blobs['data'].data[indexofdata] = transformer.preprocess('data', caffe.io.load_image
                                                                   (os.path.join(dirpath, filename)))
        out = net.forward()
for j in range(batchsize):
    output_pred = out['prob'][j].argsort()[-1:-6:-1]
    outlist = output_pred.tolist()
    templist = [str(i) for i in outlist]
    fh.write(' '.join(templist))
    fh.write('\n')
fh.close()