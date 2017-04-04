#这个例子讲述如何编辑caffe的模型参数以满足特定的需要，所有的网络数据，残差，参数都在pycaffe中
import numpy as np   
import matplotlib.pyplot as plt   
import Image 
caffe_root = '../'
import sys   
sys.path.insert(0, caffe_root + 'python')  
import caffe   
plt.rcParams['figure.figsize'] = (10, 10)  
plt.rcParams['image.interpolation'] = 'nearest'  
plt.rcParams['image.cmap'] = 'gray'  
  
model_file = caffe_root + 'examples/net_surgery/conv.prototxt'  
image_file = caffe_root + 'examples/images/cat_gray.jpg'  

caffe.set_mode_cpu()  
net = caffe.Net(model_file, caffe.TEST)  
print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))  
# load image and prepare as a single input batch for Caffe  
im = np.array(Image.open(image_file))
plt.title("original image") 
plt.imshow(im)  
plt.axis('off')  
  
im_input=im[np.newaxis, np.newaxis,:,:]  
net.blobs['data'].reshape(*im_input.shape)  
net.blobs['data'].data[...] = im_input  

def show_filter(net):
    net.forward()  
    plt.figure()  
    filt_min, filt_max = net.blobs['conv'].data.min(), net.blobs['conv'].data.max()  
    for i in range(3):  
        plt.subplot(1,4,i+2)  
        plt.title("filter #{} output".format(i))  
        plt.imshow(net.blobs['conv'].data[0, i], vmin=filt_min, vmax=filt_max)  
        plt.tight_layout()
        plt.axis('off')
# filter the image with initial
show_filters(net)  

conv0 = net.blobs['conv'].data[0, 0]  
print("pre-surgery output mean {:.2f}".format(conv0.mean()))  
# set first filter bias to 10  
net.params['conv'][1].data[0] = 1.   #['conv'][1]在这里指的是偏置
net.forward()  
print("post-surgery output mean {:.2f}".format(conv0.mean())) 

# 导入全卷积网去移植的参数  
net_full_conv = caffe.Net('examples/net_surgery/bvlc_caffenet_full_conv.prototxt',   
                          'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',  
                          caffe.TEST)  
#网络的参数必须要prototxt 和caffemodel

params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8-conv']  
# conv_params = {name: (weights, biases)}  
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}  
#0，1分别代表的是weight 和  bias

for conv in params_full_conv:  
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)  

for pr, pr_conv in zip(params, params_full_conv):  
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays  
    conv_params[pr_conv][1][...] = fc_params[pr][1]

#存储
net_full_conv.save('examples/net_surgery/bvlc_caffenet_full_conv.caffemodel')  

im = caffe.io.load_image('examples/images/cat.jpg')  
transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})  
transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))  
transformer.set_transpose('data', (2,0,1))  
transformer.set_channel_swap('data', (2,1,0))  
transformer.set_raw_scale('data', 255.0)  
# make classification map by forward and print prediction indices at each location  
out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))  
print out['prob'][0].argmax(axis=0)  
# show net input and confidence map (probability of the top prediction at each location)  
plt.subplot(1, 2, 1)  
plt.imshow(transformer.deprocess('data', net_full_conv.blobs['data'].data[0]))  
plt.subplot(1, 2, 2)  
plt.imshow(out['prob'][0,281])  
