#这个实例用于在与训练好的网络上微调flickr_style数据。
import os 
caffe_root = '../'
os.chdir(caffe_root)
import sys  
sys.path.insert(0, './python')
import caffe 
import numpy as np  
from pylab import *  

#!data/ilsvrc12/get_ilsvrc_aux.sh  
#!scripts/download_model_binary.py models/bvlc_reference_caffenet  
#!python examples/finetune_flickr_style/assemble_data.py \ 
niter = 200
# losses will also be stored in the log
train_loss = np.zeros(niter)
scratch_train_loss = np.zeros(niter)
  
caffe.set_device(0)  
caffe.set_mode_gpu()  
# We create a solver that fine-tunes from a previously trained network.  
solver = caffe.SGDSolver('models/finetune_flickr_style/solver.prototxt')  
solver.net.copy_from('models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')  
# For reference, we also create a solver that does no finetuning.  
scratch_solver = caffe.SGDSolver('models/finetune_flickr_style/solver.prototxt')  
  
# We run the solver for niter times, and record the training loss.  
for it in range(niter):
    solver.step(1)  # SGD by Caffe 
    scratch_solver.step(1)
    # store the train loss  
    train_loss[it] = solver.net.blobs['loss'].data
    scratch_train_loss[it] = scratch_solver.net.blobs['loss'].data
    if it % 10 == 0:  
        print 'iter %d, finetune_loss=%f, scratch_loss=%f' % (it, train_loss[it], scratch_train_loss[it])
print 'done'

#将较小值部分放大：
plot(np.vstack([train_loss, scratch_train_loss]).clip(0, 4).T) 

test_iters = 10
accuracy = 0
scratch_accuracy = 0
for it in arange(test_iters):
    solver.test_nets[0].forward()
    accuracy += solver.test_nets[0].blobs['accuracy'].data
    scratch_solver.test_nets[0].forward()
    scratch_accuracy += scratch_solver.test_nets[0].blobs['accuracy'].data
accuracy /= test_iters  
scratch_accuracy /= test_iters  
print 'Accuracy for fine-tuning:', accuracy  
print 'Accuracy for training from scratch:', scratch_accuracy  

