import  os   
caffe_root = '../'
os.chdir(caffe_root)
import sys   
sys.path.insert(0, './python') 

#pretrained_model = models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel
#model_def = models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt 

import numpy as np    
import pandas as pd  
import matplotlib.pyplot as plt    

df = pd.read_hdf('_temp/det_output.h5','df')
print df.shape
print df.iloc[0]


i = predictions_df['person'].argmax()
j = predictions_df['bicycle'].argmax()

f = pd.Series(df['prediction'].iloc[i], index = labels_df['name'])
print 'top detection'
print f.order(ascending = False)[:5]
print ' '


