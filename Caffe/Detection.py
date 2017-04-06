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

f = pd.Series(df['prediction'].iloc[i], index = labels_df['name'])；、、
print 'top detection'
print f.order(ascending = False)[:5]
print ' '

# Show top predictions for second-best detection.  
f = pd.Series(df['prediction'].iloc[j], index=labels_df['name'])  
print('Second-best detection:')  
print(f.order(ascending=False)[:5])  

# Show top detection in red, second-best top detection in blue.  
im = plt.imread('examples/images/fish-bike.jpg')  
plt.imshow(im)  
currentAxis = plt.gca()  
  
det = df.iloc[i]  
coords = (det['xmin'], det['ymin']), det['xmax'] - det['xmin'], det['ymax'] - det['ymin']  
currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='r', linewidth=5))  
  
det = df.iloc[j]  
coords = (det['xmin'], det['ymin']), det['xmax'] - det['xmin'], det['ymax'] - det['ymin']  
currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='b', linewidth=5))  



