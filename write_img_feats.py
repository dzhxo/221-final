#import h5py
import os
import time
import sys

sys.path.append('/home/bodun/221-final/models/research/slim/')

import tensorflow_hub as hub
import random
import numpy as np
import tensorflow as tf
import cv2
import scipy.ndimage
import time
from skimage import transform


#h5f = h5py.File('images_processed.h5', 'w')

module = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/1")
init = tf.global_variables_initializer()
sess = tf.Session()
x = tf.placeholder(tf.float32, [1, 128, 128, 3])
features = module(x)
sess.run(init)
print('to run')

#sess.run(init) 
i = 0

for filename in os.listdir('data'):
	i += 1
	if filename.endswith(".jpg"): 
		if i > 21191:
			print(filename)
			print('filename ' + str(i))
			#start = time.time()
			
			np_image = scipy.ndimage.imread('data/'+filename)
			img = transform.resize(np_image, [128,128])
			#transform_time = time.time()
			#print('transform: ' + str(transform_time - start))
			try_ = np.expand_dims(img,0)
			#tensor = tf.expand_dims(tf.convert_to_tensor(img, np.float32),0)
			#tensor_time = time.time()
			#print('tensor: ' + str(tensor_time-transform_time))

			output = sess.run(features, feed_dict={x: try_})#tensor.eval(session=sess)})	
			#output_time = time.time()
			#print('output: ' + str(output_time-tensor_time))

			#with open('data_featurized/'+filename[:-4]+'.txt', 'w') as f:
			#	f.write(np.array_str(output))
			np.savetxt('data_featurized/'+filename[:-4]+'.txt', output, delimiter=',')
			#h5f.create_dataset(filename, data=output)
			#write_time = time.time()
			#print('write: ' + str(write_time-output_time))
			#print('\n')
	else:
		continue
