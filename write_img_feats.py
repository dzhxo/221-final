import h5py
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
from skimage import transform

module = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/1")
h5f = h5py.File('images_processed.h5', 'w')

for filename in os.listdir('../data'):
    if filename.endswith(".jpg"): 
    	np_image = scipy.ndimage.imread(file_path)
    	img = transform.resize(np_image, [128,128])
    	features = module(tf.convert_to_tensor(img, np.float32))
    	h5f.create_dataset(filename, data=features)
    else:
        continue

h5f.close()