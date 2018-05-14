import os
import numpy as np
import scipy.ndimage
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = "sequence"

def load_data():
  image_files = os.listdir(DATA_PATH)
  sorted_files = sorted(image_files)
  np_images = []
  for file_name in sorted_files:
    file_path = os.path.join(DATA_PATH, file_name)
    np_image = scipy.ndimage.imread(file_path).flatten()
    np_images.append(np_image)
  return np_images, np.asarray([4] * len(np_images))

x, y = load_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
clf = svm.SVC()
clf.fit(x_train, y_train)
predicted = clf.predict(x_test)
print accuracy_score(y_test, predicted)
