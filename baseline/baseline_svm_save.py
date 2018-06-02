import load_data

import os
import numpy as np
import scipy.ndimage
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_baseline_data():
    data_map = load_data.get_data()
    np_images = []
    labels = []
    print len(data_map)
    i = 0
    for key in data_map:
        i += 1
        data_list = data_map[key]
        counter = 0
        for index, path, label in data_list:
            counter += 1
            if counter % 4 != 0:
                continue
            np_image = scipy.ndimage.imread(path).flatten()
            np_images.append(np_image)
            labels.append(label)

    return np_images, labels

x, y = load_baseline_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)
clf = svm.SVC()
print 'before fit'
clf.fit(x_train, y_train)
predicted = clf.predict(x_test)
print accuracy_score(y_test, predicted)
