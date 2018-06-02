import load_data

import os
import numpy as np
import scipy.ndimage
from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def load_baseline_data():
    data_map = load_data.get_data('../data')
    np_images = []
    labels = []
    print len(data_map)
    i = 0
    for key in data_map:
        i += 1
        data_list = data_map[key]
        #print(data_list)
        counter = 0
        for index, path, label in data_list:
            counter += 1
            if label == False:
                if counter % 2 != 0:
                    continue
            elif label == True:
                if counter % 8 != 0:
                    continue
            np_image = scipy.ndimage.imread(path).flatten()
            np_images.append(np_image)
            labels.append(label)

    return np_images, labels

print('start')
x, y = load_baseline_data()
print('finished loading data')
print float(sum(y))/len(y)
#x_scaled = preprocessing.scale(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)
x_train = preprocessing.scale(x_train)
x_test = preprocessing.scale(x_test)
print('finished split')
clf = svm.SVC()
print('before fit')
clf.fit(x_train, y_train)
predicted = clf.predict(x_test)
print("accuracy of standard/non-tuned model")
print accuracy_score(y_test, predicted)

print('tuning hyperparameters')
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-2], 'C': [.01, .1, 1, 10, 100]}]
tuned_parameters = [{'kernel': ['rbf'], 'gamma':[1e-4, 1e-2], 'C':[.1, 1]}]
clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=2)
clf.fit(x_train, y_train)

print(clf.best_params_)
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
