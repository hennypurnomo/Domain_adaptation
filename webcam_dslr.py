# this main code by Ssamot from https://github.com/ssamot/infoGA/blob/master/mnist_snes_example.py
# plot loss function from https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# loading data office-31 from Lyu Chaofan 1706987

import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

from dataSet import dataSet
from timeit import default_timer as timer
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from keras_helper import NNWeightHelper
from snes import SNES

def train_classifier(model, X, y):
    X_features = model.predict(X)
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X_features, y)
    y_pred = clf.predict(X_features)
    return clf, y_pred
def predict_classifier(model, clf, X):
    X_features = model.predict(X)
    return clf.predict(X_features)
	
# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 31

# use just a small sample of the train set to test
SAMPLE_SIZE = 300
# how many different sets of weights ask() should return for evaluation
POPULATION_SIZE = 10
# how many times we will loop over ask()/tell()
GENERATIONS = 30

Amazon_path = './Original_images/amazon/images'
dslr_path   = './Original_images/dslr/images'
webcam_path = './Original_images/webcam/images'

paths = [Amazon_path, dslr_path, webcam_path]
files = os.listdir(Amazon_path)
labels = {}
count  = 0
for key in files:
    a = {key : count}
    labels.update(a)
    count += 1
# print (labels)

images_path = []
webcam = dataSet()
dslr = dataSet()

for dirname in files:
    images_name = os.listdir(Amazon_path + '/' + dirname)
    for name in images_name:
        Image_Path = Amazon_path + '/' + dirname + '/' + name
        images_path.append(Image_Path)
        image_data = cv2.imread(Image_Path)
        image_data = cv2.resize(image_data, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
        image_data = image_data.reshape(img_rows, img_cols, 3)
        Amazon.upData(image_data, labels[dirname], labels)
        
Amazon.sHape()

for dirname in files:
    images_name = os.listdir(dslr_path + '/' + dirname)
    for name in images_name:
        Image_Path = dslr_path + '/' + dirname + '/' + name
        images_path.append(Image_Path)
        image_data = cv2.imread(Image_Path)
        image_data = cv2.resize(image_data, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
        image_data = image_data.reshape(img_rows, img_cols, 3)
        dslr.upData(image_data, labels[dirname], labels)
        
dslr.sHape()

#data for label classifier
x_test = dslr.data
y_test = dslr.label
x_train = webcam.data
y_train = webcam.label

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# data for domain classifier
x_domain = np.concatenate((x_train,x_test), axis = 0)
y_domain = np.concatenate((np.zeros(y_train.shape[0]), np.ones(y_test.shape[0])),axis = 0)

# Neural network architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',
                 input_shape=(28,28,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))

# this is irrelevant for what we want to achieve
model.compile(loss="mse", optimizer="adam")
print("compilation is over")
nnw = NNWeightHelper(model)
weights = nnw.get_weights()

# Neural network architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',
                 input_shape=(28,28,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))

# this is irrelevant for what we want to achieve
model.compile(loss="mse", optimizer="adam")
print("compilation is over")
nnw = NNWeightHelper(model)
weights = nnw.get_weights()

print("Total number of weights to evolve is:", weights.shape)
all_examples_indices = list(range(x_train.shape[0]))
clf, _ = train_classifier(model, x_train, y_train)
y_pred = predict_classifier(model, clf, x_test)

test_accuracy = accuracy_score(y_test, y_pred)
print('Non-trained NN Test accuracy:', test_accuracy)
# print('Test MSE:', test_mse)

snes = SNES(weights, 1, POPULATION_SIZE)
p = []
for i in range(0, GENERATIONS):
    start = timer()
    asked = snes.ask()

    # to be provided back to snes
    told = []

    # use a small number of training samples for speed purposes
    subsample_indices = np.random.choice(all_examples_indices, size=SAMPLE_SIZE, replace=False)
    # evaluate on another subset
    subsample_indices_valid = np.random.choice(all_examples_indices, size=SAMPLE_SIZE + 1, replace=False)

    # iterate over the population
    for asked_j in asked:
        # set nn weights
        nnw.set_weights(asked_j)
        # train the classifer and get back the predictions on the training data
        clf, _ = train_classifier(model, x_train[subsample_indices], y_train[subsample_indices])
        # train the domain classifier and get back the predictions on the training data
        clf2, _ = train_classifier(model, x_domain[subsample_indices], y_domain[subsample_indices])
        
        # calculate the label predictions on a different set
        y_pred = predict_classifier(model, clf, x_train[subsample_indices_valid])
        score = accuracy_score(y_train[subsample_indices_valid], y_pred)

        # calculate the domain predictions on a different set
        y_pred1 = predict_classifier(model, clf, x_domain[subsample_indices_valid])
        score1 = accuracy_score(y_domain[subsample_indices_valid], y_pred1)
        total = (score+(2*-score1))
        told.append(total)

    temp = snes.tell(asked, told)
    p.append(temp)
    end = timer()
    print("It took", end - start, "seconds to complete generation", i + 1)
    
nnw.set_weights(snes.center)

# predict on source data
clf, _ = train_classifier(model, x_train, y_train)
y_pred = predict_classifier(model, clf, x_train)
test_accuracy = accuracy_score(y_train, y_pred)
print('Test accuracy on source:', test_accuracy)

# predict on target data

y_pred1 = predict_classifier(model, clf, x_test)
test_accuracy1 = accuracy_score(y_test, y_pred1)
print('Test accuracy on target:', test_accuracy1)

plt.plot(p)
plt.title('Loss Plot Amazon - webcam')
plt.xlabel('generation')
plt.ylabel('value')
plt.savefig('Plot_amazon_webcam.png')
plt.show()


