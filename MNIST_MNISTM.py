# this main code by Ssamot from https://github.com/ssamot/infoGA/blob/master/mnist_snes_example.py
# plot loss function from https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# converting MNIST into 3 dimentional from Rabia Yasa Kostas (1700421)/

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from keras_helper import NNWeightHelper
from keras.utils import np_utils
from snes import SNES

# use just a small sample of the train set to test
SAMPLE_SIZE = 400
# how many different sets of weights ask() should return for evaluation
POPULATION_SIZE = 10
# how many times we will loop over ask()/tell()
GENERATIONS = 30

def train_classifier(model, X, y):
    X_features = model.predict(X)
    clf = RandomForestClassifier(n_estimators = 10)
    clf.fit(X_features, y)
    y_pred = clf.predict(X_features)
    return clf, y_pred


def predict_classifier(model, clf, X):
    X_features = model.predict(X)
    return clf.predict(X_features)
	
# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10

# Load MNIST dataset from keras for source domain
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype(np.uint8) * 255
x_train = np.concatenate([x_train, x_train, x_train], 3)

# Load MNIST-M dataset for target domain
mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
mnistm_train = mnistm['train']
mnistm_test = mnistm['test']
mnistm_valid = mnistm['valid']
x_test=mnistm_test

# create part data for domain classifier, combination from source and target data
x_domain=np.concatenate((x_train,x_test), axis =0)
y_domain = np.concatenate((np.zeros(y_train.shape[0]), np.ones(y_test.shape[0])),axis=0)

#data normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# neural network architecture
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
log = []
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
        # train the label classifer and get back the predictions on the training data
        clf, _ = train_classifier(model, x_train[subsample_indices], y_train[subsample_indices])
        # train the domain classifier and get back the predictions on the training data
        clf2, _ = train_classifier(model, x_domain[subsample_indices], y_domain[subsample_indices])

        # calculate the label predictions on a different set
        y_pred = predict_classifier(model, clf, x_train[subsample_indices_valid])
        score = accuracy_score(y_train[subsample_indices_valid], y_pred)

        # calculate the domain predictions on a different set
        y_pred2 = predict_classifier(model, clf2, x_domain[subsample_indices_valid])
        score2 = accuracy_score(y_domain[subsample_indices_valid], y_pred2)
        
        # weighted score to give back to snes
        total = (score+(2*-score2))
        told.append(total)

    temp = snes.tell(asked, told)
    log.append(temp)
    end = timer()
    print("It took", end - start, "seconds to complete generation", i + 1)
    
nnw.set_weights(snes.center)

# predict on target data
clf, _ = train_classifier(model, x_train, y_train)
y_pred = predict_classifier(model, clf, x_test)
test_accuracy = accuracy_score(y_test, y_pred)
print('Test accuracy on target:', test_accuracy)

# print loss plot
plt.plot(log)
plt.title('Loss Plot MNIST and MNIST-M')
plt.xlabel('generation')
plt.ylabel('value')
plt.savefig('Plot_mnist_mnistm.png')
plt.show()