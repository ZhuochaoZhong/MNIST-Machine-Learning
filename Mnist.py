#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:36:17 2018

@author: zhuochaozhong
"""

# Using Keras (on top of TensorFlow) built-in Mnist Dataset
from keras.datasets import mnist

# AI Library for Deep Neural Network
import tensorflow as tf

# Used for Confusion Matrix and generate reports
from sklearn.metrics import confusion_matrix, classification_report

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression

# Support Vector Classification Model
from sklearn.svm import SVC

# Multi-layer Perceptron Model
from sklearn.neural_network import MLPClassifier

# Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Data Visualization Library Based on matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Using keras for 2D Convolutional Neural Networks(CNNS)
# It is the current state-of-art architecture for image classification task
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_pixels = X_train.shape[1] * X_train.shape[2]

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
x_train, x_test = X_train/255, X_test/255

y_train = y_train.astype("int32")
y_test = y_test.astype("int32")

plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(X_train[0:5], y_train[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)

# Logistic Regression (92.02%)
logit = LogisticRegression()
logit.fit(x_train, y_train)
logit_predict = logit.predict(x_test)
logit_score = logit.score(X_test, y_test)
print('Test accuracy:', logit_score)

logit_cm = confusion_matrix(y_test, logit_predict)
logit_cm = logit_cm.astype('float') / logit_cm.sum(axis=1)[:, np.newaxis]
print(classification_report(y_test, logit_predict))

plt.figure(figsize=(9,9))
sns.heatmap(logit_cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Test accuracy: {:.3f}'.format(logit_score) 
plt.title(all_sample_title, size = 15);



# Random Forest (97.22%)
forest = RandomForestClassifier(n_estimators=1000)
forest.fit(x_train, y_train)
forest_predict = forest.predict(x_test)
forest_score = forest.score(x_test, y_test)
print("Test accuracy:", forest_score)

forest_cm = confusion_matrix(y_test, forest_predict)
forest_cm = forest_cm.astype('float') / forest_cm.sum(axis=1)[:, np.newaxis]
print(classification_report(y_test, forest_predict))

plt.figure(figsize=(9,9))
sns.heatmap(forest_cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Test accuracy: {:.3f}'.format(forest_score) 
plt.title(all_sample_title, size = 15);


# SVM (97.87%)
clf_svm = SVC(gamma=0.1, kernel='poly')
clf_svm.fit(x_train, y_train)
clf_svm_predict = clf_svm.predict(x_test)
clf_svm_score = clf_svm.score(x_test, y_test)
print("Test accuracy:", clf_svm_score)

clf_svm_cm = confusion_matrix(y_test, clf_svm_predict)
clf_svm_cm = clf_svm_cm.astype('float') / clf_svm_cm.sum(axis=1)[:, np.newaxis]
print(classification_report(y_test, clf_svm_predict))

plt.figure(figsize=(9,9))
sns.heatmap(clf_svm_cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Test accuracy: {:.3f}'.format(clf_svm_score) 
plt.title(all_sample_title, size = 15);


# Multi-Perceptron (98.13%)
mlp = MLPClassifier(hidden_layer_sizes=(512, 256))
mlp.fit(x_train, y_train)
mlp_predict = mlp.predict(x_test)
mlp_score = mlp.score(x_test, y_test)
print('Test accuracy:', mlp_score)

mlp_cm = confusion_matrix(y_test, mlp_predict)
mlp_cm = mlp_cm.astype('float') / mlp_cm.sum(axis=1)[:, np.newaxis]
print(classification_report(y_test, mlp_predict))

plt.figure(figsize=(9,9))
sns.heatmap(mlp_cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Test accuracy: {:.3f}'.format(mlp_score) 
plt.title(all_sample_title, size = 15);


# Tensorflow DNNClassifier
# Test Accuracy 96.72%(1000 steps) 97.99%(10000 steps) 98.34%(100000 steps, with hidden units [512, 256])

feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]

train_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_train}, y=y_train, batch_size=128, num_epochs=None, shuffle=True)

classifier = tf.estimator.DNNClassifier(hidden_units= [512, 256], n_classes=10, 
                                        feature_columns=feature_columns)

classifier.train(input_fn=train_fn, steps=1000) 


pred_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_test}, y=y_test, batch_size=len(x_test), shuffle=False)

prediction = list(classifier.predict(input_fn = pred_fn))

final_preds = []
for pred in prediction:
    final_preds.append(pred['class_ids'][0])
    
accuracy = classifier.evaluate(input_fn=pred_fn)["accuracy"]
print("Test Accuracy: {:.3f}".format(accuracy*100))

dnn_cm = confusion_matrix(y_test, final_preds)
dnn_cm = dnn_cm.astype('float') / dnn_cm.sum(axis=1)[:, np.newaxis]
print(classification_report(y_test, final_preds))

plt.figure(figsize=(9,9))
sns.heatmap(dnn_cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Test accuracy: {:.3f}'.format(accuracy) 
plt.title(all_sample_title, size = 15);


# Keras 2D Convolutional Neural Networks (CNN)
# Test Accuracy 99.08%
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()
#convolutional layer with rectified linear unit activation
#32 convolution filters used each of size 3x3
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

#64 convolution filters used each of size 3x3
model.add(Conv2D(64, (3, 3), activation='relu'))

#choose the best features via pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

#randomly turn neurons on and off to improve convergence
model.add(Dropout(0.25))

#flatten since too many dimensions, we only want a classification output
model.add(Flatten())

#fully connected to get all relevant data
model.add(Dense(128, activation='relu'))

#one more dropout for convergence
model.add(Dropout(0.5))

#output a softmax to squash the matrix into output probabilities
model.add(Dense(10, activation='softmax'))


#Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
#categorical ce since we have multiple classes (10) 
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
