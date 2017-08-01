# lesson2.py
# Stand-alone script to run the code from the lesson2-matsaleh.ipynb Jupyter Notebook.

'''
Lesson 2 Assignment Plan:

1.	Train a linear model using the ImageNet predictions to classify images into dogs or cats.
    *	Get the true labels for every image
    *	Get the 1,000 imagenet category predictions for every image
    *	Fit the linear model using these predictions as features.
2.	Retrain the VGG model with the new linear layer.
    *	Pop off the existing Dense last layer.
    *	Add the new Dense layer as the new last layer.
    *	Fit the updated model.
    *	Save the weights from the training in the models/ folder
3.	Run predictions on the Kaggle dogs and cats test data using the retrained model.
'''

import os
import os.path

import utils
from vgg16 import Vgg16

import bcolz

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image

#
# Data Setup
#
print
print 'Setting up folders...'
print

#INPUT_PATH = '/input/'
INPUT_PATH = '/input/sample/'
print 'Input folder: %s' % INPUT_PATH

TRAIN_PATH = os.path.join(INPUT_PATH, 'train')
print 'Training data: %s' % TRAIN_PATH

VALID_PATH = os.path.join(INPUT_PATH, 'valid')
print 'Validation data: %s' % VALID_PATH

TEST_PATH = os.path.join(INPUT_PATH, 'test')
print 'Test data: %s' % TEST_PATH
print

#OUTPUT_PATH = '/output/'
OUTPUT_PATH = '/output/sample'
print 'Output folder: %s' % OUTPUT_PATH

MODEL_PATH = os.path.join(OUTPUT_PATH, 'models')
if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)
print 'Model data: %s' % MODEL_PATH

RESULTS_PATH = os.path.join(OUTPUT_PATH, 'results')
if not os.path.exists(RESULTS_PATH): os.makedirs(RESULTS_PATH)
print 'Results: %s' % RESULTS_PATH
print

#
# Constants
#
#BATCH_SIZE = 64
BATCH_SIZE = 10

print 'Loading raw training data from %s...' % TRAIN_PATH
TRAIN_BATCHES = utils.get_batches(TRAIN_PATH, shuffle=False, batch_size=1)
print '\tshape: %s' % (TRAIN_BATCHES.image_shape,)

# Can't pickle a generator
# TRAIN_DATA = os.path.join(MODEL_PATH, 'train_data.bc')
# print 'Saving processed training data to %s...' % TRAIN_DATA
# utils.save_array(TRAIN_DATA, TRAIN_BATCHES)

print 'Loading raw validation data from %s...' % VALID_PATH
VALID_BATCHES = utils.get_batches(VALID_PATH, shuffle=False, batch_size=1)
print '\tshape: %s' % (VALID_BATCHES.image_shape,)

# Can't pickle a generator
# VALID_DATA = os.path.join(MODEL_PATH, 'valid_data.bc')
# print 'Saving processed validation data to %s...' % VALID_DATA
# utils.save_array(VALID_DATA, VALID_BATCHES)

print 'Getting the true labels for every image...'

def onehot(x):
    # Returns two-column matrix with one row for each class.
    return np.array(OneHotEncoder().fit_transform(x.reshape(-1, 1)).todense())

TRAIN_CLASSES = TRAIN_BATCHES.classes
print '\tTraining classes look like this: \n%s ... %s' % (TRAIN_CLASSES[:5], TRAIN_CLASSES[-5:])
print
TRAIN_LABELS = onehot(TRAIN_CLASSES)
print '\tTraining labels look like this: \n%s\n...\n%s' % (TRAIN_LABELS[:5], TRAIN_LABELS[-5:])
print

VALID_CLASSES = VALID_BATCHES.classes
print '\tValidation classes look like this: \n%s ... %s' % (VALID_CLASSES[:5], VALID_CLASSES[-5:])
print

VALID_LABELS = onehot(VALID_CLASSES)
print '\tValidation labels look like this: \n%s\n...\n%s' % (VALID_LABELS[:5], VALID_LABELS[-5:])

print
print 'Getting the ImageNet category predictions for every image.'

vgg = Vgg16()

print 'Getting training set predictions...'
TRAIN_FEATURES = vgg.model.predict_generator(
    TRAIN_BATCHES, TRAIN_BATCHES.nb_sample)
utils.save_array(os.path.join(
    MODEL_PATH, 'train_lastlayer_features.bc'), TRAIN_FEATURES)
print 'Training features shape: %s' % (TRAIN_FEATURES.shape,)

print 'Getting validation set predictions...'
VALID_FEATURES = vgg.model.predict_generator(
    VALID_BATCHES, VALID_BATCHES.nb_sample)
utils.save_array(os.path.join(
    MODEL_PATH, 'valid_lastlayer_features.bc'), VALID_FEATURES)
print 'Validation features shape: %s' % (VALID_FEATURES.shape,)

print 'Defining the linear model...'
lm = Sequential([Dense(2, activation='softmax', input_shape=(1000,))])
lm.compile(optimizer=RMSprop(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

print 'Fitting the linear model using the new features...'
lm.fit_generator(TRAIN_FEATURES, TRAIN_LABELS, nb_epoch=3, batch_size=BATCH_SIZE,
    validation_data=(VALID_FEATURES, VALID_LABELS), verbose=1)

print 'Our new model looks like this: \%s' % (lm.summary(),)
