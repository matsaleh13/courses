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

NOTE: This is the same as lesson2.py, but with some (hopefully) improvements.

1. Train last layer with fewer epochs (e.g. 2).
2. Train last layer with smaller data set (e.g. 25% of training set).
3. Use LR=0.01 on last layer (instead of 0.001)
4. Re-train additional Dense layers earlier than last layer (after training last layer).
5. Use more epochs on remaining layers (e.g. 4).
6. Use LR=0.001 on earlier layers.

'''

import os
import os.path

import utils
from vgg16 import Vgg16

import bcolz

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

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
# Utility Functions
#


def onehot(x):
    # Returns two-column matrix with one row for each class.
    return np.array(OneHotEncoder().fit_transform(x.reshape(-1, 1)).todense())


#
# Data Setup
#
print
print 'Setting up folders...'
print

INPUT_PATH = '/input/'
# INPUT_PATH = '/input/sample/'
print 'Input folder: %s' % INPUT_PATH

TRAIN_PATH = os.path.join(INPUT_PATH, 'train')
print 'Training data: %s' % TRAIN_PATH

VALID_PATH = os.path.join(INPUT_PATH, 'valid')
print 'Validation data: %s' % VALID_PATH

TEST_PATH = os.path.join(INPUT_PATH, 'test')
print 'Test data: %s' % TEST_PATH
print

OUTPUT_PATH = '/output/'
# OUTPUT_PATH = '/output/sample'
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

#
# FloydHub
#
BATCH_SIZE = 64
NUM_EPOCHS = 5
LR = 0.001
LL_DATA_SIZE = 0.25    # Fraction of train/valid set to use for last layer training.
LL_NUM_EPOCHS = 3      # Number of epochs to train last layer only.
LL_LR = 0.01

#
# Sample Set
#
# BATCH_SIZE = 10
# NUM_EPOCHS = 2
# LR = 0.001
# LL_DATA_SIZE = 0.50    # Fraction of train/valid set to use for last layer training.
# LL_NUM_EPOCHS = 2      # Number of epochs to train last layer only.
# LL_LR = 0.001


#
# NOTE: Loading and use of data structures is pretty fucked up here.
# Some things require getting data from generators, others require NumPy arrays.
# In the end we use both, and sometimes re-load the data from disk and/or re-transform
# it more than once.
#


print 'Loading raw training data from %s...' % TRAIN_PATH
TRAIN_BATCHES = utils.get_batches(TRAIN_PATH, shuffle=False, batch_size=1)
print '\tshape: %s' % (TRAIN_BATCHES.image_shape,)
print 'Loading array from generator...'
TRAIN_ARRAY = utils.get_data(TRAIN_PATH)
print '\tshape: %s' % (TRAIN_ARRAY.shape,)

print

TRAIN_DATA = os.path.join(MODEL_PATH, 'train_data.bc')
print 'Saving processed training data to %s...' % TRAIN_DATA
utils.save_array(TRAIN_DATA, TRAIN_ARRAY)

print 'Loading raw validation data from %s...' % VALID_PATH
VALID_BATCHES = utils.get_batches(VALID_PATH, shuffle=False, batch_size=1)
print '\tshape: %s' % (VALID_BATCHES.image_shape,)
print 'Loading array from generator...'
VALID_ARRAY = utils.get_data(VALID_PATH)
print '\tshape: %s' % (VALID_ARRAY.shape,)

print

VALID_DATA = os.path.join(MODEL_PATH, 'valid_data.bc')
print 'Saving processed validation data to %s...' % VALID_DATA
utils.save_array(VALID_DATA, VALID_ARRAY)

print 'Getting the true labels for every image...'

TRAIN_CLASSES = TRAIN_BATCHES.classes
TRAIN_LABELS = onehot(TRAIN_CLASSES)
# print '\tTraining labels look like this: \n%s\n...\n%s' % (TRAIN_LABELS[:5], TRAIN_LABELS[-5:])
# print

VALID_CLASSES = VALID_BATCHES.classes
VALID_LABELS = onehot(VALID_CLASSES)
# print '\tValidation labels look like this: \n%s\n...\n%s' % (VALID_LABELS[:5], VALID_LABELS[-5:])
# print


print 'Preparing image data generators...'
gen = image.ImageDataGenerator()
# NOTE: TRAIN_BATCHES and VALID_BATCHES here are generators,
# but still not quite the same as above.
TRAIN_BATCHES = gen.flow(TRAIN_ARRAY, TRAIN_LABELS,
                         batch_size=BATCH_SIZE, shuffle=True)
VALID_BATCHES = gen.flow(VALID_ARRAY, VALID_LABELS,
                         batch_size=BATCH_SIZE, shuffle=False)


print 'Replacing last layer of model...'
vgg = Vgg16()
vgg.model.pop()

for layer in vgg.model.layers: layer.trainable=False
vgg.model.add(Dense(2, activation='softmax'))

OPTIMIZER = RMSprop(lr=LL_LR)   # Larger LR for this run
vgg.model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

print 'Fitting last layer of model using a subset of samples...'
vgg.model.fit_generator(TRAIN_BATCHES, samples_per_epoch=TRAIN_BATCHES.n * LL_DATA_SIZE, nb_epoch=LL_NUM_EPOCHS,
                        validation_data=VALID_BATCHES, nb_val_samples=VALID_BATCHES.n * LL_DATA_SIZE)

print 'Saving model weights...'
vgg.model.save_weights(os.path.join(MODEL_PATH, 'finetune_1_ll.h5'))

print 'Confusion matrix after last layer retraining'
PREDS = vgg.model.predict_classes(VALID_ARRAY, batch_size=BATCH_SIZE)
PROBS = vgg.model.predict_proba(VALID_ARRAY, batch_size=BATCH_SIZE)[:, 0]

CM = confusion_matrix(VALID_CLASSES, PREDS)
print CM

print 'Re-training other Dense layers...'
LAYERS = vgg.model.layers
FIRST_DENSE_IDX = [index for index, layer in enumerate(LAYERS) if type(layer) is Dense][0]

for layer in LAYERS[FIRST_DENSE_IDX:]:
    layer.trainable = True  # unlock what we locked earlier

# Set Learning Rate smaller for this pass.
K.set_value(OPTIMIZER.lr, LR)

# full data set now
vgg.model.fit_generator(TRAIN_BATCHES, samples_per_epoch=TRAIN_BATCHES.n, nb_epoch=NUM_EPOCHS,
                        validation_data=VALID_BATCHES, nb_val_samples=VALID_BATCHES.n)

print 'Saving model weights...'
vgg.model.save_weights(os.path.join(MODEL_PATH, 'finetune_2_full.h5'))


print 'Evaluating model with validation data...'
TEST_LOSS = vgg.model.evaluate(VALID_ARRAY, VALID_LABELS)
print 'TEST_LOSS: %s' % (TEST_LOSS,)

print 'Confusion matrix after full retraining'
PREDS = vgg.model.predict_classes(VALID_ARRAY, batch_size=BATCH_SIZE)
PROBS = vgg.model.predict_proba(VALID_ARRAY, batch_size=BATCH_SIZE)[:,0]

CM = confusion_matrix(VALID_CLASSES, PREDS)
print CM

print ('Predicting labels for test data set...')
TEST_BATCHES = utils.get_batches(TEST_PATH, shuffle=False, batch_size=BATCH_SIZE)
TEST_PREDS = vgg.model.predict_generator(TEST_BATCHES, TEST_BATCHES.nb_sample)
TEST_FILENAMES = TEST_BATCHES.filenames

#Save our test results arrays so we can use them again later
print('Saving raw prediction results.')
utils.save_array(os.path.join(MODEL_PATH, 'test_preds.dat'), TEST_PREDS)
utils.save_array(os.path.join(MODEL_PATH, 'filenames.dat'), TEST_FILENAMES)

# Grab the dog prediction column
IS_DOG = TEST_PREDS[:, 1]

# To play it safe, we use a sneaky trick to round down our edge predictions
# Swap all ones with .95 and all zeros with .05
IS_DOG = IS_DOG.clip(min=0.05, max=0.95)

# Extract imageIds from the filenames in our test/unknown directory
IDS = np.array([int(os.path.splitext(os.path.basename(f))[0])
                for f in TEST_FILENAMES])

# Combine the ids and IS_DOG columns into a single 2-column array.
SUBMIT = np.stack([IDS, IS_DOG], axis=1)

print('Formatting and saving data for Kaggle submission.')
np.savetxt(os.path.join(RESULTS_PATH, 'kaggle_submission.csv'), SUBMIT,
           fmt='%d,%.5f', header='id,label', comments='')

print('Model training and prediction complete.')




