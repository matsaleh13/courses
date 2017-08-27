# lesson2.py
# Stand-alone script to run the code from the lesson2-matsaleh.ipynb Jupyter Notebook.

'''
Lesson 3 Assignment Plan:

1.	Start with Vgg16 model with binary output and weights from lesson2.5.py.
2.	Create an overfitted model:
    a. Split conv and FC layers into two separate models.
    b. Precalculate FC layer inputs from conv layer output.
    c. Remove dropout from the FC model.
    d. Fit the FC model to the data.
    e. Save the weights.
3.	Add data augmentation to the training set:
    a. Combine the Conv (locked) and FC models.
    b. Compile and train the combined model.
    c. Save the weights.
4.	Add batchnorm to the combined model:
    a. Create a standalone model from the Vgg16bn model's BN layers.
    b. Fit the BN model to the data.
    c. Save the weights.
    d. Create another BN model and combine it with the conv model into a final model.
    e. Set the BN layer weights from the first BN model (why not just combine *that* BN model with the conv model)?
    f. Save the weights.
5.	Fit the final model:
    a. Incrementally, saving the weights along the way.

lesson3.0.py:
- Based on lesson2.5.py
- now with functions

 '''

import os
import os.path
import click

import utils
from vgg16 import Vgg16

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
from keras.optimizers import SGD, RMSprop, Nadam
from keras.preprocessing import image

#
# Utility Functions
#
def onehot(x):
    # Returns two-column matrix with one row for each class.
    return np.array(OneHotEncoder().fit_transform(x.reshape(-1, 1)).todense())


#
# Constants
#
INPUT_PATH = None
OUTPUT_PATH = None
TRAIN_PATH = None
VALID_PATH = None
TEST_PATH = None
MODEL_PATH = None
RESULTS_PATH = None

BATCH_SIZE = None
NUM_EPOCHS = None

#
# Data Setup
#

def setup_folders():
    click.echo()
    click.echo('Setting up folders...')
    click.echo()

    click.echo('Input folder: %s' % INPUT_PATH)

    global TRAIN_PATH
    TRAIN_PATH = os.path.join(INPUT_PATH, 'train')
    click.echo('Training data: %s' % TRAIN_PATH)

    global VALID_PATH
    VALID_PATH = os.path.join(INPUT_PATH, 'valid')
    click.echo('Validation data: %s' % VALID_PATH)

    global TEST_PATH
    TEST_PATH = os.path.join(INPUT_PATH, 'test')
    click.echo('Test data: %s' % TEST_PATH)
    click.echo()

    click.echo('Output folder: %s' % OUTPUT_PATH)

    global MODEL_PATH
    MODEL_PATH = os.path.join(OUTPUT_PATH, 'models')
    if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)
    click.echo('Model data: %s' % MODEL_PATH)

    global RESULTS_PATH
    RESULTS_PATH = os.path.join(OUTPUT_PATH, 'results')
    if not os.path.exists(RESULTS_PATH): os.makedirs(RESULTS_PATH)
    click.echo('Results: %s' % RESULTS_PATH)
    click.echo()


def load_data():
    #
    # NOTE: Loading and use of data structures is pretty fucked up here.
    # Some things require getting data from generators, others require NumPy arrays.
    # In the end we use both, and sometimes re-load the data from disk and/or re-transform
    # it more than once.
    #
    click.echo('Loading raw training data from %s...' % TRAIN_PATH)
    global TRAIN_BATCHES
    TRAIN_BATCHES = utils.get_batches(TRAIN_PATH, shuffle=False, batch_size=1)

    click.echo('Loading array from generator...')
    global TRAIN_ARRAY
    TRAIN_ARRAY = utils.get_data(TRAIN_PATH)
    click.echo('\tshape: %s' % (TRAIN_ARRAY.shape,))
    click.echo()

    # TRAIN_DATA = os.path.join(MODEL_PATH, 'train_data.bc')
    # click.echo('Saving processed training data to %s...' % TRAIN_DATA)
    # utils.save_array(TRAIN_DATA, TRAIN_ARRAY)

    click.echo('Loading raw validation data from %s...' % VALID_PATH)
    global VALID_BATCHES
    VALID_BATCHES = utils.get_batches(VALID_PATH, shuffle=False, batch_size=1)

    click.echo('Loading array from generator...')
    global VALID_ARRAY
    VALID_ARRAY = utils.get_data(VALID_PATH)
    click.echo('\tshape: %s' % (VALID_ARRAY.shape,))
    click.echo()


def get_true_labels():
    click.echo('Getting the true labels for every image...')

    global TRAIN_CLASSES
    TRAIN_CLASSES = TRAIN_BATCHES.classes
    global TRAIN_LABELS
    TRAIN_LABELS = onehot(TRAIN_CLASSES)
    # click.echo('\tTraining labels look like this: \n%s\n...\n%s' % (TRAIN_LABELS[:5], TRAIN_LABELS[-5:]))
    # click.echo()

    global VALID_CLASSES
    VALID_CLASSES = VALID_BATCHES.classes
    global VALID_LABELS
    VALID_LABELS = onehot(VALID_CLASSES)
    # click.echo('\tValidation labels look like this: \n%s\n...\n%s' % (VALID_LABELS[:5], VALID_LABELS[-5:]))
    # click.echo()


def prepare_generators():
    click.echo('Preparing image data generators...')
    gen = image.ImageDataGenerator()
    # NOTE: Why do we overwrite these generators using the arrays?
    # TRAIN_BATCHES and VALID_BATCHES here are generators,
    # but still not quite the same as above.
    global TRAIN_BATCHES
    TRAIN_BATCHES = gen.flow(TRAIN_ARRAY, TRAIN_LABELS,
                            batch_size=BATCH_SIZE, shuffle=True)
    global VALID_BATCHES
    VALID_BATCHES = gen.flow(VALID_ARRAY, VALID_LABELS,
                            batch_size=BATCH_SIZE, shuffle=False)


def create_model():
    vgg = Vgg16()
    vgg.model.pop()

    click.echo('Replacing last layer of model...')
    for layer in vgg.model.layers: layer.trainable=False
    vgg.model.add(Dense(2, activation='softmax'))

    # OPTIMIZER = Nadam()
    OPTIMIZER = RMSprop(lr=0.001)
    vgg.model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

    return vgg, OPTIMIZER


def fit_model(model, opt):
    # First epoch higher LR
    LR=0.01
    K.set_value(opt.lr, LR)
    click.echo('Fitting last layer of model using LR=%s' % LR)
    model.fit_generator(TRAIN_BATCHES, samples_per_epoch=TRAIN_BATCHES.n, nb_epoch=NUM_EPOCHS,
                            validation_data=VALID_BATCHES, nb_val_samples=VALID_BATCHES.n)

    # Next batch, lower again
    LR=0.001
    K.set_value(opt.lr, LR)
    click.echo('Fitting last layer of model using LR=%s' % LR)
    model.fit_generator(TRAIN_BATCHES, samples_per_epoch=TRAIN_BATCHES.n, nb_epoch=NUM_EPOCHS,
                            validation_data=VALID_BATCHES, nb_val_samples=VALID_BATCHES.n)

    click.echo('Saving model weights...')
    model.save_weights(os.path.join(MODEL_PATH, 'finetune_1_ll.h5'))


def eval_model(model):
    click.echo('Evaluating model with validation data...')
    TEST_LOSS = model.evaluate(VALID_ARRAY, VALID_LABELS)
    click.echo('TEST_LOSS: %s' % (TEST_LOSS,))

    click.echo('Confusion matrix after last layer retraining')
    PREDS = model.predict_classes(VALID_ARRAY, batch_size=BATCH_SIZE)
    PROBS = model.predict_proba(VALID_ARRAY, batch_size=BATCH_SIZE)[:, 0]

    CM = confusion_matrix(VALID_CLASSES, PREDS)
    click.echo(CM)


def predict(model):
    click.echo('Predicting labels for test data set...')
    TEST_BATCHES = utils.get_batches(TEST_PATH, shuffle=False, batch_size=BATCH_SIZE)
    TEST_PREDS = model.predict_generator(TEST_BATCHES, TEST_BATCHES.nb_sample)
    TEST_FILENAMES = TEST_BATCHES.filenames

    #Save our test results arrays so we can use them again later
    # click.echo('Saving raw prediction results.')
    # utils.save_array(os.path.join(MODEL_PATH, 'test_preds.dat'), TEST_PREDS)
    # utils.save_array(os.path.join(MODEL_PATH, 'filenames.dat'), TEST_FILENAMES)

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

    click.echo('Formatting and saving data for Kaggle submission.')
    np.savetxt(os.path.join(RESULTS_PATH, 'kaggle_submission.csv'), SUBMIT,
            fmt='%d,%.5f', header='id,label', comments='')

    click.echo('Model training and prediction complete.')


@click.command()
@click.option('--sample', is_flag=True, default=True, help='Use sample dataset for training.')
@click.option('--sample-set', default='sample', help='Sample dataset to train on.')
@click.option('--local', default=True, help='Local environment (vs. FloydHub)')
def main(sample, sample_set, local):

    global BATCH_SIZE
    global NUM_EPOCHS
    global INPUT_PATH
    global OUTPUT_PATH

    if local:
        BATCH_SIZE = 32
    else:
        BATCH_SIZE = 64

    INPUT_PATH = os.path.join('.', 'input')
    OUTPUT_PATH = os.path.join('.', 'output')

    if sample:
        INPUT_PATH = os.path.join(INPUT_PATH, sample_set)
        OUTPUT_PATH = os.path.join(OUTPUT_PATH, sample_set)

        NUM_EPOCHS = 4
    else:
        NUM_EPOCHS = 10

    setup_folders()
    load_data()

    get_true_labels()
    prepare_generators()

    vgg, opt = create_model()

    fit_model(vgg.model, opt)

    eval_model(vgg.model)
    predict(vgg.model)



if __name__ == '__main__':
    main()
