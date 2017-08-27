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

lesson3.1.py:
- Based on lesson3.0.py
- Replacing lesson 2 models/logic with that from lesson3.ipynb


'''

import os
import os.path
import click

import utils
from vgg16 import Vgg16

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

import keras
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, Nadam, Adam
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

TRAIN_BATCHES = None
TRAIN_ARRAY = None

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
    TRAIN_BATCHES = utils.get_batches(TRAIN_PATH, shuffle=False, batch_size=BATCH_SIZE)

    click.echo('Loading array from generator...')
    TRAIN_ARRAY = utils.get_data(TRAIN_PATH)
    click.echo('\tshape: %s' % (TRAIN_ARRAY.shape,))
    click.echo()

    # TRAIN_DATA = os.path.join(MODEL_PATH, 'train_data.bc')
    # click.echo('Saving processed training data to %s...' % TRAIN_DATA)
    # utils.save_array(TRAIN_DATA, TRAIN_ARRAY)

    click.echo('Loading raw validation data from %s...' % VALID_PATH)
    VALID_BATCHES = utils.get_batches(VALID_PATH, shuffle=False, batch_size=BATCH_SIZE)

    click.echo('Loading array from generator...')
    VALID_ARRAY = utils.get_data(VALID_PATH)
    click.echo('\tshape: %s' % (VALID_ARRAY.shape,))
    click.echo()

    return TRAIN_BATCHES, VALID_BATCHES, TRAIN_ARRAY, VALID_ARRAY


def get_true_labels(train_batches, valid_batches):
    click.echo('Getting the true labels for every image...')

    train_classes = train_batches.classes
    train_labels = onehot(train_classes)
    # click.echo('\ttraining labels look like this: \n%s\n...\n%s' % (train_labels[:5], train_labels[-5:]))
    # click.echo()

    valid_classes = valid_batches.classes
    valid_labels = onehot(valid_classes)
    # click.echo('\tvalidation labels look like this: \n%s\n...\n%s' % (valid_labels[:5], valid_labels[-5:]))
    # click.echo()

    return train_classes, valid_classes, train_labels, valid_labels


# def prepare_generators():
#     click.echo('Preparing image data generators...')
#     gen = image.ImageDataGenerator()
#     # NOTE: Why do we overwrite these generators using the arrays?
#     # TRAIN_BATCHES and VALID_BATCHES here are generators,
#     # but still not quite the same as above.
#     global TRAIN_BATCHES
#     TRAIN_BATCHES = gen.flow(TRAIN_ARRAY, TRAIN_LABELS,
#                             batch_size=BATCH_SIZE, shuffle=True)
#     global VALID_BATCHES
#     VALID_BATCHES = gen.flow(VALID_ARRAY, VALID_LABELS,
#                             batch_size=BATCH_SIZE, shuffle=False)


def create_model(opt):
    vgg = Vgg16()
    vgg.model.pop()

    click.echo('Replacing last layer of model...')
    for layer in vgg.model.layers: layer.trainable=False
    vgg.model.add(Dense(2, activation='softmax'))

    vgg.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    vgg.model.load_weights(os.path.join(MODEL_PATH, 'finetune_1_ll.h5'))

    return vgg


def split_conv_and_fc_layers(model):
    click.echo('Splitting convolutional and fully-connected layers...')
    layers = model.layers
    last_conv_idx = [index for index, layer in enumerate(layers) if type(layer) is Convolution2D][-1]   # last index of Conv layers

    click.echo('Last convolutional layer is: %d' % last_conv_idx)

    conv_layers = layers[:last_conv_idx + 1]  # conv layers only; i.e. first N layers until the index.
    conv_model = Sequential(conv_layers)
    fc_layers = layers[last_conv_idx + 1:] # remaining layers are Dense or fully-connected (FC)

    return conv_model, conv_layers, fc_layers


def precalculate_conv_output(model, train_batches, valid_batches):
    click.echo('Precalculating convolutional layer outputs...')
    train_features = model.predict_generator(train_batches, train_batches.nb_sample)
    click.echo('train_features shape: %s' % (train_features.shape,))

    valid_features = model.predict_generator(valid_batches, valid_batches.nb_sample)
    click.echo('valid_features shape: %s' % (valid_features.shape,))

    click.echo('Saving data...')
    utils.save_array(os.path.join(MODEL_PATH, 'train_convlayer_features.bc'), train_features)
    utils.save_array(os.path.join(MODEL_PATH, 'valid_convlayer_features.bc'), valid_features)

    return train_features, valid_features


def augment_data():
    gen = image.ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                   height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)
    train_batches = utils.get_batches(TRAIN_PATH, gen, batch_size=BATCH_SIZE)

    # NB: We don't want to augment or shuffle the validation set
    valid_batches = utils.get_batches(VALID_PATH, shuffle=False, batch_size=BATCH_SIZE)

    return train_batches, valid_batches


# Copy the weights from the pre-trained model.
# NB: Since we're removing dropout, we want to half the weights
def proc_no_dropout_wgts(layer):
    return [o / 2 for o in layer.get_weights()]


def get_fc_model_no_dropout(opt, conv_layers, fc_layers):
    '''
    Create a Dense model with dropout removed.
    '''
    model = Sequential([
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.),
        Dense(4096, activation='relu'),
        Dropout(0.),
        Dense(2, activation='softmax')
    ])

    for l1, l2 in zip(model.layers, fc_layers):
        l1.set_weights(proc_no_dropout_wgts(l2))

    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def proc_bn_wgts(layer, prev_p, new_p):
    scal = (1-prev_p)/(1-new_p)
    return [o*scal for o in layer.get_weights()]


def get_bn_layers(p, conv_layers):
    return [
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dense(4096, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(4096, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(1000, activation='softmax')
    ]


def load_fc_weights_from_vgg16bn(model):
    "Load weights for model from the dense layers of the Vgg16BN model."
    # See imagenet_batchnorm.ipynb for info on how the weights for
    # Vgg16BN can be generated from the standard Vgg16 weights.
    from vgg16bn import Vgg16BN
    vgg16_bn = Vgg16BN()
    _, fc_layers = utils.split_at(vgg16_bn.model, Convolution2D)
    utils.copy_weights(fc_layers, model.layers)


def combine_models(opt, conv_model, fc_model):
    # Look how easy it is to connect two models together!
    for layer in conv_model.layers:
        layer.trainable = False
    conv_model.add(fc_model)
    conv_model.compile(
        optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return conv_model


def create_final_model(opt, conv_layers, bn_model):
    final_model = Sequential(conv_layers)
    for layer in final_model.layers: layer.trainable = False
    final_model.add(bn_model)
    final_model.compile(optimizer=opt,
                        loss='categorical_crossentropy', metrics=['accuracy'])
    return final_model


def get_fc_model_batchnorm(p, conv_layers):
    bn_model = Sequential(get_bn_layers(p, conv_layers))
    load_fc_weights_from_vgg16bn(bn_model)
    for l in bn_model.layers:
        if type(l) == Dense:
            l.set_weights(proc_bn_wgts(l, 0.5, 0.6))

    # Remove last layer and lock all the others
    bn_model.pop()
    for layer in bn_model.layers:
        layer.trainable = False

    # Add linear layer (2-class) (just doing the ImageNet mapping to Kaggle dogs and cats)
    bn_model.add(Dense(2, activation='softmax'))

    bn_model.compile(Adam(), 'categorical_crossentropy', metrics=[
                     'accuracy'])      # NOTE: Adam optimizer

    return bn_model



def fit_model(model, train_batches, valid_batches):
    click.echo('Fitting model...')
    model.fit_generator(train_batches, samples_per_epoch=train_batches.n, nb_epoch=NUM_EPOCHS,
                            validation_data=valid_batches, nb_val_samples=valid_batches.n)


def get_array(batches):
    array = np.concatenate([batches.next() for i in range(batches.nb_sample)])
    batches.reset()

    return array



def eval_model(model, valid_array, valid_classes, valid_labels):
    click.echo('evaluating model with validation data...')
    test_loss = model.evaluate(valid_array, valid_labels)
    click.echo()
    click.echo('test_loss: %s' % (test_loss,))

    click.echo('confusion matrix...')
    preds = model.predict_classes(valid_array, batch_size=BATCH_SIZE)
    probs = model.predict_proba(valid_array, batch_size=BATCH_SIZE)[:, 0]
    click.echo()

    cm = confusion_matrix(valid_classes, preds)
    click.echo(cm)


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


@click.command(context_settings={'allow_extra_args': True})
@click.option('--sample', is_flag=True, default=True, help='Use sample dataset for training.')
@click.option('--sample-set', default='sample', help='Sample dataset to train on.')
@click.option('--local', default=True, help='Local environment (vs. FloydHub)')
def main(sample, sample_set, local):

    # setup

    global BATCH_SIZE
    global NUM_EPOCHS
    global INPUT_PATH
    global OUTPUT_PATH

    if local:
        BATCH_SIZE = 16
    else:
        BATCH_SIZE = 64

    INPUT_PATH = os.path.join('.', 'input')
    OUTPUT_PATH = os.path.join('.', 'output')

    if sample:
        INPUT_PATH = os.path.join(INPUT_PATH, sample_set)
        OUTPUT_PATH = os.path.join(OUTPUT_PATH, sample_set)

        NUM_EPOCHS = 8
    else:
        NUM_EPOCHS = 10

    setup_folders()

    # model creation and modification
    opt = RMSprop(lr=0.001)
    vgg = create_model(opt)
    conv_model, conv_layers, fc_layers = split_conv_and_fc_layers(vgg.model)

    # load data and labels
    train_batches, valid_batches, train_array, valid_array = load_data()
    train_classes, valid_classes, train_labels, valid_labels = get_true_labels(
        train_batches, valid_batches)

    # precalculate outputs from conv layers
    train_features, valid_features = precalculate_conv_output(
        conv_model, train_batches, valid_batches)

    # Remove dropout from the fc layers and train.
    # Such a finely tuned model needs to be updated very slowly!
    opt = RMSprop(lr=0.00001, rho=0.7)
    fc_model = get_fc_model_no_dropout(opt, conv_layers, fc_layers)
    fc_model.fit(train_features, train_labels, nb_epoch=NUM_EPOCHS,
                 batch_size=BATCH_SIZE, validation_data=(valid_features, valid_labels))

    click.echo('Saving model weights...')
    fc_model.save_weights(os.path.join(MODEL_PATH, 'lesson3_no_dropout.h5'))

    eval_model(fc_model, valid_features, valid_classes, valid_labels)

    # Now reduce overfitting using data augmentation
    train_batches, valid_batches = augment_data()

    combined_model = combine_models(opt, conv_model, fc_model)
    fit_model(combined_model, train_batches, valid_batches)

    click.echo('Saving combined model weights...')
    combined_model.save_weights(os.path.join(MODEL_PATH, 'aug1.h5'))

    #eval_model(combined_model, valid_features, valid_classes, valid_labels)

    # Further reduce overfitting using batchnorm
    p = 0.6
    bn_model = get_fc_model_batchnorm(p, conv_layers)
    bn_model.fit(train_features, train_labels, nb_epoch=NUM_EPOCHS,
                 batch_size=BATCH_SIZE, validation_data=(valid_features, valid_labels))

    click.echo('Saving batchnorm model weights...')
    bn_model.save_weights(os.path.join(MODEL_PATH, 'bn1.h5'))

    eval_model(bn_model, valid_features, valid_classes, valid_labels)

    #predict(vgg.model)


if __name__ == '__main__':
    main()
