# lesson1.py
# Stand-alone script to run the code from the lesson1.ipynb Jupyter Notebook.

from __future__ import division, print_function
import os
import json
from glob import glob
import numpy as np
import utils
from vgg16 import Vgg16

# INPUT_PATH = '/input/sample/'
INPUT_PATH = '/input/'
OUTPUT_PATH = '/output/'

np.set_printoptions(precision=4, linewidth=100)

# TODO: command line args


# As large as you can, but no larger than 64 is recommended.
# If you have an older or cheaper GPU, you'll run out of memory, so will have to decrease this.
BATCH_SIZE = 64
# NUM_EPOCHS = 3
NUM_EPOCHS = 6

# BATCH_SIZE = 32
# NUM_EPOCHS = 1


# The model class
vgg = Vgg16()

print ('Getting training and validation data batches.')

# TODO: load weights file
for epoch in range(NUM_EPOCHS):
    print ('Getting training and validation batches for epoch %d' % epoch)
    train_batches = vgg.get_batches(os.path.join(INPUT_PATH, 'train'), batch_size=BATCH_SIZE)
    val_batches = vgg.get_batches(os.path.join(INPUT_PATH, 'valid'), batch_size=BATCH_SIZE * 2)

    print ('Fine-tuning and fitting data for epoch %d.' % epoch)
    vgg.finetune(train_batches)
    vgg.fit(train_batches, val_batches, nb_epoch=2)

    print ('Saving updated weights for epoch %d.' % epoch)
    weights_file = os.path.join(OUTPUT_PATH, 'ft_%d.h5' % epoch)
    vgg.model.save_weights(weights_file)

print ('Predicting labels for new data.')
batches, preds = vgg.test(os.path.join(INPUT_PATH, 'test'), batch_size=BATCH_SIZE * 2)
filenames = batches.filenames

#Save our test results arrays so we can use them again later
print('Saving raw prediction results.')
utils.save_array('/output/test_preds.dat', preds)
utils.save_array('/output/filenames.dat', filenames)

# Grab the dog prediction column
isdog = preds[:, 1]

# To play it safe, we use a sneaky trick to round down our edge predictions
# Swap all ones with .95 and all zeros with .05
isdog = isdog.clip(min=0.05, max=0.95)

# Extract imageIds from the filenames in our test/unknown directory
filenames = batches.filenames
ids = np.array([int(os.path.splitext(os.path.basename(f))[0]) for f in filenames])

# Combine the ids and isdog columns into a single 2-column array.
subm = np.stack([ids, isdog], axis=1)

print('Formatting and saving data for Kaggle submission.')
np.savetxt(os.path.join(OUTPUT_PATH, 'kaggle_submission.csv'), subm,
           fmt='%d,%.5f', header='id,label', comments='')

print('Model training and prediction complete.')
