#!/bin/sh
#
# dogscats-redux/setup-restructure-data.sh
#
# This is mainly so I don't have to incur the time to do the download and unzipping all over again.
#
# NOTE: Run this using 'floyd run --data <ID of the data from dogscats-redux-data job> "sh setup-restructure-data.sh"'

# Copy the data from /input (RO) to /output (RW)
cp -r -v /input/* /output   # if it fails, assume we don't need it

# Add the 'unknown subfolder to the test data'
mkdir /output/test/unknown && \
mv /output/test/*.jpg /output/test/unknown && \

# Install split_training_data.py
cd ./split_training_data && \
pip install . && \
cd .. && \

# create classes
split_training_data partition /output/train && \

# split training set into training and validation sets
split_training_data split --percent 20 /output/train /output/valid && \

# make sample sets from each of training, validation, and test sets
split_training_data sample --percent 1 /output/train /output/sample/train && \
split_training_data sample --percent 1 /output/valid /output/sample/valid && \
split_training_data sample --percent 1 /output/test /output/sample/test/unknown     # test set must have a single 'unknown' class

echo Done!
