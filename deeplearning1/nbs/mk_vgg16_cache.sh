#!/bin/sh
#
# Copy vgg16.h5 file from mounted data source to .keras cache

echo 'Copying vgg16.h5 file to local cache.'
mkdir -p /root/.keras/models && \
cp /vgg16/vgg16.h5 /root/.keras/models/
set EXIT_CODE=$?
echo 'Done copying vgg16.h5 file to local cache.'
exit $EXIT_CODE
