#!/bin/sh
#
# dogscats-redux/setup-get-kaggle-data.sh
#
# Shell script for installing the Kaggle CLI and data and extracting it.
#
# NOTE: Run this using 'env PASSWD=<password> floyd run "sh setup-get-kaggle-data.sh"'


pip install -U kaggle-cli && \

kg config -g -u matsaleh -p ${PASSWD} -c dogs-vs-cats-redux-kernels-edition && \
kg download && \

unzip -a -o train.zip -d /output && \
unzip -a -o test.zip -d /output && \

echo Done!
