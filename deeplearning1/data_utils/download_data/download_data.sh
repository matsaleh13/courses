#!/bin/sh
# download_data.sh
# Fetches data from remote site and saves it in the /output folder.
wget http://files.fast.ai/data/dogscats.zip && \
unzip -a -o dogscats.zip -d /output && \
rm dogscats.zip