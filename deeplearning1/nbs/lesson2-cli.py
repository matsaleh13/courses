#!/bin/python

"""
lesson2.py

This script implements the key points from the Lesson 2 exercises, expressed in the lesson2.pydb Jupyter Notebook.

Usage:
    lesson2.py [options] <command> [command-options] <data-folder>

Where:
    commands:
        train               Train and/or fine-tune the model using training/validation data in the data-folder.
            options:
                --batch-size
                --epochs
                --learn-rate

        predict             Use a trained model to predict the image classes from test data in the data-folder.

    options:
        --cache             Load data from local data files from previous runs to speed processing.
        --weights <file>    Load model weights from file.
        --test              Just go through the motions, don't actually do anything.

Example:
    lesson2.py --data <folder> --cache train
    lesson2.py --data <folder> --weights <file> predict

"""

import os
import random
import shutil
import fnmatch

import click


TRAIN_FOLDER = 'train'
VALID_FOLDER = 'valid'
TEST_FOLDER = 'test'
SAMPLE_FOLDER = 'sample'
MODEL_FOLDER = 'model'

TEST_ONLY = False
CLASSES = [
    # (filename, foldername),
    ('dog', 'dogs'),
    ('cat', 'cats')
]


@click.group()
@click.option('--cache', is_flag=True, help='Load data from local data files from previous runs to speed processing.')
@click.option('--weights', is_flag=True, help='Load data from local data files from previous runs to speed processing.')
@click.option('--test', is_flag=True, help='Show what will happen, but don\'t actually do it.')
def cli(test):
    '''Root (pseudo) command'''
    global TEST_ONLY
    TEST_ONLY = test
    if TEST_ONLY:
        click.echo('Running in TEST MODE; no changes will be made.')


@cli.command(help='Train and/or fine-tune the model using training/validation data in the data-folder.')
@click.argument('data-folder')
def train(folder):
    '''
    Trains and/or fine-tunes the vgg16 model to recognize dog or cat images only.

    The given folder must contain the following sub-folders:
        train/
        test/
        valid/

    Arguments:

    folder -- The root folder of the training, test, and validation data sets.
    '''
    folder_path = os.path.normpath(folder)

    click.echo('Validating data folder %s' % (folder_path,))

    validateFolder(folder_path)
    validateFolder(os.path.join(folder_path, TRAIN_FOLDER))
    validateFolder(os.path.join(folder_path, VALID_FOLDER))
    validateFolder(os.path.join(folder_path, TEST_FOLDER))

    click.echo('Training the model using data from %s' % (folder_path,))

    click.echo('Finished training the model using data from %s.' % (folder_path,))


@cli.command(help='Use a trained model to predict the image classes from test data in the data-folder.')
@click.argument('data-folder')
def predict(folder):
    '''
    Predicts whether images in the test data set are dogs or cats.

    The given folder must contain the following sub-folders:
        train/
        test/
        valid/

    Arguments:

    folder -- The root folder of the training, test, and validation data sets.
    '''
    folder_path = os.path.normpath(folder)

    click.echo('Validating data folder %s' % (folder_path,))

    validateFolder(folder_path)
    validateFolder(os.path.join(folder_path, TRAIN_FOLDER))
    validateFolder(os.path.join(folder_path, VALID_FOLDER))
    validateFolder(os.path.join(folder_path, TEST_FOLDER))

    click.echo('Predicting using data from %s' % (folder_path,))

    click.echo('Finished training the model using data from %s.' %
               (folder_path,))

@cli.command(help='Splits part of source_folder contents into dest_folder.')
@click.option('--percent', default=20, help='Size of validation data set as percent of all files.')
@click.argument('source_folder')
@click.argument('dest_folder')
def split(source_folder, dest_folder, percent):
    '''
    Moves a subset of the contents of source_folder into dest_folder.
    '''
    source_folder = os.path.normpath(source_folder)
    dest_folder = os.path.normpath(dest_folder)

    click.echo('Splitting [%s%%] of data set from folder [%s] into folder [%s].' % (
        percent, source_folder, dest_folder))

    validateFolder(source_folder)

    if not os.path.exists(dest_folder):
        if TEST_ONLY:
            click.echo('TEST: creating folder [%s]' % (dest_folder,))
        else:
            os.makedirs(dest_folder)

    file_paths = getFilePaths(source_folder)
    selected_paths = selectRandomPaths(source_folder, percent)

    click.echo('Selected [%s] random files of [%s] from [%s]' %
               (len(selected_paths), len(file_paths), source_folder))

    moveFiles(selected_paths, source_folder, dest_folder)
    click.echo('Finished splitting [%s%%] of data set from folder [%s] into folder [%s].' %
               (percent, source_folder, dest_folder))


@cli.command(help='Create a sample data set from folder contents.')
@click.option('--percent', default=1, help='Size of sample data set as percent of source data set.')
@click.argument('source_folder')
@click.argument('dest_folder')
def sample(source_folder, dest_folder, percent):
    '''
    Copies the a percentage of the files in source_folder using into dest_folder.
    '''
    click.echo('Creating sample dataset, using [%s percent] of [%s] contents in folder [%s]' %
               (percent, source_folder, dest_folder))

    validateFolder(source_folder)

    if not os.path.exists(dest_folder):
        if TEST_ONLY:
            click.echo('TEST: creating folder [%s]' % (dest_folder,))
        else:
            os.makedirs(dest_folder)

    file_paths = getFilePaths(source_folder)
    selected_paths = selectRandomPaths(source_folder, percent)

    click.echo('Selected [%s] random files of [%s] from [%s]' %
               (len(selected_paths), len(file_paths), source_folder))

    copyFiles(selected_paths, source_folder, dest_folder)
    click.echo('Finished copying [%s%%] of data set from folder [%s] into folder [%s].' %
               (percent, source_folder, dest_folder))


def validateFolder(folder):
    '''
    Assert that the given folder exists.

    Arguments:

    folder -- The folder to validate.
    '''
    assert os.path.exists(folder), 'Folder [%s] does not exist' % folder


def selectRandomPaths(folder_path, percent):
    '''
    Randomly select a subset of files from the given folder.

    Return a sequence containing the paths to the selected files.
    '''
    file_paths = getFilePaths(folder_path)
    selected_paths = selectRandomPercent(file_paths, percent)

    return selected_paths


def getFilePaths(folder_path, file_match=None):
    '''
    Return a list of path names contained within the folder identified by folder_path.
    Optionally filter the result to match the file name pattern specified by file_match.
    Returned paths are not absolute; they are relative to the folder_path argument.

    Keyword arguments:

    folder_path -- Path to the folder containing the files.

    file_match -- Optional file name pattern by which to filter the result.
    '''
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        filtered_paths = [os.path.relpath(
            os.path.join(root, f), folder_path) for f in files]
        if file_match:
            filtered_paths = fnmatch.filter(filtered_paths, file_match)

        file_paths += filtered_paths

    return file_paths


def selectRandomPercent(source_seq, percent):
    '''
    Randomly select elements from source_seq such that the
    number of selected elements comprises the given percentage of the
    total number of elements in the sequence.

    Return a sequence containing the selected elements.
    The original sequence isn't changed.

    Arguments:

    source_seq -- A sequence containing the set of elements from which to select.

    percent -- The percentage of source_seq elements to select.
    '''
    select_count = int(len(source_seq) * percent / 100.0)
    selected_seq = random.sample(source_seq, select_count)

    return selected_seq


def moveFiles(paths, source_folder, dest_folder):
    '''
    Move all the files in the paths array from the source_folder to the dest_folder.

    The files to move are represented using relative paths, relative to both the
    source_folder and dest_folder.

    If dest_folder doesn't exist, it will be created, down to the leafmost level.

    Arguments:

    paths -- List of relative paths identifying the files to move.

    source_folder -- Path to the folder containing the files to move.

    dest_folder -- Path to the folder into which the files will be moved.
    '''
    for path in paths:
        source_path = os.path.join(source_folder, path)
        dest_path = os.path.join(dest_folder, path)
        dest_parent_path = os.path.dirname(dest_path)

        if not os.path.exists(dest_parent_path):
            if TEST_ONLY:
                click.echo('TEST: creating folder %s' % (dest_parent_path,))
            else:
                os.makedirs(dest_parent_path)

        if TEST_ONLY:
            click.echo('TEST: Moving %s to %s' % (source_path, dest_path))
        else:
            click.echo('Moving %s to %s' % (source_path, dest_path))
            shutil.move(source_path, dest_path)

    click.echo('Moved %s files from %s to %s' %
               (len(paths), source_folder, dest_folder))


def copyFiles(paths, source_folder, dest_folder):
    '''
    Copy all the files in the paths array from the source_folder to the dest_folder.

    The files to move are represented using relative paths, relative to both the
    source_folder and dest_folder.

    If dest_folder doesn't exist, it will be created, down to the leafmost level.

    Arguments:

    paths -- List of relative paths identifying the files to copy.

    source_folder -- Path to the folder containing the files to copy.

    dest_folder -- Path to the folder into which the files will be copied.
    '''
    for path in paths:
        source_path = os.path.join(source_folder, path)
        dest_path = os.path.join(dest_folder, path)
        dest_parent_path = os.path.dirname(dest_path)

        if not os.path.exists(dest_parent_path):
            if TEST_ONLY:
                click.echo('TEST: creating folder %s' % (dest_parent_path,))
            else:
                os.makedirs(dest_parent_path)

        if TEST_ONLY:
            click.echo('TEST: Copying %s to %s' % (source_path, dest_path))
        else:
            click.echo('Copying %s to %s' % (source_path, dest_path))
            shutil.copy(source_path, dest_path)

    click.echo('Copied %s files from %s to %s' %
               (len(paths), source_folder, dest_folder))
