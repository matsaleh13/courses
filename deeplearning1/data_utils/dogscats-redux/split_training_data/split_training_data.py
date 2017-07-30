#!/bin/python

"""
split_training_data.py

This script refactors a set of ML training data files into two data sets, a training set and a validation set. 
It optionally creates a sample data set that is a subset of the training and validation data sets.

Usage:
  split_training_data.py split --size <percent_of_total> <folder>

  No files are duplicated between the two sets. Files are simply moved from the training set into the validation set.
  The folder structure of the training set is preserved and mirrored in the validation set.
  The files to be moved to the validation set are chosen randomly from the training set.
  The validation set is smaller than the training set. Its size is represented as a percentage of the number of files
  in the original training set.
  The script takes a path to a folder that contains the training data as an argument. The folder must contain
  a single subfolder called 'train' and no other folders.

Usage:
  split_training_data.py sample --size <percent_of_original> <folder>

  In this case, files from the training and validation sets are copied into the sample training and
  validation subsets, and therefore duplicate their source files.
  The files to be copied to the sample training and validation subsets are chosen randomly from their
  source folders.
  The size of the sample training and validation subsets is expressed as a percentage of the number of files
  in their source data sets. 
  The sizes of the sample training and validation subsets maintain the same proportion to each other as that
  of their source data sets. E.g. if the source validation set is 20% of the overall file count, then the 
  size of the sample validation set will be 20% of the over sample file count.
  The script takes a path to a folder that contains the training and validation data as an argument. The folder
  must contain two subfolders, one called 'train' and the other, 'valid', and no other folders.
"""

import click
import os
import math
import random
import shutil
import fnmatch


TRAIN_FOLDER = 'train'
VALID_FOLDER = 'valid'
SAMPLE_FOLDER = 'sample'
TEST_ONLY = False
CLASSES = [
    # (filename, foldername),
    ('dog', 'dogs'), 
    ('cat', 'cats')
]


@click.group()
@click.option('--test', is_flag=True, help='Show what will happen, but don\'t actually do it.')
def cli(test):
    global TEST_ONLY
    TEST_ONLY = test
    if TEST_ONLY:
        click.echo('Running in TEST MODE; no changes will be made.')


@cli.command(help='Partition contents of folder into classes.')
@click.argument('folder')
def partition(folder):
    '''
    Partitions the files in the given folder into subfolders matching the class of each file,
    as identified by the file name (e.g. train/cat.9999.jpg => train/cats/cat.9999.jpg).

    Arguments:

    folder -- Path to the folder to partition.
    '''
    folder_path = os.path.normpath(folder)
    validateFolder(folder_path)

    click.echo('Partitioning [%s] folder with separate subfolders for each class.' % (folder_path,))

    for file_class, folder_class in CLASSES:
        file_match = '*' + file_class + '*'
        folder_class_path = os.path.join(folder_path, folder_class)
        class_paths = getFilePaths(folder_path, file_match)
        if class_paths:
            moveFiles(class_paths, folder_path, folder_class_path)
        else:
            click.echo('No files to move to folder [%s].' % folder_class_path)

    click.echo('Finished partitioning [%s] folder.' % (folder_path,))


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
    
    click.echo('Splitting [%s%%] of data set from folder [%s] into folder [%s].' % (percent, source_folder, dest_folder))

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
        filtered_paths = [os.path.relpath(os.path.join(root, f), folder_path) for f in files]
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
    
    click.echo('Moved %s files from %s to %s' % (len(paths), source_folder, dest_folder))


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
