#!/usr/bin/python

import argparse
import os
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--is_shuffled', default='1', type=int,
                    help='Needed to shuffle')

if __name__ == "__main__":

    args = parser.parse_args()

    train_filename = 'train.flist'
    validation_filename = 'val.flist'

    train_path = '/home/path/'
    val_path = '/home/path/'

    training_file_names = []
    validation_file_names = []

    training_folder = os.listdir(train_path)

    for training_item in training_folder:
        training_item = train_path + "/" + training_item
        training_file_names.append(training_item)

    validation_folder = os.listdir(val_path)

    for validation_item in validation_folder:
        validation_item = val_path + "/" + validation_item
        validation_file_names.append(validation_item)

    # print all file paths
    for i in training_file_names:
        print(i)
    for i in validation_file_names:
        print(i)

    # This would print all the files and directories

    # shuffle file names if set
    if args.is_shuffled == 1:
        shuffle(training_file_names)
        shuffle(validation_file_names)

    # make output file if not existed
    if not os.path.exists(train_filename):
        os.mknod(train_filename)

    if not os.path.exists(validation_filename):
        os.mknod(validation_filename)

    # write to file
    fo = open(train_filename, "w")
    fo.write("\n".join(training_file_names))
    fo.close()

    fo = open(validation_filename, "w")
    fo.write("\n".join(validation_file_names))
    fo.close()

    # print process
    print("Written file is: ", train_filename, ", is_shuffle: ", args.is_shuffled)
