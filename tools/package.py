# coding: utf-8
from __future__ import print_function
from __future__ import absolute_import

import argparse
import glob
import os
import pickle
import random


def pickle_examples(path, train_path, val_path, train_val_split=0.1):
    """
    Compile a list of examples into pickled format, so during the training, all io will happen in memory.
    :param path: 
    :param train_path: 
    :param val_path: 
    :param train_val_split: 
    :return: 
    """
    with open(train_path, 'wb') as ft:
        with open(val_path, 'wb') as fv:
            for p in path:
                with open(p, 'rb') as f:
                    print("image %s" % p)
                    img_bytes = f.read()
                    r = random.random()
                    if r < train_val_split:
                        pickle.dump(img_bytes, fv)
                    else:
                        pickle.dump(img_bytes, ft)

parser = argparse.ArgumentParser(description="Compile list of images into a pickled object for training")
parser.add_argument("--dir", dest="dir", required=True, help="path of images")
parser.add_argument("--save_dir", dest="save_dir", required=True, help="path to save pickled files")
parser.add_argument("--split_ratio", type=float, default=0.1, dest="split_ratio",
                    help="split ratio between train and val")

args = parser.parse_args()

if __name__ == "__main__":
    train_path = os.path.join(args.save_dir, "train.obj")
    val_path = os.path.join(args.save_dir, "val.obj")
    pickle_examples(glob.glob(os.path.join(args.dir, "*.jpg")), train_path=train_path, val_path=val_path,
                    train_val_split=args.split_ratio)
