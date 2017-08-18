# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import os
import argparse

from models.font2font_cGAN import Font2Font
from util.dataset import InjectDataProvider

parser = argparse.ArgumentParser(description="test for the cgan model")
parser.add_argument('--model_dir', dest='model_dir', required=True,
                    help='Directory that saves the model checkpoints')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--source_obj', dest='source_obj', type=str, required=True, help='the source images for inference')
parser.add_argument('--save_dir', default='save_dir', type=str, help='path to save inferred images')

args = parser.parse_args()


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        source_provider = InjectDataProvider(args.source_obj)
        source_len = len(source_provider.data.examples)
        # source_len = min(10, source_len)

        model = Font2Font(batch_size=source_len)
        model.register_session(sess)
        model.build_model(is_training=False)

        model.test(source_provider, args.model_dir, args.save_dir)


if __name__ == '__main__':
    tf.app.run()