# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import argparse
from datetime import datetime

from models.font2font_cGAN import Font2Font

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--experiment_dir', dest='experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--experiment_id', dest='experiment_id', type=int, default=0,
                    help='sequence id for the experiments you prepare to run')
parser.add_argument('--image_size', dest='image_size', type=int, default=256,
                    help="size of your input and output image")
parser.add_argument('--L1_penalty', dest='L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lconst_penalty', dest='Lconst_penalty', type=int, default=15, help='weight for const loss')
parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--schedule', dest='schedule', type=int, default=10, help='number of epochs to half learning rate')
parser.add_argument('--resume', dest='resume', type=int, default=1, help='resume from previous training')
parser.add_argument('--freeze_encoder', dest='freeze_encoder', type=int, default=0,
                    help="freeze encoder weights during training")

parser.add_argument('--inst_norm', dest='inst_norm', type=int, default=0,
                    help='use conditional instance normalization in your model')
parser.add_argument('--sample_steps', dest='sample_steps', type=int, default=10,
                    help='number of batches in between two samples are drawn from validation set')
parser.add_argument('--checkpoint_steps', dest='checkpoint_steps', type=int, default=500,
                    help='number of batches in between two checkpoints')

args = parser.parse_args()


def main(_):
    start = datetime.now()
    print("Begin time: {}".format(start.isoformat(timespec='seconds')))

    print("Args:{}".format(args))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = Font2Font(args.experiment_dir, batch_size=args.batch_size, experiment_id=args.experiment_id,
                          input_width=args.image_size, output_width=args.image_size, L1_penalty=args.L1_penalty,
                          Lconst_penalty=args.Lconst_penalty, Ltv_penalty=args.Ltv_penalty)
        model.register_session(sess)
        model.build_model(is_training=True)

        model.train(lr=args.lr, epoch=args.epoch, resume=args.resume,
                    schedule=args.schedule, freeze_encoder=args.freeze_encoder,
                    sample_steps=args.sample_steps, checkpoint_steps=args.checkpoint_steps)

    end = datetime.now()
    print("Ending time: {}".format(end))
    duration = end - start
    print("Duration hours: {}".format(duration.total_seconds() / 3600.0))

if __name__ == '__main__':
    tf.app.run()