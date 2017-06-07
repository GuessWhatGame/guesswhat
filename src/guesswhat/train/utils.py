from __future__ import unicode_literals
import collections
import hashlib
import json
import logging
import numpy as np
import os
import shutil
import tensorflow as tf
import sys
import pickle

import gzip

from logging.handlers import RotatingFileHandler
from guesswhat.data_provider.guesswhat_dataset import Dataset

from generic.data_provider.image_loader import *

Environment = collections.namedtuple('Environment',  ['trainset', 'validset', 'testset'])

def load_checkpoint(sess, saver, args, save_path):
    ckpt_path = save_path.format('params.ckpt')

    if args.continue_exp:
        if not os.path.exists(save_path.format('checkpoint')):
            raise ValueError("Checkpoint " + save_path.format('checkpoint') + " could not be found.")

        saver.restore(sess, ckpt_path)
        status_path = save_path.format('status.pkl')
        status = pickle.load(open(status_path, 'rb'))

        return status['epoch'] + 1

    if args.load_checkpoint is not None:
        #if not os.path.exists(save_path.format('checkpoint')):
        #    raise ValueError("Checkpoint " + args.load_checkpoint + " could not be found.")
        saver.restore(sess, args.load_checkpoint)

        return 0

    return 0


def get_img_loader(config, input_type, image_dir):
    # load images
    loader = None
    if config['inputs'].get(input_type, False):
        use_conv = len(config[input_type]["dim"]) > 1
        if use_conv:
            loader = ConvLoader(image_dir)
        else:
            loader = fcLoader(image_dir)
    return loader