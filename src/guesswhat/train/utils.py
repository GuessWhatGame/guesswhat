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


def get_img_loader(config, image_dir):

    image_input = config["image_input"]

    if image_input == "features":
        is_flat = len(config["dim"]) == 1
        if is_flat:
            loader = fcLoader(image_dir)
        else:
            loader = ConvLoader(image_dir)
    elif image_input == "raw":
        loader = RawImageLoader(image_dir,
                                height=config["dim"][0],
                                width=config["dim"][1],
                                channel=config.get("channel", None),
                                extension=config.get("extension", "jpg"))
    else:
        assert False, "incorrect image input: {}".format(image_input)

    return loader