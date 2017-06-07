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
from guesswhat.data_provider.dataset import Dataset
from guesswhat.data_provider.nlp_preprocessors import GWTokenizer


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


def load_data(data_dir, load_crop=False, load_picture=False, image_dir=None):
    logger = logging.getLogger()
    ###############################
    #  LOAD DATA
    #############################

    fc8_img, fc8_crop = None, None
    if load_picture:
        logger.info('Loading picture vgg..')
        fc8_img = pickle_loader(os.path.join(data_dir, "vgg_img.pkl"))

    if load_crop:
        logger.info('Loading crop vgg..')
        fc8_crop = pickle_loader(os.path.join(data_dir, "vgg_crop.pkl"))

    # Load data
    logger.info('Loading data..')
    im_dir = os.path.join(data_dir, 'guesswhat_images')
    if image_dir is not None:
        im_dir = os.path.join(image_dir, 'guesswhat_images')
    trainset = Dataset(data_dir, 'train', fc8_img=fc8_img, fc8_crop=fc8_crop, image_folder=im_dir)
    validset = Dataset(data_dir, 'valid', fc8_img=fc8_img, fc8_crop=fc8_crop, image_folder=im_dir)
    testset = Dataset(data_dir, 'test', fc8_img=fc8_img, fc8_crop=fc8_crop, image_folder=im_dir)

    # Load dictionary
    logger.info('Loading dictionary..')
    tokenizer = GWTokenizer(os.path.join(data_dir, 'dict.json'))

    environment = Environment(
        trainset=trainset,
        validset=validset,
        testset=testset,
    )

    return environment, tokenizer
