#!/usr/bin/env python
import numpy
import os
import pickle
import tensorflow as tf
from multiprocessing import Pool
import collections
from tqdm import tqdm
import numpy as np


import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.vgg as vgg

from guesswhat.train.utils import RawImageLoader

from generic.data_provider.iterator import Iterator
from generic.utils.file_handlers import pickle_dump

from guesswhat.data_provider.guesswhat_dataset import OracleDataset
from guesswhat.data_provider.oracle_batchifier import OracleBatchifier


image_w, image_h = 224, 224
batch_size = 6
channel_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)


chekpt =  '/home/sequel/fstrub/guesswhat_data/vgg_16.ckpt'
dataset_dir = '/home/sequel/fstrub/guesswhat_data/'
image_dir = '/home/sequel/fstrub/guesswhat_data/guesswhat_images/'
data_out = '/home/sequel/fstrub/guesswhat_data/'
set_type = ["valid", "train", "test"]


output_name = "vgg_16/fc8"
filename="vgg_fc8.pkl"






print("Load parameters...")

images = tf.placeholder(tf.float32, [None, image_h, image_w, 3], name='images')
_, end_points = vgg.vgg_16(images)


image_loader = RawImageLoader(image_dir,
                                height=image_h,
                                width=image_w,
                                channel=channel_mean)


cpu_pool = Pool(1, maxtasksperchild=1000)
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, chekpt)

    features = dict()
    for one_set in set_type:

        print("Load dataset -> set: {}".format(one_set))
        dataset = OracleDataset.load(dataset_dir, one_set, image_loader)
        batchifier = OracleBatchifier(tokenizer=None, sources=["image"])
        iterator = Iterator(dataset,
                            batch_size=batch_size,
                            pool=cpu_pool,
                            batchifier=batchifier)

        for batch in tqdm(iterator):
            feat = sess.run(end_points[output_name], feed_dict={images: numpy.array(batch['images'])})
            for f, game in zip(feat, batch["raw"]):
                f = f.squeeze()
                features[game.picture.id] = f


# mix all set together
pickle_dump(features, os.path.join(data_out, filename))

