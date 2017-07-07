#!/usr/bin/env python
import numpy
import os
import tensorflow as tf
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import argparse

import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.vgg as vgg
import tensorflow.contrib.slim.python.slim.nets.resnet_v1 as resnet_v1
import tensorflow.contrib.slim.python.slim.nets.resnet_utils as slim_utils

from guesswhat.train.utils import RawImageLoader, RawCropLoader

from generic.data_provider.iterator import Iterator
from generic.utils.file_handlers import pickle_dump

from guesswhat.data_provider.guesswhat_dataset import OracleDataset
from guesswhat.data_provider.oracle_batchifier import OracleBatchifier

parser = argparse.ArgumentParser('Feature extractor! ')

parser.add_argument("-image_dir", type=str, required=True, help="Input Image folder")
parser.add_argument("-data_dir", type=str, required=True,help="Dataset folder")
parser.add_argument("-set_type", type=list, default=["valid", "train", "test"], help='Select the dataset to dump')

parser.add_argument("-data_out", type=str, required=True, help="Output folder")

parser.add_argument("-network", type=str, choices=["resnet", "vgg"], help="Use resnet/vgg network")
parser.add_argument("-ckpt", type=str, required=True, help="Path for network checkpoint: ")
parser.add_argument("-feature_name", type=str, default="", help="Pick the name of the network features default=(fc8 - block4)")

parser.add_argument("-mode", type=str, choices=["img", "crop"], help="Select to either dump the img/crop feature")
parser.add_argument("-subtract_mean", type=bool, default=True, help="Preprocess the image by substracting the mean")
parser.add_argument("-img_size", type=int, default=224, help="image size (pixels)")
parser.add_argument("-crop_scale", type=float, default=1.1, help="crop scale around the bbox")
parser.add_argument("-batch_size", type=int, default=64, help="Batch size to extract features")

parser.add_argument("-gpu_ratio", type=float, default=1., help="How many GPU ram is required? (ratio)")
parser.add_argument("-no_thread", type=int, default=1, help="No thread to load batch")

args = parser.parse_args()




# define image
if args.subtract_mean:
    channel_mean = np.array([123.68, 116.779, 103.939])
else:
    channel_mean = None


# define the image loader (raw vs crop)
if args.mode == "img":
    images = tf.placeholder(tf.float32, [None, args.img_size, args.img_size, 3], name='image')
    source = 'image'
    image_loader = RawImageLoader(args.image_dir,
                                height=args.img_size,
                                width=args.img_size,
                                channel=channel_mean)
    crop_loader=None
elif args.mode == "crop":
    images = tf.placeholder(tf.float32, [None, args.img_size, args.img_size, 3], name='crop')
    source = 'crop'
    image_loader = None
    crop_loader = RawCropLoader(args.image_dir,
                                  height=args.img_size,
                                  width=args.img_size,
                                  scale=args.crop_scale,
                                  channel=channel_mean)
else:
    assert False, "Invalid mode: {}".format(args.mode)


# Define the output folder
out_file = "gw_{mode}_{network}_{feature_name}_{size}".format(
    mode=args.mode, network=args.network, feature_name=args.feature_name, size=args.img_size)


print("Create networks...")
if args.network == "resnet":

    # create network
    with slim.arg_scope(slim_utils.resnet_arg_scope(is_training=False)):
        _, end_points = resnet_v1.resnet_v1_152(images, 1000)  # 1000 is the number of softmax class

    # define the feature name according slim standard
    feature_name = os.path.join("resnet_v1_152", args.feature_name)

    # create the output directory
    out_dir = os.path.join(args.data_dir, out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

elif args.network == "vgg":
    _, end_points = vgg.vgg_16(images)
    out_dir = os.path.join(args.data_dir, out_file + ".pkl")
    feature_name = os.path.join("vgg_16", args.feature_name)
else:
    assert False, "Incorrect Network"

# check that the feature name is correct
assert feature_name in end_points, \
    "Invalid Feature name ({}), Must be on of the following {}"\
        .format({feature_name}, end_points.keys())




# CPU/GPU option
cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, args.ckpt)

    features = dict()
    for one_set in args.set_type:

        print("Load dataset -> set: {}".format(one_set))
        dataset = OracleDataset.load(args.data_dir, one_set, image_loader=image_loader, crop_loader=crop_loader)
        batchifier = OracleBatchifier(tokenizer=None, sources=[source])
        iterator = Iterator(dataset,
                            batch_size=args.batch_size,
                            pool=cpu_pool,
                            batchifier=batchifier)

        for batch in tqdm(iterator):
            feat = sess.run(end_points[feature_name], feed_dict={images: numpy.array(batch[source])})
            for f, game in zip(feat, batch["raw"]):
                f = f.squeeze()

                if args.mode == "crop":
                    id =  game.object_id
                else:
                    id = game.picture.id


                if args.network == "resnet":
                    np.savez_compressed(os.path.join(out_dir, "{}.npz".format(id)), x="features")
                else:
                    features[id] = f

if args.network == "vgg":
    print("Dump file...")
    pickle_dump(features, out_dir)

print("Done!")
