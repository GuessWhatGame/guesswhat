#!/usr/bin/env python
import os
import tensorflow as tf
import numpy as np
import argparse
from distutils.util import strtobool

from neural_toolbox import resnet

import tensorflow.contrib.slim.python.slim.nets.vgg as vgg

from generic.data_provider.image_loader import RawImageBuilder, RawCropBuilder
from generic.preprocess_data.extract_img_features import extract_features

from guesswhat.data_provider.guesswhat_dataset import Dataset, CropDataset
from guesswhat.data_provider.oracle_batchifier import OracleBatchifier

parser = argparse.ArgumentParser('Feature extractor! ')

parser.add_argument("-img_dir", type=str, required=True, help="Input Image folder")
parser.add_argument("-data_dir", type=str, required=True, help="Dataset folder")
parser.add_argument("-set_type", type=list, default=["valid", "train", "test"], help='Select the dataset to dump')

parser.add_argument("-out_dir", type=str, required=True, help="Output folder")

parser.add_argument("-network", type=str, choices=["resnet", "vgg"], help="Use resnet/vgg network")
parser.add_argument("-resnet_version", type=int, default=152, choices=[50, 101, 152], help="Pick the resnet version [50/101/152]")
parser.add_argument("-ckpt", type=str, required=True, help="Path for network checkpoint: ")
parser.add_argument("-feature_name", type=str, default="", help="Pick the name of the network features default=(fc8 - block4)")

parser.add_argument("-mode", type=str, choices=["img", "crop"], help="Select to either dump the img/crop feature")
parser.add_argument("-subtract_mean", type=lambda x: bool(strtobool(x)), default="True", help="Preprocess the image by substracting the mean")
parser.add_argument("-img_size", type=int, default=224, help="image size (pixels)")
parser.add_argument("-crop_scale", type=float, default=1.1, help="crop scale around the bbox")
parser.add_argument("-batch_size", type=int, default=64, help="Batch size to extract features")

parser.add_argument("-gpu_ratio", type=float, default=0.95, help="How many GPU ram is required? (ratio)")
parser.add_argument("-no_thread", type=int, default=2, help="No thread to load batch")

args = parser.parse_args()

# define image
if args.subtract_mean:
    channel_mean = np.array([123.68, 116.779, 103.939])
else:
    channel_mean = None

# define the image loader
dataset_args = {"folder": args.data_dir,
                "dataset": args.dataset
                }


# define the image loader (raw vs crop)
if args.mode == "img":
    images = tf.placeholder(tf.float32, [None, args.img_size, args.img_size, 3], name='image')
    source = 'image'
    dataset_cstor = Dataset
    image_builder = RawImageBuilder(args.img_dir,
                                    height=args.img_size,
                                    width=args.img_size,
                                    channel=channel_mean)

    dataset_args["image_builder"] = image_builder

elif args.mode == "crop":
    images = tf.placeholder(tf.float32, [None, args.img_size, args.img_size, 3], name='crop')
    source = 'crop'
    dataset_cstor = CropDataset.load
    crop_builder = RawCropBuilder(args.img_dir,
                                  height=args.img_size,
                                  width=args.img_size,
                                  scale=args.crop_scale,
                                  channel=channel_mean)

    dataset_args["expand_objects"] = (args.mode == "crop_all")
    dataset_args["crop_builder"] = crop_builder
else:
    assert False, "Invalid mode: {}".format(args.mode)

# Define the output folder
out_file = "gw_{mode}_{network}_{feature_name}_{size}".format(
    mode=args.mode, network=args.network, feature_name=args.feature_name, size=args.img_size)

print("Create networks...")
if args.network == "resnet":
    ft_output = resnet.create_resnet(images,
                                     resnet_out=args.feature_name,
                                     resnet_version=args.resnet_version,
                                     is_training=False)

elif args.network == "vgg":
    _, end_points = vgg.vgg_16(images, is_training=False, dropout_keep_prob=1.0)
    ft_name = os.path.join("vgg_16", args.feature_name)
    ft_output = end_points[ft_name]
else:
    assert False, "Incorrect Network"

extract_features(
    img_input=images,
    ft_output=ft_output,
    dataset_cstor=dataset_cstor,
    dataset_args=dataset_args,
    batchifier_cstor=OracleBatchifier,
    out_dir=args.out_dir,
    set_type=args.set_type,
    network_ckpt=args.ckpt,
    batch_size=args.batch_size,
    no_threads=args.no_thread,
    gpu_ratio=args.gpu_ratio)
