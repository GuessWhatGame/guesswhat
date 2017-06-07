import argparse
import collections
import json
import logging
import os
import pickle
from multiprocessing import Pool

import tensorflow as tf

from generic.data_provider.iterator import Iterator
from generic.tensorflow.evaluator import Evaluator
from generic.tensorflow.optimizer import create_optimizer

from guesswhat.data_provider.guesswhat_dataset import Dataset
from guesswhat.data_provider.guesser_batchifier import GuesserBatchifier
from guesswhat.data_provider.nlp_preprocessors import GWTokenizer
from generic.utils.config import load_config


from guesswhat.models.guesser.guesser_network import GuesserNetwork
from guesswhat.train.utils import get_img_loader, load_checkpoint



###############################
#  LOAD CONFIG
#############################

parser = argparse.ArgumentParser('Guesser network baseline!')

parser.add_argument("-data_dir", type=str, help="Directory with data")
parser.add_argument("-exp_dir", type=str, help="Directory in which experiments are stored")
parser.add_argument("-config", type=str, help="Configuration file")
parser.add_argument("-from_checkpoint", type=bool, help="Start from checkpoint?")
parser.add_argument("-gpu_ratio", type=float, default=1., help="How muany GPU ram is required? (ratio)")

args = parser.parse_args()
config, exp_identifier, save_path = load_config(args.config, args.exp_dir)
logger = logging.getLogger()


###############################
#  LOAD DATA
#############################

# Load image
logger.info('Loading images..')
image_loader = get_img_loader(config, 'image')
crop_loader = get_img_loader(config, 'crop')

# Load data
logger.info('Loading data..')
trainset = Dataset(args.data_dir, "train", image_loader, crop_loader)
validset = Dataset(args.data_dir, "valid", image_loader, crop_loader)
testset = Dataset(args.data_dir, "test", image_loader, crop_loader)

# Load dictionary
logger.info('Loading dictionary..')
tokenizer = GWTokenizer(os.path.join(args.data_dir, 'dict.json'))

# Build Network
logger.info('Building network..')
network = GuesserNetwork(config, num_words=tokenizer.no_words)

# Build Optimizer
logger.info('Building optimizer..')
optimizer, outputs = create_optimizer(network, network.loss, config)


###############################
#  START  TRAINING
#############################

# Load config
batch_size = config['optimizer']['batch_size']
no_epoch = config["optimizer"]["no_epoch"]

# create a saver to store/load checkpoint
saver = tf.train.Saver()

#CPU/GPU option
cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

    sources = network.get_sources(sess)
    logger.info("Sources: " + ', '.join(sources))

    sess.run(tf.global_variables_initializer())
    start_epoch = load_checkpoint(sess, saver, args, save_path)

    best_val_err = 1e5
    best_train_err = None

    # create training tools
    evaluator = Evaluator(sources, network.scope_name)
    batchifier = GuesserBatchifier(tokenizer, sources, **config['model']['crop'])

    for t in range(start_epoch, no_epoch):
        logger.info('Epoch {}..'.format(t + 1))

        train_iterator = Iterator(trainset,
                                  batch_size=batch_size, pool=cpu_pool,
                                  batchifier=batchifier,
                                  shuffle=True)
        train_loss, train_error = evaluator.process(sess, train_iterator, outputs=outputs + [optimizer])

        valid_iterator = Iterator(validset, pool=cpu_pool,
                                  batch_size=batch_size,
                                  batchifier=batchifier,
                                  shuffle=False)
        valid_loss, valid_error = evaluator.process(sess, valid_iterator, outputs=outputs)

        logger.info("Training loss: {}".format(train_loss))
        logger.info("Training error: {}".format(train_error))
        logger.info("Validation loss: {}".format(valid_loss))
        logger.info("Validation error: {}".format(valid_error))

        if valid_error < best_val_err:
            best_train_err = train_error
            best_val_err = valid_error
            saver.save(sess, save_path.format('params.ckpt'))
            logger.info("Guesser checkpoint saved...")

        pickle.dump({'epoch': t}, open(save_path.format('status.pkl'), 'wb'))

    saver.restore(sess, save_path.format('params.ckpt'))

    test_iterator = Iterator(testset, pool=cpu_pool,
                             batch_size=batch_size,
                             batchifier=batchifier,
                             shuffle=True)
    [test_loss, test_error] = evaluator.process(sess, test_iterator, outputs)