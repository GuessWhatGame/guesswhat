import argparse
import json
import os
import pickle
import tensorflow as tf


import guesswhat.data_provider as provider
from guesswhat.models.guesser.guesser_network import GuesserNetwork

import sys


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

config, env, save_path, exp_identifier, logger = provider.load_data_from_args(args, load_picture=True)

###############################
#  START TRAINING
#############################

logger.info('Building network..')
network = GuesserNetwork(config)
best_val_loss = 0

saver = tf.train.Saver()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    if args.from_checkpoint:
        saver.restore(sess, save_path.format(exp_identifier + '/params.ckpt'))
    else:
        sess.run(tf.global_variables_initializer())

    best_val_err = 1e5
    logs = []
    for t in range(0, config['optimizer']['no_epoch']):
        logger.info('Epoch {}..'.format(t + 1))

        iterator = provider.GameIterator(
            env.trainset,
            env.tokenizer,
            batch_size=config['optimizer']['batch_size'],
            shuffle=True,
            status=('success',))

        train_loss, train_err = network.train(sess, iterator)

        # Validation
        iterator = provider.GameIterator(
            env.validset,
            env.tokenizer,
            batch_size=config['optimizer']['batch_size'],
            shuffle=False,
            status=('success',))
        valid_loss, valid_err = network.evaluate(sess, iterator)

        logger.info("Training loss:  {}".format(train_loss))
        logger.info("Training error: {}".format(train_err))
        logger.info("Validation loss:  {}".format(valid_loss))
        logger.info("Validation error: {}".format(valid_err))

        logs.append({'epoch': t,
                     'train_loss': train_loss,
                     'train_err': train_err,
                     'valid_loss': valid_loss,
                     'valid_err': valid_err})

        pickle.dump(logs, open(save_path.format('logs.pkl'), 'wb'))

        if valid_err < best_val_err:
            best_val_loss = valid_loss
            saver.save(sess, save_path.format('params.ckpt'))

    # Calculate test error
    saver.restore(sess, save_path.format('params.ckpt'))

    iterator = provider.GameIterator(
        env.testset,
        env.tokenizer,
        batch_size=config['optimizer']['batch_size'],
        shuffle=False,
        status=('success',))

    test_loss, test_err = network.evaluate(sess, iterator)

    logger.info("Test loss: {}".format(test_loss))
    logger.info("Test error: {}".format(test_err))

    # Experiment done; write results to experiment database (jsonl file)
    with open(os.path.join(args.exp_dir, 'experiments.jsonl'), 'a') as f:
        exp = dict()
        exp['config'] = config
        exp['best_val_loss'] = best_val_loss
        exp['test_error'] = test_err
        exp['test_loss'] = test_loss
        exp['identifier'] = exp_identifier

        f.write(json.dumps(exp))
        f.write('\n')
