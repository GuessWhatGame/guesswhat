import argparse
import logging
import os
from multiprocessing import Pool
from distutils.util import strtobool

import tensorflow as tf

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator
from generic.tf_utils.optimizer import create_optimizer
from generic.tf_utils.ckpt_loader import load_checkpoint, create_resnet_saver
from generic.utils.config import load_config
from generic.utils.file_handlers import pickle_dump
from generic.data_provider.image_loader import get_img_builder

from guesswhat.data_provider.guesswhat_dataset import Dataset
from guesswhat.data_provider.questioner_batchifier import QuestionerBatchifier
from guesswhat.data_provider.guesswhat_tokenizer import GWTokenizer
from guesswhat.models.guesser.guesser_network import GuesserNetwork


if __name__ == '__main__':

    ###############################
    #  LOAD CONFIG
    #############################

    parser = argparse.ArgumentParser('Guesser network baseline!')

    parser.add_argument("-data_dir", type=str, help="Directory with data")
    parser.add_argument("-exp_dir", type=str, help="Directory in which experiments are stored")
    parser.add_argument("-img_dir", type=str, help='Directory with images')
    parser.add_argument("-config", type=str, help="Configuration file")
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Dictionary file name")
    parser.add_argument("-load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
    parser.add_argument("-continue_exp", lambda x: bool(strtobool(x)), default="False", help="Continue previously started experiment?")
    parser.add_argument("-gpu_ratio", type=float, default=1., help="How many GPU ram is required? (ratio)")
    parser.add_argument("-no_thread", type=int, default=1, help="No thread to load batch")


    args = parser.parse_args()
    config, exp_identifier, save_path = load_config(args.config, args.exp_dir)
    logger = logging.getLogger()

    ###############################
    #  LOAD DATA
    #############################

    image_builder, crop_builder = None, None

    # Load image
    logger.info('Loading images..')
    use_resnet = False
    if 'image' in config['model']:
        logger.info('Loading images..')
        image_builder = get_img_builder(config['model']['image'], args.img_dir)
        use_resnet = image_builder.is_raw_image()

        assert False, "Guesser + Image is not yet available"


    # Load data
    logger.info('Loading data..')
    trainset = Dataset(args.data_dir, "train", image_builder, crop_builder)
    validset = Dataset(args.data_dir, "valid", image_builder, crop_builder)
    testset = Dataset(args.data_dir, "test", image_builder, crop_builder)

    # Load dictionary
    logger.info('Loading dictionary..')
    tokenizer = GWTokenizer(os.path.join(args.data_dir, args.dict_file))

    # Build Network
    logger.info('Building network..')
    network = GuesserNetwork(config['model'], num_words=tokenizer.no_words)

    # Build Optimizer
    logger.info('Building optimizer..')
    optimizer, outputs = create_optimizer(network, config)

    ###############################
    #  START  TRAINING
    #############################

    # Load config
    batch_size = config['optimizer']['batch_size']
    no_epoch = config["optimizer"]["no_epoch"]

    # create a saver to store/load checkpoint
    saver = tf.train.Saver()

    # Retrieve only resnet variabes
    if use_resnet:
        resnet_saver = create_resnet_saver([network])

    # CPU/GPU option
    cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

        sources = network.get_sources(sess)
        logger.info("Sources: " + ', '.join(sources))

        sess.run(tf.global_variables_initializer())
        start_epoch = load_checkpoint(sess, saver, args, save_path)

        best_val_err = 0
        best_train_err = None

        # create training tools
        evaluator = Evaluator(sources, network.scope_name, network=network, tokenizer=tokenizer)
        batchifier = QuestionerBatchifier(tokenizer, sources, status=('success',))

        for t in range(start_epoch, no_epoch):
            logger.info('Epoch {}..'.format(t + 1))

            train_iterator = Iterator(trainset,
                                      batch_size=batch_size, pool=cpu_pool,
                                      batchifier=batchifier,
                                      shuffle=True)
            train_loss, train_accuracy = evaluator.process(sess, train_iterator, outputs=outputs + [optimizer])

            valid_iterator = Iterator(validset, pool=cpu_pool,
                                      batch_size=batch_size*2,
                                      batchifier=batchifier,
                                      shuffle=False)
            valid_loss, valid_accuracy = evaluator.process(sess, valid_iterator, outputs=outputs)

            logger.info("Training loss: {}".format(train_loss))
            logger.info("Training error: {}".format(1-train_accuracy))
            logger.info("Validation loss: {}".format(valid_loss))
            logger.info("Validation error: {}".format(1-valid_accuracy))

            if valid_accuracy > best_val_err:
                best_train_err = train_accuracy
                best_val_err = valid_accuracy
                saver.save(sess, save_path.format('params.ckpt'))
                logger.info("Guesser checkpoint saved...")

                pickle_dump({'epoch': t}, save_path.format('status.pkl'))

        # Load early stopping
        saver.restore(sess, save_path.format('params.ckpt'))
        test_iterator = Iterator(testset, pool=cpu_pool,
                                 batch_size=batch_size,
                                 batchifier=batchifier,
                                 shuffle=True)
        [test_loss, test_accuracy] = evaluator.process(sess, test_iterator, outputs)

        logger.info("Testing loss: {}".format(test_loss))
        logger.info("Testing error: {}".format(1-test_accuracy))
