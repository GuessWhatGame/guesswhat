import argparse
import logging
import os
import collections

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
from generic.data_provider.nlp_utils import GloveEmbeddings
from generic.utils.thread_pool import create_cpu_pool

from guesswhat.data_provider.guesswhat_dataset import Dataset
from guesswhat.data_provider.oracle_batchifier import OracleBatchifier, BatchifierSplitMode
from guesswhat.data_provider.guesswhat_tokenizer import GWTokenizer
from guesswhat.data_provider.guesswhat_dataset import dump_oracle
from guesswhat.models.oracle.oracle_factory import create_oracle

from guesswhat.train.eval_listener import OracleListener

if __name__ == '__main__':

    ###############################
    #  LOAD CONFIG
    #############################

    parser = argparse.ArgumentParser('Oracle network baseline!')

    parser.add_argument("-data_dir", type=str, help="Directory with data")
    parser.add_argument("-out_dir", type=str, help="Directory in which experiments are stored")
    parser.add_argument("-config", type=str, help='Config file')
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Dictionary file name")
    parser.add_argument("-glove_file", type=str, default="glove_dict.pkl", help="Glove file name")
    parser.add_argument("-img_dir", type=str, help='Directory with images')
    parser.add_argument("-crop_dir", type=str, help='Directory with crops')
    parser.add_argument("-load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
    parser.add_argument("-continue_exp", type=lambda x: bool(strtobool(x)), default="False", help="Continue previously started experiment?")
    parser.add_argument("-gpu_ratio", type=float, default=0.95, help="How many GPU ram is required? (ratio)")
    parser.add_argument("-no_thread", type=int, default=2, help="No thread to load batch")
    parser.add_argument("-no_games_to_load", type=int, default=float("inf"), help="No games to use during training Default : all")

    args = parser.parse_args()

    config, exp_identifier, save_path = load_config(args.config, args.out_dir)
    logger = logging.getLogger()

    # Load config
    resnet_version = config['model']["image"].get('resnet_version', 50)
    finetune = config["model"]["image"].get('finetune', list())
    split_mode = BatchifierSplitMode.from_string(config["model"]["question"]["input_type"])
    batch_size = config['optimizer']['batch_size']
    no_epoch = config["optimizer"]["no_epoch"]

    ###############################
    #  LOAD DATA
    #############################

    # Load image
    image_builder, crop_builder = None, None
    use_resnet, use_process = False, False
    if config["model"]['inputs'].get('image', False):
        logger.info('Loading images..')
        image_builder = get_img_builder(config['model']['image'], args.img_dir)
        use_resnet = image_builder.is_raw_image()

    if config["model"]['inputs'].get('crop', False):
        logger.info('Loading crops..')
        crop_builder = get_img_builder(config['model']['crop'], args.crop_dir, is_crop=True)
        use_resnet = crop_builder.is_raw_image()

    # Load data
    logger.info('Loading data..')
    trainset = Dataset(args.data_dir, "train", image_builder, crop_builder, args.no_games_to_load)
    validset = Dataset(args.data_dir, "valid", image_builder, crop_builder, args.no_games_to_load)
    testset = Dataset(args.data_dir, "test", image_builder, crop_builder, args.no_games_to_load)

    # Load dictionary
    logger.info('Loading dictionary..')
    tokenizer = GWTokenizer(args.dict_file)

    # Load glove
    glove = None
    if config["model"]["question"]['glove']:
        logger.info('Loading glove..')
        glove = GloveEmbeddings(args.glove_file)

    # Build Network
    logger.info('Building network..')
    network = create_oracle(config["model"], num_words=tokenizer.no_words)

    # Build Optimizer
    logger.info('Building optimizer..')
    optimizer, outputs = create_optimizer(network, config["optimizer"], finetune=finetune)

    ###############################
    #  START  TRAINING
    #############################

    # create a saver to store/load checkpoint
    saver = tf.train.Saver()
    resnet_saver = None

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
        if use_resnet:
            resnet_version = config['model']["image"]['resnet_version']
            resnet_saver.restore(sess, os.path.join(args.data_dir, 'resnet_v1_{}.ckpt'.format(resnet_version)))

        start_epoch = load_checkpoint(sess, saver, args, save_path)

        best_val_acc = 0
        best_train_acc = None

        # create training tools
        evaluator = Evaluator(sources, network.scope_name, network=network, tokenizer=tokenizer)
        batchifier = OracleBatchifier(tokenizer, sources, glove=glove, status=config['status'],
                                      split_mode=split_mode)

        for t in range(start_epoch, no_epoch):
            logger.info('Epoch {}..'.format(t + 1))

            # Create cpu pools (at each iteration otherwise threads may become zombie - python bug)
            cpu_pool = create_cpu_pool(args.no_thread, use_process=use_process)

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

            logger.info("Training loss   : {}".format(train_loss))
            logger.info("Training error  : {}".format(1-train_accuracy))
            logger.info("Validation loss : {}".format(valid_loss))
            logger.info("Validation error: {}".format(1-valid_accuracy))

            if valid_accuracy > best_val_acc:
                best_train_acc = train_accuracy
                best_val_acc = valid_accuracy
                saver.save(sess, save_path.format('params.ckpt'))
                logger.info("Oracle checkpoint saved...")

                pickle_dump({'epoch': t}, save_path.format('status.pkl'))

        # Load early stopping
        saver.restore(sess, save_path.format('params.ckpt'))

        # Create     Listener
        oracle_listener = OracleListener(tokenizer=tokenizer, require=network.prediction)
        # batchifier.status = ["success", "failure", "incomplete"]

        cpu_pool = create_cpu_pool(args.no_thread, use_process=use_process)
        test_iterator = Iterator(testset, pool=cpu_pool,
                                 batch_size=batch_size*2,
                                 batchifier=batchifier,
                                 shuffle=False)
        [test_loss, test_accuracy] = evaluator.process(sess, test_iterator, outputs)

        logger.info("Testing loss: {}".format(test_loss))
        logger.info("Testing error: {}".format(1-test_accuracy))
