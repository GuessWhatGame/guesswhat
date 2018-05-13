import argparse
import os
from multiprocessing import Pool
import logging
from distutils.util import strtobool

import tensorflow as tf

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator
from generic.tf_utils.optimizer import create_optimizer
from generic.data_provider.image_loader import get_img_builder
from generic.utils.config import load_config, get_config_from_xp

from guesswhat.models.oracle.oracle_network import OracleNetwork
from guesswhat.models.qgen.qgen_lstm_network import QGenNetworkLSTM
from guesswhat.models.guesser.guesser_network import GuesserNetwork
from guesswhat.models.looper.basic_looper import BasicLooper

from guesswhat.models.qgen.qgen_wrapper import QGenWrapper
from guesswhat.models.oracle.oracle_wrapper import OracleWrapper
from guesswhat.models.guesser.guesser_wrapper import GuesserWrapper

from guesswhat.data_provider.guesswhat_dataset import Dataset
from guesswhat.data_provider.looper_batchifier import LooperBatchifier
from guesswhat.data_provider.guesswhat_tokenizer import GWTokenizer

from guesswhat.train.utils import test_model, compute_qgen_accuracy


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Question generator (policy gradient baseline))')

    parser.add_argument("-data_dir", type=str, required=True, help="Directory with data")
    parser.add_argument("-exp_dir", type=str, required=True, help="Directory in which experiments are stored")
    parser.add_argument("-img_dir", type=str, help='Directory with images')
    parser.add_argument("-crop_dir", type=str, help='Directory with images')
    parser.add_argument("-config", type=str, required=True, help='Config file')
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Dictionary file name")

    parser.add_argument("-networks_dir", type=str, help="Directory with pretrained networks")
    parser.add_argument("-oracle_identifier", type=str, required=True , help='Oracle identifier')  # Use checkpoint id instead?
    parser.add_argument("-qgen_identifier", type=str, required=True, help='Qgen identifier')
    parser.add_argument("-guesser_identifier", type=str, required=True, help='Guesser identifier')

    parser.add_argument("-continue_exp", type=bool, default=True, help="Continue previously started experiment?")
    #parser.add_argument("-from_checkpoint", type=str, help="Start from checkpoint?")
    parser.add_argument("-skip_training",  type=lambda x: bool(strtobool(x)), default="False", help="Start from checkpoint?")
    parser.add_argument("-evaluate_all", type=lambda x: bool(strtobool(x)), default="False", help="Evaluate sampling, greedy and BeamSearch?")  #TODO use an input list
    parser.add_argument("-store_games", type=lambda x: bool(strtobool(x)), default="True", help="Should we dump the game at evaluation times")


    parser.add_argument("-gpu_ratio", type=float, default=0.95, help="How muany GPU ram is required? (ratio)")
    parser.add_argument("-no_thread", type=int, default=1, help="No thread to load batch")

    args = parser.parse_args()

    loop_config, exp_identifier, save_path = load_config(args.config, args.exp_dir)

    # Load all  networks configs
    oracle_config = get_config_from_xp(os.path.join(args.networks_dir, "oracle"), args.oracle_identifier)
    guesser_config = get_config_from_xp(os.path.join(args.networks_dir, "guesser"), args.guesser_identifier)
    qgen_config = get_config_from_xp(os.path.join(args.networks_dir, "qgen"), args.qgen_identifier)


    logger = logging.getLogger()

    ###############################
    #  LOAD DATA
    #############################

    # Load image
    logger.info('Loading images..')
    image_builder = get_img_builder(qgen_config['model']['image'], args.img_dir)

    crop_builder = None
    if oracle_config['inputs'].get('crop', False):
        logger.info('Loading crops..')
        crop_builder = get_img_builder(oracle_config['model']['crop'], args.crop_dir, is_crop=True)

    # Load data
    logger.info('Loading data..')
    trainset = Dataset(args.data_dir, "train", image_builder, crop_builder)
    validset = Dataset(args.data_dir, "valid", image_builder, crop_builder)
    testset = Dataset(args.data_dir, "test", image_builder, crop_builder)

    # Load dictionary
    logger.info('Loading dictionary..')
    tokenizer = GWTokenizer(os.path.join(args.data_dir, args.dict_file))

    ###############################
    #  LOAD NETWORKS
    #############################

    logger.info('Building networks..')

    qgen_network = QGenNetworkLSTM(qgen_config["model"], num_words=tokenizer.no_words, policy_gradient=True)
    qgen_var = [v for v in tf.global_variables() if "qgen" in v.name] # and 'rl_baseline' not in v.name
    qgen_saver = tf.train.Saver(var_list=qgen_var)

    oracle_network = OracleNetwork(oracle_config, num_words=tokenizer.no_words)
    oracle_var = [v for v in tf.global_variables() if "oracle" in v.name]
    oracle_saver = tf.train.Saver(var_list=oracle_var)

    guesser_network = GuesserNetwork(guesser_config["model"], num_words=tokenizer.no_words)
    guesser_var = [v for v in tf.global_variables() if "guesser" in v.name]
    guesser_saver = tf.train.Saver(var_list=guesser_var)

    loop_saver = tf.train.Saver(allow_empty=False)

    ###############################
    #  REINFORCE OPTIMIZER
    #############################

    logger.info('Building optimizer..')

    pg_variables = [v for v in tf.trainable_variables() if "qgen" in v.name and 'rl_baseline' not in v.name]
    baseline_variables = [v for v in tf.trainable_variables() if "qgen" in v.name and 'rl_baseline' in v.name]

    pg_optimize, _ = create_optimizer(qgen_network, loop_config["optimizer"],
                                   var_list=pg_variables,
                                   optim_cst=tf.train.GradientDescentOptimizer,
                                   loss=qgen_network.policy_gradient_loss)
    baseline_optimize, _= create_optimizer(qgen_network, loop_config["optimizer"],
                                        var_list=baseline_variables,
                                        optim_cst=tf.train.GradientDescentOptimizer,
                                        apply_update_ops=False,
                                        loss=qgen_network.baseline_loss)

    optimizer = [pg_optimize, baseline_optimize]


    ###############################
    #  START TRAINING
    #############################

    # Load config
    batch_size = loop_config['optimizer']['batch_size']
    no_epoch = loop_config["optimizer"]["no_epoch"]

    mode_to_evaluate = ["greedy"]
    if args.evaluate_all:
        mode_to_evaluate = ["greedy", "sampling", "beam_search"]

    # create a saver to store/load checkpoint
    saver = tf.train.Saver()

    # CPU/GPU option
    cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        ###############################
        #  LOAD PRE-TRAINED NETWORK
        #############################

        sess.run(tf.global_variables_initializer())
        if args.continue_exp: # TODO only reload qgen ckpt
            qgen_saver.restore(sess, save_path.format('params.ckpt'))
        else:
            qgen_var_supervized = [v for v in tf.global_variables() if "qgen" in v.name and 'rl_baseline' not in v.name] 
            qgen_loader_supervized = tf.train.Saver(var_list=qgen_var_supervized)
            qgen_loader_supervized.restore(sess, os.path.join(args.networks_dir, 'qgen', args.qgen_identifier, 'params.ckpt'))

        oracle_saver.restore(sess, os.path.join(args.networks_dir, 'oracle', args.oracle_identifier, 'params.ckpt'))
        guesser_saver.restore(sess, os.path.join(args.networks_dir, 'guesser', args.guesser_identifier, 'params.ckpt'))

        # create training tools
        loop_sources = qgen_network.get_sources(sess)
        logger.info("Sources: " + ', '.join(loop_sources))

        evaluator = Evaluator(loop_sources, qgen_network.scope_name, network=qgen_network, tokenizer=tokenizer)

        train_batchifier = LooperBatchifier(tokenizer,  generate_new_games=True)
        eval_batchifier = LooperBatchifier(tokenizer, generate_new_games=False)

        # Initialize the looper to eval/train the game-simulation

        oracle_wrapper = OracleWrapper(oracle_network, tokenizer)
        guesser_wrapper = GuesserWrapper(guesser_network)
        qgen_network.build_sampling_graph(qgen_config["model"], tokenizer=tokenizer, max_length=loop_config['loop']['max_depth'])
        qgen_wrapper = QGenWrapper(qgen_network, tokenizer,
                                   max_length=loop_config['loop']['max_depth'],
                                   k_best=loop_config['loop']['beam_k_best'])

        looper_evaluator = BasicLooper(loop_config,
                                       oracle_wrapper=oracle_wrapper,
                                       guesser_wrapper=guesser_wrapper,
                                       qgen_wrapper=qgen_wrapper,
                                       tokenizer=tokenizer,
                                       batch_size=loop_config["optimizer"]["batch_size"])


        # Compute the initial scores
        logger.info(">>>-------------- INITIAL SCORE ---------------------<<<")

        logger.info(">>>  Initial models  <<<")
        test_model(sess, testset, cpu_pool=cpu_pool, tokenizer=tokenizer,
                   oracle=oracle_network,guesser=guesser_network, qgen=qgen_network,
                   batch_size=batch_size*2, logger=logger)

        logger.info(">>>  New Objects  <<<")
        compute_qgen_accuracy(sess, trainset, batchifier=train_batchifier, evaluator=looper_evaluator, tokenizer=tokenizer,
                              mode=mode_to_evaluate, save_path=save_path, cpu_pool=cpu_pool, batch_size=batch_size,
                              store_games=args.store_games, dump_suffix="init.new_object")

        logger.info(">>>  New Games  <<<")
        compute_qgen_accuracy(sess, testset, batchifier=eval_batchifier, evaluator=looper_evaluator, tokenizer=tokenizer,
                              mode=mode_to_evaluate, save_path=save_path, cpu_pool=cpu_pool, batch_size=batch_size,
                              store_games=args.store_games, dump_suffix="init.new_games")
        logger.info(">>>------------------------------------------------<<<")

        if args.skip_training:
            logger.info("Skip training...")
            exit(0)

        logs = []
        # Start training
        final_val_score = 0.
        for epoch in range(no_epoch):

            logger.info("Epoch {}/{}".format(epoch, no_epoch))

            train_iterator = Iterator(trainset, batch_size=batch_size,
                                      pool=cpu_pool,
                                      batchifier=train_batchifier,
                                      use_padding=True)
            train_score = looper_evaluator.process(sess, train_iterator,
                                                   optimizer=optimizer,
                                                   mode="sampling")

            valid_iterator = Iterator(validset, pool=cpu_pool,
                                      batch_size=batch_size,
                                      batchifier=eval_batchifier,
                                      shuffle=False,
                                      use_padding=True)
            val_score = looper_evaluator.process(sess, valid_iterator, mode="sampling")

            logger.info("Accuracy (train - sampling) : {}".format(train_score))
            logger.info("Accuracy (valid - sampling) : {}".format(val_score))

            if val_score > final_val_score:
                logger.info("Save checkpoint...")
                final_val_score = val_score
                loop_saver.save(sess, save_path.format('params.ckpt'))


        # Compute the test score with early stopping
        logger.info(">>>-------------- FINAL SCORE ---------------------<<<")
        loop_saver.restore(sess, save_path.format('params.ckpt'))

        logger.info(">>>  New Objects  <<<")
        compute_qgen_accuracy(sess, trainset, batchifier=train_batchifier, evaluator=looper_evaluator, tokenizer=tokenizer,
                              mode=mode_to_evaluate, save_path=save_path, cpu_pool=cpu_pool, batch_size=batch_size,
                              store_games=args.store_games, dump_suffix="final.new_object")

        logger.info(">>>  New Games  <<<")
        compute_qgen_accuracy(sess, testset, batchifier=eval_batchifier, evaluator=looper_evaluator, tokenizer=tokenizer,
                              mode=mode_to_evaluate, save_path=save_path, cpu_pool=cpu_pool, batch_size=batch_size,
                              store_games=args.store_games, dump_suffix="final.new_games")
        logger.info(">>>------------------------------------------------<<<")

