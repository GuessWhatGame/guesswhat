
import argparse
import os
from multiprocessing import Pool
import logging
import random
import copy
import tensorflow as tf

from generic.data_provider.iterator import BasicIterator
from generic.tf_utils.evaluator import Evaluator
from generic.data_provider.image_loader import get_img_builder

from guesswhat.models.oracle.oracle_network import OracleNetwork
from guesswhat.models.qgen.qgen_lstm_network import QGenNetworkLSTM
from guesswhat.models.guesser.guesser_network import GuesserNetwork
from guesswhat.models.looper.basic_looper import BasicLooper

from guesswhat.models.qgen.qgen_wrapper import QGenWrapper, QGenUserWrapper
from guesswhat.models.oracle.oracle_wrapper import OracleWrapper, OracleUserWrapper
from guesswhat.models.guesser.guesser_wrapper import GuesserWrapper, GuesserUserWrapper


from guesswhat.data_provider.guesswhat_dataset import Dataset

from guesswhat.data_provider.looper_batchifier import LooperBatchifier

from guesswhat.data_provider.guesswhat_tokenizer import GWTokenizer
from generic.utils.config import load_config, get_config_from_xp


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Question generator (policy gradient baseline))')

    parser.add_argument("-data_dir", type=str, required=True, help="Directory with data")
    parser.add_argument("-img_dir", type=str, help='Directory with images to feed networks')
    parser.add_argument("-img_raw_dir", type=str, help='Directory with images to display')
    parser.add_argument("-crop_dir", type=str, help='Directory with crops')
    parser.add_argument("-exp_dir", type=str, required=False, help="Directory to output dialogue")
    parser.add_argument("-config", type=str, required=True, help='Config file')
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Dictionary file name")

    parser.add_argument("-networks_dir", type=str, help="Directory with pretrained networks")
    parser.add_argument("-oracle_identifier", type=str, default="156cb3d352b97ba12ffd6cf547281ae2", required=False , help='Oracle identifier - if none: user must be the oracle')  # Use checkpoint id instead?
    parser.add_argument("-qgen_identifier", type=str, default="7b24d8b68f94bde9774cd9555584fd93", required=False, help='Qgen identifier - if none: user must be the Qgen')
    parser.add_argument("-guesser_identifier", type=str, required=False, help='Guesser identifier - if none: user must be the guesser')

    parser.add_argument("-gpu_ratio", type=float, default=0.95, help="How many GPU ram is required? (ratio)")

    args = parser.parse_args()

    eval_config, exp_identifier, save_path = load_config(args.config, args.exp_dir)

    # Load all  networks configs
    logger = logging.getLogger()

    ###############################
    #  LOAD DATA
    #############################

    # Load image
    logger.info('Loading images..')
    image_builder = get_img_builder(eval_config['image'], args.img_dir)

    crop_builder = None
    if eval_config.get('crop', False):
        logger.info('Loading crops..')
        crop_builder = get_img_builder(eval_config['crop'], args.crop_dir, is_crop=True)

    # Load data
    logger.info('Loading data..')
    trainset = Dataset(args.data_dir, "train", image_builder, crop_builder)
    validset = Dataset(args.data_dir, "valid", image_builder, crop_builder)
    testset = Dataset(args.data_dir, "test", image_builder, crop_builder)

    dataset, dummy_dataset = trainset, validset
    dataset.games = trainset.games + validset.games + testset.games
    dummy_dataset.games = []

    # hack dataset to only keep one game by image
    image_id_set = {}
    games = []
    for game in dataset.games:
        if game.image.id not in image_id_set:
            games.append(game)
            image_id_set[game.image.id] = 1
    dataset.games = games


    # Load dictionary
    logger.info('Loading dictionary..')
    tokenizer = GWTokenizer(os.path.join(args.data_dir, args.dict_file))



    ###############################
    #  START TRAINING
    #############################

    # CPU/GPU option
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        ###############################
        #  LOAD NETWORKS
        #############################

        if args.oracle_identifier is not None:
            oracle_config = get_config_from_xp(os.path.join(args.networks_dir, "oracle"), args.oracle_identifier)

            oracle_network = OracleNetwork(oracle_config, num_words=tokenizer.no_words)
            oracle_var = [v for v in tf.global_variables() if "oracle" in v.name]
            oracle_saver = tf.train.Saver(var_list=oracle_var)

            oracle_saver.restore(sess, os.path.join(args.networks_dir, 'oracle', args.oracle_identifier, 'params.ckpt'))

            oracle_wrapper = OracleWrapper(oracle_network, tokenizer)

        else:
            oracle_wrapper = OracleUserWrapper(tokenizer)
            logger.info("No Oracle was registered >>> use user input")

        if args.guesser_identifier is not None:
            guesser_config = get_config_from_xp(os.path.join(args.networks_dir, "guesser"), args.guesser_identifier)

            guesser_network = GuesserNetwork(guesser_config["model"], num_words=tokenizer.no_words)
            guesser_var = [v for v in tf.global_variables() if "guesser" in v.name]
            guesser_saver = tf.train.Saver(var_list=guesser_var)

            guesser_saver.restore(sess, os.path.join(args.networks_dir, 'guesser', args.guesser_identifier, 'params.ckpt'))

            guesser_wrapper = GuesserWrapper(guesser_network)

        else:
            guesser_wrapper = GuesserUserWrapper(tokenizer, img_raw_dir=args.img_raw_dir)
            logger.info("No Guesser was registered >>> use user input")

        if args.qgen_identifier is not None:
            qgen_config = get_config_from_xp(os.path.join(args.networks_dir, "qgen"), args.qgen_identifier)

            qgen_network = QGenNetworkLSTM(qgen_config["model"], num_words=tokenizer.no_words, policy_gradient=False)
            qgen_var = [v for v in tf.global_variables() if "qgen" in v.name]  # and 'rl_baseline' not in v.name
            qgen_saver = tf.train.Saver(var_list=qgen_var)

            qgen_saver.restore(sess, os.path.join(args.networks_dir, 'qgen', args.qgen_identifier, 'params.ckpt'))

            qgen_network.build_sampling_graph(qgen_config["model"], tokenizer=tokenizer, max_length=eval_config['loop']['max_depth'])
            qgen_wrapper = QGenWrapper(qgen_network, tokenizer,
                                       max_length=eval_config['loop']['max_depth'],
                                       k_best=eval_config['loop']['beam_k_best'])

        else:
            qgen_wrapper = QGenUserWrapper(tokenizer)
            logger.info("No QGen was registered >>> use user input")




        looper_evaluator = BasicLooper(eval_config,
                                       oracle_wrapper=oracle_wrapper,
                                       guesser_wrapper=guesser_wrapper,
                                       qgen_wrapper=qgen_wrapper,
                                       tokenizer=tokenizer,
                                       batch_size=1)



        logs = []
        # Start training
        final_val_score = 0.

        batchifier = LooperBatchifier(tokenizer, generate_new_games=False)
        while True:

            # Start new game
            while True:
                id_str = input('Do you want to play a new game? (Yes/No) -->  ').lower()
                if id_str == "y" or id_str == "yes": break
                elif id_str == "n" or id_str == "no": exit(0)

            # Pick id image
            image_id = 0
            while True:
                id_str = int(input('What is the image id you want to select? (-1 for random id) -->  '))
                if id_str in image_id_set:
                    image_id = id_str
                    break
                elif id_str == -1:
                    image_id = random.choice(list(image_id_set.keys()))
                    break
                else:
                    print("Could not find the following image id: {}".format(id_str))

            game = [g for g in dataset.games if g.image.id == image_id][0]
            game = copy.deepcopy(game)
            print("Selecting image {}".format(game.image.filename))


            # Pick id object

            print("Available objects")
            for i, obj in enumerate(game.objects):
                print(" -", i, ":", obj.category, "\t", obj.bbox)
            print("Type '(S)how' to display the image with the object")

            while True:
                id_str = input('Which object id do you want to select? (-1 for random id) -->  ')

                if id_str == "S" or id_str.lower() == "show":
                    game.show(img_raw_dir=args.img_raw_dir, display_index=True)
                    continue

                id_str = int(id_str)
                if 0 <= id_str < len(game.objects):
                    object_index = id_str
                    object_id = game.objects[object_index].id
                    break
                elif id_str == -1:
                    object_id = random.choice(game.objects).id
                    break

                else:
                    print("Could not find the following object index: {}".format(id_str))

            game.object_id = object_id
            dummy_dataset.games = [game]

            iterator = BasicIterator(dummy_dataset, batch_size=1, batchifier=batchifier)
            success = looper_evaluator.process(sess, iterator, mode="greedy")


