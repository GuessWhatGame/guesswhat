from __future__ import unicode_literals

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator
import logging
import copy

def test_one_model(sess, dataset, cpu_pool, batch_size, network, batchifier, loss, listener=None):
    sources = network.get_sources(sess)
    evaluator = Evaluator(sources, network.scope_name, network=network, )
    iterator = Iterator(dataset, pool=cpu_pool, batch_size=batch_size, batchifier=batchifier)
    return evaluator.process(sess, iterator, outputs=loss, listener=listener)


def test_models(sess, dataset, cpu_pool, batch_size,
                oracle, oracle_batchifier,
                guesser, guesser_batchifier, guesser_listener,
                qgen, qgen_batchifier):

    # Retrieve logger
    logger = logging.getLogger()

    # Oracle:
    [oracle_loss, oracle_accuracy] = test_one_model(sess, dataset, cpu_pool, batch_size,
                                                    network=oracle,
                                                    batchifier=oracle_batchifier,
                                                    loss=[oracle.loss, oracle.accuracy])
    logger.info("Oracle test loss: {}".format(oracle_loss))
    logger.info("Oracle test accuracy: {}".format(oracle_accuracy))

    # Guesser:
    [guesser_loss] = test_one_model(sess, dataset, cpu_pool, batch_size,
                                    network=guesser,
                                    batchifier=guesser_batchifier,
                                    loss=[guesser.loss],
                                    listener=guesser_listener)
    logger.info("Guesser test loss: {}".format(guesser_loss))
    logger.info("Guesser test accuracy: {}".format(guesser_listener.accuracy()))

    # QGen:
    qgen_batchifier = copy.copy(qgen_batchifier)
    qgen_batchifier.supervised = True
    qgen_batchifier.generate = False
    [guesser_loss] = test_one_model(sess, dataset, cpu_pool, batch_size,
                                    network=qgen,
                                    batchifier=qgen_batchifier,
                                    loss=[qgen.ml_loss])
    logger.info("QGen test loss: {}".format(guesser_loss))


def compute_qgen_accuracy(sess, dataset, batchifier, looper, mode, cpu_pool, batch_size, store_games=False):

    logger = logging.getLogger()

    for m in mode:
        test_iterator = Iterator(dataset, pool=cpu_pool,
                                 batch_size=batch_size,
                                 batchifier=batchifier,
                                 shuffle=False,
                                 use_padding=True)

        [test_score, _] = looper.process(sess, test_iterator, mode=m, store_games=store_games)

        logger.info("Accuracy ({} - {}): {}".format(dataset.set, m, test_score))


# def compute_qgen_accuracy(sess, dataset, batchifier, evaluator, mode, tokenizer, save_path, cpu_pool, batch_size, store_games, dump_suffix):
#
#     logger = logging.getLogger()
#
#     for m in mode:
#         test_iterator = Iterator(dataset, pool=cpu_pool,
#                                  batch_size=batch_size,
#                                  batchifier=batchifier,
#                                  shuffle=False,
#                                  use_padding=True)
#         test_score = evaluator.process(sess, test_iterator, mode=m, store_games=store_games)
#
#         # Retrieve the generated games and dump them as a dataset
#         if store_games:
#             generated_dialogues = evaluator.get_storage()
#             dump_samples_into_dataset(generated_dialogues,
#                                       save_path=save_path,
#                                       tokenizer=tokenizer,
#                                       name=dump_suffix + "." + m)
#
#         logger.info("Accuracy ({} - {}): {}".format(dataset.set, m, test_score))