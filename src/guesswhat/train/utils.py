from __future__ import unicode_literals

from guesswhat.data_provider.questioner_batchifier import QuestionerBatchifier
from guesswhat.data_provider.oracle_batchifier import OracleBatchifier
from guesswhat.data_provider.guesswhat_dataset import OracleDataset

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator
import logging
from guesswhat.data_provider.guesswhat_dataset import dump_samples_into_dataset

def test_oracle(sess, testset, tokenizer, oracle, cpu_pool, batch_size, logger):

    oracle_dataset = OracleDataset(testset)
    oracle_sources = oracle.get_sources(sess)
    oracle_evaluator = Evaluator(oracle_sources, oracle.scope_name, network=oracle, tokenizer=tokenizer)
    oracle_batchifier = OracleBatchifier(tokenizer, oracle_sources, status=('success',))
    oracle_iterator = Iterator(oracle_dataset, pool=cpu_pool,
                             batch_size=batch_size,
                             batchifier=oracle_batchifier)
    [oracle_loss, oracle_error] = oracle_evaluator.process(sess, oracle_iterator, [oracle.loss, oracle.error])

    logger.info("Oracle test loss: {}".format(oracle_loss))
    logger.info("Oracle test error: {}".format(oracle_error))


def test_guesser(sess, testset, tokenizer, guesser, cpu_pool, batch_size, logger):
    guesser_sources = guesser.get_sources(sess)
    guesser_evaluator = Evaluator(guesser_sources, guesser.scope_name, network=guesser, tokenizer=tokenizer)
    guesser_batchifier = QuestionerBatchifier(tokenizer, guesser_sources, status=('success',))
    guesser_iterator = Iterator(testset, pool=cpu_pool,
                             batch_size=batch_size,
                             batchifier=guesser_batchifier)
    [guesser_loss, guesser_error] = guesser_evaluator.process(sess, guesser_iterator, [guesser.loss, guesser.error])
    logger.info("Guesser test loss: {}".format(guesser_loss))
    logger.info("Guesser test error: {}".format(guesser_error))


def test_qgen(sess, testset, tokenizer, qgen, cpu_pool, batch_size, logger):
    qgen_sources = qgen.get_sources(sess)
    qgen_evaluator = Evaluator(qgen_sources, qgen.scope_name, network=qgen, tokenizer=tokenizer)
    qgen_batchifier = QuestionerBatchifier(tokenizer, qgen_sources, status=('success',))
    qgen_iterator = Iterator(testset, pool=cpu_pool,
                                 batch_size=batch_size,
                                 batchifier=qgen_batchifier)
    [qgen_loss] = qgen_evaluator.process(sess, qgen_iterator, outputs=[qgen.ml_loss])
    logger.info("QGen test loss: {}".format(qgen_loss))

def test_model(sess, testset, tokenizer, oracle, guesser, qgen, cpu_pool, batch_size, logger):
    test_oracle(sess, testset, tokenizer, oracle, cpu_pool, batch_size, logger)
    test_guesser(sess, testset, tokenizer, guesser, cpu_pool, batch_size, logger)
    test_qgen(sess, testset, tokenizer, qgen, cpu_pool, batch_size, logger)


def compute_qgen_accuracy(sess, dataset, batchifier, evaluator, mode, tokenizer, save_path, cpu_pool, batch_size, store_games, dump_suffix):

    logger = logging.getLogger()

    for m in mode:
        test_iterator = Iterator(dataset, pool=cpu_pool,
                                 batch_size=batch_size,
                                 batchifier=batchifier,
                                 shuffle=False,
                                 use_padding=True)
        test_score = evaluator.process(sess, test_iterator, mode=m, store_games=store_games)

        # Retrieve the generated games and dump them as a dataset
        if store_games:
            generated_dialogues = evaluator.get_storage()
            dump_samples_into_dataset(generated_dialogues,
                                      save_path=save_path,
                                      tokenizer=tokenizer,
                                      name=dump_suffix + "." + m)

        logger.info("Accuracy ({} - {}): {}".format(dataset.set, m, test_score))
