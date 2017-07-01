from __future__ import unicode_literals
import collections
import pickle


from generic.data_provider.image_loader import *

Environment = collections.namedtuple('Environment',  ['trainset', 'validset', 'testset'])

def load_checkpoint(sess, saver, args, save_path):
    ckpt_path = save_path.format('params.ckpt')

    if args.continue_exp:
        if not os.path.exists(save_path.format('checkpoint')):
            raise ValueError("Checkpoint " + save_path.format('checkpoint') + " could not be found.")

        saver.restore(sess, ckpt_path)
        status_path = save_path.format('status.pkl')
        status = pickle.load(open(status_path, 'rb'))

        return status['epoch'] + 1

    if args.load_checkpoint is not None:
        #if not os.path.exists(save_path.format('checkpoint')):
        #    raise ValueError("Checkpoint " + args.load_checkpoint + " could not be found.")
        saver.restore(sess, args.load_checkpoint)

        return 0

    return 0


def get_img_loader(config, image_dir, is_crop=False):

    image_input = config["image_input"]

    if image_input == "features":
        is_flat = len(config["dim"]) == 1
        if is_flat:
            loader = fcLoader(image_dir)
        else:
            loader = ConvLoader(image_dir)
    elif image_input == "raw":
        if is_crop:
            loader = RawCropLoader(image_dir,
                                    height=config["dim"][0],
                                    width=config["dim"][1],
                                    scale=config["scale"],
                                    channel=config.get("channel", None),
                                    extension=config.get("extension", "jpg"))
        else:
            loader = RawImageLoader(image_dir,
                                    height=config["dim"][0],
                                    width=config["dim"][1],
                                    channel=config.get("channel", None),
                                    extension=config.get("extension", "jpg"))
    else:
        assert False, "incorrect image input: {}".format(image_input)

    return loader


from guesswhat.data_provider.questioner_batchifier import QuestionerBatchifier
from guesswhat.data_provider.oracle_batchifier import OracleBatchifier
from guesswhat.data_provider.guesswhat_dataset import OracleDataset

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator


def test_oracle(sess, testset, tokenizer, oracle, cpu_pool, batch_size, logger):

    oracle_dataset = OracleDataset(testset)
    oracle_sources = oracle.get_sources(sess)
    oracle_evaluator = Evaluator(oracle_sources, oracle.scope_name, network=oracle, tokenizer=tokenizer)
    oracle_batchifier = OracleBatchifier(tokenizer, oracle_sources, status=('success',))
    oracle_iterator = Iterator(oracle_dataset, pool=cpu_pool,
                             batch_size=batch_size,
                             batchifier=oracle_batchifier)
    [oracle_loss, oracle_error] = oracle_evaluator.process(sess, oracle_iterator, oracle.get_outputs())

    logger.info("Oracle test loss: {}".format(oracle_loss))
    logger.info("Oracle test error: {}".format(oracle_error))


def test_guesser(sess, testset, tokenizer, guesser, cpu_pool, batch_size, logger):
    guesser_sources = guesser.get_sources(sess)
    guesser_evaluator = Evaluator(guesser_sources, guesser.scope_name, network=guesser, tokenizer=tokenizer)
    guesser_batchifier = QuestionerBatchifier(tokenizer, guesser_sources, status=('success',))
    guesser_iterator = Iterator(testset, pool=cpu_pool,
                             batch_size=batch_size,
                             batchifier=guesser_batchifier)
    [guesser_loss, guesser_error] = guesser_evaluator.process(sess, guesser_iterator, guesser.get_outputs())
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
