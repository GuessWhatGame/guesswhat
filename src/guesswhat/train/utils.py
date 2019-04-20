from __future__ import unicode_literals

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator
import logging
import copy


def test_one_model(sess, dataset, cpu_pool, batch_size, network, batchifier, loss, listener=None):
    sources = network.get_sources(sess)
    evaluator = Evaluator(sources, network.scope_name, network=network)
    iterator = Iterator(dataset, pool=cpu_pool, batch_size=batch_size, batchifier=batchifier)
    return evaluator.process(sess, iterator, outputs=loss, listener=listener)


def test_models(sess, dataset, cpu_pool, batch_size,
                oracle, oracle_batchifier,
                guesser, guesser_batchifier, guesser_listener,
                qgen, qgen_batchifier):

    # Retrieve logger
    logger = logging.getLogger()

    # Oracle:
    oracle_batchifier = copy.copy(oracle_batchifier)
    oracle_batchifier.status = []  # test on full dataset
    [oracle_loss, oracle_accuracy] = test_one_model(sess, dataset, cpu_pool, batch_size,
                                                    network=oracle,
                                                    batchifier=oracle_batchifier,
                                                    loss=[oracle.loss, oracle.accuracy])
    logger.info("Oracle test loss: {}".format(oracle_loss))
    logger.info("Oracle test error: {}".format(1-oracle_accuracy))
    logger.info("Oracle test accuracy: {}".format(oracle_accuracy))

    # Guesser:
    guesser_batchifier = copy.copy(guesser_batchifier)
    guesser_batchifier.status = ["success"]  # test on successful dataset
    [guesser_loss] = test_one_model(sess, dataset, cpu_pool, batch_size,
                                    network=guesser,
                                    batchifier=guesser_batchifier,
                                    loss=[guesser.loss],
                                    listener=guesser_listener)
    logger.info("Guesser test loss: {}".format(guesser_loss))
    logger.info("Guesser test error: {}".format(1-guesser_listener.accuracy()))
    logger.info("Guesser test accuracy: {}".format(guesser_listener.accuracy()))

    # QGen:
    qgen_batchifier = copy.copy(qgen_batchifier)
    qgen_batchifier.status = ["success"]  # test on successful dataset
    qgen_batchifier.supervised = True
    qgen_batchifier.generate = False
    [guesser_loss] = test_one_model(sess, dataset, cpu_pool, batch_size,
                                    network=qgen,
                                    batchifier=qgen_batchifier,
                                    loss=[qgen.ml_loss])
    logger.info("QGen test loss: {}".format(guesser_loss))


def compute_qgen_accuracy(sess, dataset, batchifier, looper, mode, cpu_pool, batch_size, name, save_path, store_games=False):

    logger = logging.getLogger()

    for m in mode:
        test_iterator = Iterator(dataset, pool=cpu_pool,
                                 batch_size=batch_size,
                                 batchifier=batchifier,
                                 shuffle=False)

        [test_score, games] = looper.process(sess, test_iterator, mode=m, store_games=store_games)

        logger.info("Accuracy ({} - {}): {}".format(dataset.set, m, test_score))

        if store_games:
            dump_dataset(games,
                         save_path=save_path,
                         tokenizer=looper.tokenizer,
                         name=name + "." + m)


def dump_dataset(games, save_path, tokenizer, name="model"):
    import gzip
    import os
    import json

    with gzip.open(os.path.join(save_path, 'guesswhat.' + name + '.jsonl.gz'), 'wb') as f:

        for _, game in enumerate(games):

            sample = {}

            qas = []
            for id, question, answers in zip(game.question_ids, game.questions, game.answers):
                qas.append({"question": question,
                            "answer": answers,
                            "id": id,
                            "p": 0})

            sample["id"] = game.dialogue_id
            sample["qas"] = qas
            sample["image"] = {
                "id": game.image.id,
                "width": game.image.width,
                "height": game.image.height,
                "coco_url": game.image.url
            }

            sample["objects"] = [{"id": o.id,
                                  "category_id": o.category_id,
                                  "category": o.category,
                                  "area": o.area,
                                  "bbox": o.bbox.coco_bbox,
                                  "segment": o.segment,  # no segment to avoid making the file too big
                                  } for o in game.objects]

            sample["object_id"] = game.object.id
            sample["guess_object_id"] = game.id_guess_object
            sample["status"] = game.status

            f.write(str(json.dumps(sample)).encode())
            f.write(b'\n')

# def compute_qgen_accuracy(sess, dataset, batchifier, evaluator, mode, tokenizer, save_path, cpu_pool, batch_size, store_games, dump_suffix):
#
#     logger = logging.getLogger()
#
#     for m in mode:
#         test_iterator = Iterator(dataset, pool=cpu_pool,
#                                  batch_size=batch_size,
#                                  batchifier=batchifier,
#                                  shuffle=False)
#         test_score = evaluator.process(sess, test_iterator, mode=m, store_games=store_games)
#
#         # Retrieve the generated games and dump them as a dataset
#         if store_games:
#             generated_dialogues = evaluator.get_storage()
#
#
#         logger.info("Accuracy ({} - {}): {}".format(dataset.set, m, test_score))