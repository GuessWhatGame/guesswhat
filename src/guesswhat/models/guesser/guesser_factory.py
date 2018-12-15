from guesswhat.models.guesser.guesser_baseline import GuesserNetwork
from guesswhat.models.oracle.oracle_film import FiLM_Oracle

from guesswhat.data_provider.guesser_crop_batchifier import GuesserCropBatchifier
from guesswhat.data_provider.guesser_batchifier import GuesserBatchifier

from guesswhat.train.eval_listener import GuesserAccuracyListener, AccuracyListener


# factory class to create networks and the related batchifier

def create_guesser(config, num_words, reuse=False):

    network_type = config["type"]

    if network_type == "oracle":
        network = FiLM_Oracle(config, num_words, num_answers=2)
        batchifier = GuesserCropBatchifier
        listener = GuesserAccuracyListener(require=network.softmax)

    elif network_type == "baseline":
        network = GuesserNetwork(config, num_words=num_words, reuse=reuse)
        batchifier = GuesserBatchifier
        listener = AccuracyListener(require=network.softmax)

    else:
        assert False, "Invalid network_type: should be: baseline/oracle"

    return network, batchifier, listener

