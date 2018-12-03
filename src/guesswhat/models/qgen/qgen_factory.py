from guesswhat.models.qgen.qgen_lstm_network import QGenNetworkLSTM
# from guesswhat.models.qgen.oracle_film import FiLM_Oracle

from guesswhat.data_provider.rnn_batchifier import RNNBatchifier


from guesswhat.train.eval_listener import GuesserAccuracyListener, DummyAccuracyListener


# factory class to create networks and the related batchifier

def create_qgen(config, num_words, reuse=False):

    network_type = config["type"]

    if network_type == "lstm":
        network = QGenNetworkLSTM(config, num_words=num_words, reuse=reuse)
        batchifier = RNNBatchifier

    else:
        assert False, "Invalid network_type: should be: baseline/oracle"

    return network, batchifier

