from guesswhat.models.qgen.qgen_rnn_network import QGenNetworkRNN
from guesswhat.models.qgen.qgen_decoder_network import QGenNetworkDecoder
from guesswhat.data_provider.qgen_seq2seq_batchifier import Seq2SeqBatchifier
from guesswhat.data_provider.qgen_rnn_batchifier import RNNBatchifier


def create_qgen(config, num_words, rl_module=None):

    network_type = config["type"]

    if network_type == "rnn":
        network = QGenNetworkRNN(config, num_words, rl_module=rl_module)
        batchifier = RNNBatchifier

    elif network_type == "seq2seq":
        network = QGenNetworkDecoder(config, num_words, rl_module=rl_module)
        batchifier = Seq2SeqBatchifier

    else:
        assert False, "Invalid network_type: should be: baseline/oracle"

    return network, batchifier

