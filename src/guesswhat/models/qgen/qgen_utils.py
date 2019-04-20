import tensorflow.contrib.seq2seq as tfc_seq

from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest

import collections


class BasicDecoderWithStateOutput(
    collections.namedtuple('BasicDecoderWithStateOutput', ('rnn_output', 'rnn_state', 'sample_id'))):
    """ Basic Decoder Named Tuple with rnn_output, rnn_state, and sample_id """
    pass

class BasicDecoderWithState(tfc_seq.BasicDecoder):

    def __init__(self, cell, helper, initial_state, output_layer=None):
        super(BasicDecoderWithState, self).__init__(cell=cell,
                                                    helper=helper,
                                                    initial_state=initial_state,
                                                    output_layer=output_layer)

    @property
    def output_size(self):
        return BasicDecoderWithStateOutput(rnn_output=self._rnn_output_size(),
                                           rnn_state=tensor_shape.TensorShape([self._cell.output_size]),  # TODO: Do not support lstm for now
                                           sample_id=self._helper.sample_ids_shape)

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = nest.flatten(self._initial_state)[0].dtype
        return BasicDecoderWithStateOutput(nest.map_structure(lambda _: dtype, self._rnn_output_size()),
                                           dtype,
                                           self._helper.sample_ids_dtype)

    def step(self, time, inputs, state, name=None):
        (outputs, next_state, next_inputs, finished) = super(BasicDecoderWithState, self).step(time, inputs, state, name)

        # store state
        outputs = BasicDecoderWithStateOutput(
            rnn_output=outputs.rnn_output,
            rnn_state=next_state,
            sample_id=outputs.sample_id)

        return outputs, next_state, next_inputs, finished
