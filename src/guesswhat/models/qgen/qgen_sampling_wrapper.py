# import tensorflow as tf
#
# from generic.tf_models import utils
# from  generic.tf_utils.abstract_network import AbstractNetwork
#
#

from generic.tf_utils.evaluator import Evaluator

import numpy as np


class QGenSamplingWrapper(object):
    def __init__(self, qgen, tokenizer, max_length):

        self.qgen = qgen

        self.tokenizer = tokenizer
        self.max_length=max_length

        self.evaluator = None

        # Track the hidden state of LSTM
        self.state_c = None
        self.state_h = None
        self.state_size = int(qgen.decoder_zero_state_c.get_shape()[1])

    def initialize(self, sess):
        self.evaluator = Evaluator(self.qgen.get_sources(sess), self.qgen.scope_name)

    def reset(self, batch_size):
        # reset state
        self.state_c = np.zeros((batch_size, self.state_size))
        self.state_h = np.zeros((batch_size, self.state_size))


    def sample_next_question(self, sess, prev_answers, game_data, greedy):

        game_data["dialogues"] = prev_answers
        game_data["seq_length"] = [1]*len(prev_answers)
        game_data["state_c"] = self.state_c
        game_data["state_h"] = self.state_h
        game_data["greedy"] = greedy

        # sample
        res = self.evaluator.execute(sess, self.qgen.samples, game_data)

        self.state_c = res[0]
        self.state_h = res[1]
        transpose_questions = res[2]
        seq_length = res[3]

        # Get questions
        padded_questions = transpose_questions.transpose([1, 0])
        padded_questions = padded_questions[:,1:]  # ignore first token

        for i, l in enumerate(seq_length):
            padded_questions[i, l:] = self.tokenizer.padding_token

        questions = [q[:l] for q, l in zip(padded_questions, seq_length)]

        return padded_questions, questions, seq_length











# class QGenSampler(AbstractNetwork):
#     def __init__(self, qgen, config, tokenizer, max_length=12):
#         AbstractNetwork.__init__(self, "sampler")
#
#         batch_size = tf.shape(qgen.seq_length)[0]
#         with tf.variable_scope(self.scope_name):
#             # Picture
#             self.image = tf.placeholder(tf.float32, [batch_size] + config['image']["dim"], name='images')
#
#             # dialogues
#             self.input_tokens = tf.placeholder(tf.int64, shape=[None], name="input_tokens")
#
#             # DECODER Hidden state (for sampling and beam-search)
#             zero_state = tf.zeros([1, config['num_lstm_units']])  # default LSTM state is a zero-vector
#             zero_state = tf.tile(zero_state, [batch_size, 1])  # trick to do a dynamic size 0 tensors
#
#             self.decoder_zero_state_c = tf.placeholder_with_default(zero_state, [batch_size, config['num_lstm_units']], name="stace_c")
#             self.decoder_zero_state_h = tf.placeholder_with_default(zero_state, [batch_size, config['num_lstm_units']], name="stace_h")
#
#             # Misc
#             self.greedy = tf.placeholder_with_default(False, shape=(), name="is_greedy")
#             self.stop_token = tf.constant(tokenizer.stop_token)
#             self.stop_dialogue_token = tf.constant(tokenizer.stop_dialogue)
#
#             # initialialize sequence
#             tokens = tf.expand_dims(self.input_tokens, 0)
#             seq_length = tf.fill([batch_size], 0)
#
#         # retrieve picture embedding from qgen
#         with tf.variable_scope(qgen.scope_name, reuse=True):
#             with tf.variable_scope("picture_embedding"):
#                 picture_emb = utils.fully_connected(self.image,
#                                                     config['picture_embedding_size'],
#                                                     reuse=True)
#
#         def stop_cond(states_c, states_h, tokens, seq_length, stop_indicator):
#             return tf.logical_and(tf.less(tf.shape(tf.where(stop_indicator))[0], tf.shape(stop_indicator)[0]), tf.less(tf.reduce_max(seq_length), max_length))
#
#         def step(state_c, state_h, tokens, seq_length, stop_indicator):
#             input = tf.gather(tokens, tf.shape(tokens)[0] - 1)
#
#             is_stop_token = tf.equal(input, self.stop_token)
#             is_stop_dialogue_token = tf.equal(input, self.stop_dialogue_token)
#             is_stop = tf.logical_or(is_stop_token, is_stop_dialogue_token)
#             stop_indicator = tf.logical_or(stop_indicator, is_stop)
#
#             seq_length = tf.where(stop_indicator, seq_length, tf.add(seq_length, 1))
#
#             with tf.variable_scope(qgen.scope_name, reuse=True):
#                 word_emb = utils.get_embedding(
#                     input,
#                     n_words=tokenizer.no_words,
#                     n_dim=config['word_embedding_size'],
#                     scope="word_embedding",
#                     reuse=True)
#
#                 inp_emb = tf.concat([word_emb, picture_emb], axis=1)
#                 with tf.variable_scope("word_decoder"):
#                     lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
#                         config['num_lstm_units'],
#                         layer_norm=False,
#                         dropout_keep_prob=1.0)
#
#                     state = tf.contrib.rnn.LSTMStateTuple(c=state_c, h=state_h)
#                     out, state = lstm_cell(inp_emb, state)
#                     cond = tf.greater_equal(seq_length, tf.subtract(tf.reduce_max(seq_length), 1))
#                     state_c = tf.where(cond, state.c, state_c)
#                     state_h = tf.where(cond, state.h, state_h)
#
#                 with tf.variable_scope('decoder_output'):
#                     output = utils.fully_connected(state_h, tokenizer.no_words, reuse=True)
#
#                     sampled_tokens = tf.cond(self.greedy,
#                                              lambda: tf.argmax(output, 1),
#                                              lambda: tf.reshape(tf.multinomial(output, 1), [-1])
#                                              )
#
#             tokens = tf.concat(0, [tokens, tf.expand_dims(sampled_tokens, 0)])
#
#             return state_c, state_h, tokens, seq_length, stop_indicator
#
#         with tf.variable_scope(self.scope_name):
#             stop_indicator = tf.fill([batch_size], False)
#             self.samples = tf.while_loop(stop_cond, step, [self.decoder_zero_state_c,
#                                                            self.decoder_zero_state_h,
#                                                            tokens,
#                                                            seq_length,
#                                                            stop_indicator],
#                                          shape_invariants=[self.decoder_zero_state_c.get_shape(),
#                                                            self.decoder_zero_state_h.get_shape(),
#                                                            tf.TensorShape([None, None]),
#                                                            seq_length.get_shape(),
#                                                            stop_indicator.get_shape()])
#
#     def get_sampling_output(self):
#         return [self.samples]
#
#     def decode_sampling(self, samples):
#         # 'state_c': samples[0],
#         # 'state_h': samples[1],
#         # 'tokens': samples[2],
#         # 'seq_length': samples[3],
#         # 'stop_indicator': samples[4]
#
#         return samples[2], samples[3]
#
#
# def get_index(l, index, default=-1):
#     try:
#         return l.index(index)
#     except ValueError:
#         return default
#
#         # # Warning modify the input (use defensive copy?)
#         # def clear_after_stop_dialogue(dialogues, tokenizer):
#         #     no_questions = []
#         #     answer_indices = []
#         #     stop_indices = []
#         #     for i, dialogue in enumerate(dialogues):
#         #         stop_dialogue_index = get_index(dialogue.tolist(), tokenizer.stop_dialogue, default=len(dialogue)-1)
#         #         answers_index = [j for j,token in enumerate(dialogue[:stop_dialogue_index]) if token in tokenizer.answers]
#         #         if answers_index:
#         #             no_questions.append(len(answers_index))
#         #             answer_indices.append(answers_index)
#         #             dialogues[i] = dialogue[:stop_dialogue_index+1]
#         #             stop_indices.append(stop_dialogue_index)
#         #         else:
#         #             dialogues[i] = []
#         #             no_questions.append(0)
#         #             answer_indices.append([])
#         #             stop_indices.append(0)
#         #
#         #
#         #     return dialogues, no_questions, answer_indices, stop_indices
#         #
#         #
#         # def list_to_padded_tokens(dialogues, tokenizer):
#         #
#         #     # compute the length of the dialogue
#         #     seq_length = [len(d) for d in dialogues]
#         #
#         #     # Get dialogue numpy max size
#         #     batch_size = len(dialogues)
#         #     max_seq_length = max(seq_length)
#         #
#         #     # Initialize numpy array (Idea re-use previous numpy array to optimize memory consumption)
#         #     padded_tokens = np.full((batch_size, max_seq_length), tokenizer.padding_token, dtype=np.int32)
#         #
#         #     # fill the padded array with word_id
#         #     for i, (one_path, l) in enumerate(zip(dialogues, seq_length)):
#         #        padded_tokens[i, 0:l] = one_path
#         #
#         #     return padded_tokens, seq_length
#         #
#         #
#         # # TODO move to index
#
# #
# #
# # if __name__ == '__main__':
# #
# #     with open("../../../config/qgen/config_fc8.json", 'r') as f_config:
# #         config = json.load(f_config)
# #
# #     import guesswhat.data_provider as provider
# #     tokenizer = provider.GWTokenizer('../../../data/dict.json')
# #     config['model']['no_words'] = tokenizer.no_words
# #
# #     network = QGenNetworkLSTM(config)
# #
# #     # save_path = "/home/sequel/fstrub/rl_dialogue/tmp/qgen/{}"
# #     #
# #     # saver = tf.train.Saver()
# #     # with tf.Session() as sess:
# #     #     sess.run(tf.global_variables_initializer())
# #     #     saver.save(sess, save_path.format('params.ckpt'))
# #     #
# #     # tf.reset_default_graph()
# #     #
# #     # network = QGenNetworkLSTM(config)
# #     # saver = tf.train.Saver()
# #     with tf.Session() as sess:
# #         # saver.restore(sess, save_path.format('params.ckpt'))
# #         sess.run(tf.global_variables_initializer())
# #
# #         pictures = np.zeros((2, 1000))
# #
# #         dialogues = np.array([
# #                 [2, 5, 3, 23,  5, 9, 3, 24, 4, 4],  # 2: start / 3: stop / 4: padding
# #                 [2, 5, 6,  3, 24, 8, 9, 9, 3, 23],
# #             ])
# #
# #         answer_mask = np.array([
# #                         [1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
# #                         [1., 1., 1., 1., 0., 1., 1., 1., 1., 0.]
# #             ])
# #
# #         padding_mask = np.array([
# #                         [1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
# #                         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
# #         ])
# #
# #         seq_length = np.array([8, 10])
# #
# #         sess.run(network.ml_loss,
# #                  feed_dict={
# #                      network.picture_fc8: pictures,
# #                      network.dialogues: dialogues,
# #                      network.answer_mask: answer_mask,
# #                      network.padding_mask: padding_mask,
# #                      network.seq_length: seq_length
# #                  })
# #
# #         one_sample = {
# #             "picture_fc8": np.zeros((1, 1000)),
# #         }
# #         network.eval_one_beam_search(sess, one_sample, tokenizer, max_depth=3)
# #
# #         one_sample = {
# #             "picture_fc8": np.zeros((1, 1000)),
# #             "padded_tokens": np.array([[2, 5, 6, 3, 24, 8, 9, 9, 3, 23]])
# #         }
# #
# #         network.eval_one_beam_search(sess, one_sample, tokenizer, max_depth=3)
# #
# #         batch = {
# #             "picture_fc8": np.zeros((2, 1000)),
# #         }
# #         network.generate_next_question(sess, batch, tokenizer, max_depth=3)
# #         network.generate_next_question(sess, batch, tokenizer, greedy=True, max_depth=3)
# #
# #         batch = {
# #             "picture_fc8": np.zeros((2, 1000)),
# #             "padded_tokens": dialogues
# #         }
# #         network.generate_next_question(sess, batch, tokenizer, max_depth=3)
# #         network.generate_next_question(sess, batch, tokenizer, greedy=True, max_depth=3)
# #
# #         one_sample = {
# #             "picture_fc8": np.zeros((1, 1000)),
# #             "padded_tokens": np.array([[0, 2, 5, 6, 1, 23]])
# #         }
# #
# #         trajectory = network.generate_next_question(sess, one_sample, tokenizer, greedy=True, max_depth=3, keep_trajectory=True)
