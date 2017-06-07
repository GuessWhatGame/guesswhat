import json
import numpy as np
import tensorflow as tf
import os

from abc import ABCMeta, abstractmethod
from tqdm import *

from guesswhat.models.rnn import variable_length_LSTM
from guesswhat.models import utils

import abc


class Guesser(object):
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def predict(self, sess, input):
        raise NotImplementedError()


class GuesserNetwork(Guesser):
    def __init__(self, config):

        self.scope_name = "guesser"

        with tf.variable_scope(self.scope_name):

            mini_batch_size = None
            # PICTURE
            # self.picture_fc8 = tf.placeholder(
            #     tf.float32,
            #     [mini_batch_size, config['model']['fc8_dim']],
            #     name='picture_fc8')

            # FLATTEN dialogue_lstm
            self.dialogues = tf.placeholder(tf.int32,
                                            [mini_batch_size, None],
                                            name='question')

            self.seq_length = tf.placeholder(tf.int32,
                                             [mini_batch_size],
                                             name='seq_length')
            # OBJECTS
            self.mask = tf.placeholder(tf.float32, [mini_batch_size, None],
                                       name='mask')
            self.obj_cats = tf.placeholder(tf.int32,
                                           [mini_batch_size, None],
                                           name='obj_cats')
            self.obj_spats = tf.placeholder(tf.float32,
                                            [mini_batch_size, None, config['model']['spat_dim']],
                                            name='obj_spats')

            self.object_cats_emb = utils.get_embedding(
                self.obj_cats,
                config['no_categories'],
                config['model']['cat_emb_dim'],
                scope='cat_embedding')

            # TARGETS
            self.targets = tf.placeholder(tf.int32, [mini_batch_size])

            self.objects_inp = tf.concat(2, [self.object_cats_emb, self.obj_spats])

            obj_inp_dim = config['model']['cat_emb_dim'] + config['model']['spat_dim']
            self.flat_objects_inp = tf.reshape(self.objects_inp, [-1, obj_inp_dim])

            with tf.variable_scope('obj_mlp'):
                h1 = utils.fully_connected(
                    self.flat_objects_inp,
                    config['model']['obj_mlp_units'],
                    activation='relu',
		    scope='l1')
                h2 = utils.fully_connected(
                    h1,
                    config['model']['dialog_emb_dim'],
                    activation='relu',
		    scope='l2')

            obj_embs = tf.reshape(h2, [-1, tf.shape(self.obj_cats)[1], config['model']['dialog_emb_dim']])

            # Compute the word embedding
            input_words = utils.get_embedding(self.dialogues,
                                              n_words=config['model']["no_words"],
                                              n_dim=config['model']['word_emb_dim'],
                                              scope="input_word_embedding")

            last_states = variable_length_LSTM(input_words,
                                               num_hidden=config['model']['num_lstm_units'],
                                               seq_length=self.seq_length)

            last_states = tf.reshape(last_states, [-1, config['model']['num_lstm_units'], 1])
            scores = tf.matmul(obj_embs, last_states)
            scores = tf.reshape(scores, [-1, tf.shape(self.obj_cats)[1]])

            def masked_softmax(scores, mask):
                # subtract max for stability
                scores = scores - tf.tile(tf.reduce_max(scores, axis=(1,), keep_dims=True), [1, tf.shape(scores)[1]])
                # compute padded softmax
                exp_scores = tf.exp(scores)
                exp_scores *= mask
                exp_sum_scores = tf.reduce_sum(exp_scores, axis=1, keep_dims=True)
                return exp_scores / tf.tile(exp_sum_scores, [1, tf.shape(exp_scores)[1]])

            self.softmax = masked_softmax(scores, self.mask)
            self.selected_object = tf.argmax(self.softmax, axis=1)

            self.loss = tf.reduce_mean(utils.cross_entropy(self.softmax, self.targets))
            self.error = tf.reduce_mean(utils.error(self.softmax, self.targets))

            lrt = config['optimizer']['learning_rate']

            train_vars = [v for v in tf.trainable_variables() if v.name.startswith(self.scope_name)]
            optimizer = tf.train.AdamOptimizer(learning_rate=lrt)
            gvs = optimizer.compute_gradients(self.loss, var_list=train_vars)
            clipped_gvs = [(tf.clip_by_norm(grad, config['optimizer']['clip_val']), var) for grad, var in gvs]

            self.optimize = optimizer.apply_gradients(clipped_gvs)

    @classmethod
    def from_exp_identifier(cls, identifier, exp_dir):
        file = os.path.join(exp_dir, 'experiments.jsonl')
        with open(file, 'rb') as f:
            config = None
            for line in f:
                s = json.loads(line.decode('utf-8'))
                if s['identifier'] == identifier:
                    config = s['config']
            assert(config is not None), "Couldn't find Guesser config"
        return cls(config)

    def train(self, sess, iterator):
        return self._execute(sess, iterator, True, 'Training..')

    def evaluate(self, sess, iterator):
        return self._execute(sess, iterator, False, 'Validation..')

    def _execute(self, sess, iterator, is_training, desc):
        if is_training:
            nodes = [self.loss, self.error, self.optimize]
        else:
            nodes = [self.loss, self.error]

        loss = 0.0
        err = 0.0
        N = 0

        for batch in tqdm(iterator, desc=desc):
            result = sess.run(nodes, feed_dict={
                self.dialogues: batch['padded_tokens'],
                self.seq_length: batch['seq_length'],
                self.mask: batch['obj_mask'],
                self.obj_spats: batch['obj_spats'],
                self.obj_cats: batch['obj_cats'],
                self.targets: batch['targets']
            })
            loss += result[0]
            err += result[1]
            N += 1

        return loss / N, err / N

    def predict(self, sess, input):
        return None  # TODO

    # def predict(self,  :
    #     out = sess.run(self.softmax,feed_dict={
    #         self.dialogues: [dialogue],
    #         self.seq_length: [len(dialogue)],
    #         self.mask: np.ones((1, len(obj_cat))),
    #         self.obj_spats: [obj_spat],
    #         self.obj_cats: [obj_cat],
    #     })
    #
    #     return out

    def find_object(self, sess, dialogue, seq_length, ground_data):
        """Inputs:
        High level method that return whether the guesser manage to find the correct object according a dialogue and game information

        Example
        --------
        {'question': [[1, 500, 3, 5, 2], [1, 48, 12, 2, 4]],
         'seq_length': [5, 4] length of the sequence (=first padding token(4) or array shape)
         'ground_data', {
            obj_mask : [[1,1,1,1], [1,1,,1,0]], #4 object / 3 objects
            obj_spats : np.array((2,8)), # spatial info by object
            obj_spats : [[10,22,11,10], [5,10,10]], # obj cat Ex [person,dog,cat,person],[kite,person,person]
            object_indices : [3,0] # indices for correct object, e.g. person:1 / kite:0
            },
         'spatial': [[-0.5, 0.8, 0.7, 0.5, 0.4, 0.56, -0.3, -0.1]],
         'seq_length': [5]}
        """

        # TO avoid code duplication, we can:
        # - create a predict method
        # guesser_input = dict(ground_data) # shallow copy
        # guesser_input["question"] = dialogue
        # guesser_input["seq_length"] = seq_length
        # selected_object = self.predict(sess, guesser_input) # return predicted_object (or softmax)
        # found = (selected_object == ground_data["targets"])
        # OR found = (np.argmax(selected_object, axis=1) == ground_data["targets"]) if softmax

        selected_object, softmax = sess.run([self.selected_object, self.softmax], feed_dict={
            self.dialogues: dialogue,
            self.seq_length: seq_length,
            self.mask: ground_data["obj_mask"],
            self.obj_spats: ground_data["obj_spats"],
            self.obj_cats: ground_data["obj_cats"],
        })

        found = (selected_object == ground_data["targets"])

        return found, softmax


def test_forward_pass():
    with open('../config/config_guesser.json', 'r') as f_config:
        config = json.load(f_config)

    import guesswhat.data_provider as provider
    tokenizer = provider.GWTokenizer('../data/dict.json')
    config['model']['no_words'] = tokenizer.no_words

    network = GuesserNetwork(config)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        dialogues = np.array(
            [
                [0, 1, 5, 6, 0, 0, 0],
                [0, 1, 2, 3, 100, 250, 1],
            ]
        )

        tmp = sess.run(network.loss,
                       feed_dict={
                           network.dialogues: dialogues,
                           network.seq_length: np.array([6, 7]),
                           network.mask: np.ones((2, 3), dtype=np.float32),
                           network.obj_cats: np.array([[3, 4, 5], [0, 6, 9]]),
                           network.obj_spats: np.ones((2, 3, 8), dtype=np.float32),
                           network.targets: np.array([1, 2])
                       })
        print(tmp)


def test_reload():
    import os
    from guesswhat.models.oracle.oracle_network import OracleNetwork

    identifier = '-8725679733986404343'
    exp_dir = '/data/lisa/data/guesswhat/exp/guesser'

    params_file = os.path.join(exp_dir, identifier, 'params.ckpt')
    network = OracleNetwork.from_exp_identifier(identifier, exp_dir)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, params_file)

        inputs = {
            'dialogues': [[1, 2, 4, 5, 6, 9, 10]],
            'obj_cats': [[1, 4, 5, 7]],
            'obj_spats': [[]]
        }
        print(network.predict(sess, inputs))


if __name__ == '__main__':
    test_reload()
    # test_forward_pass()
