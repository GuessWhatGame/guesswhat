import tensorflow as tf

import tensorflow.contrib.layers as tfc_layers

from neural_toolbox import rnn

from generic.tf_factory.image_factory import get_image_features
from generic.tf_factory.attention_factory import get_attention
from generic.tf_utils.abstract_network import AbstractNetwork

from guesswhat.models.qgen.qgen_utils import *


class QGenNetworkRNN(AbstractNetwork):

    def __init__(self, config, num_words, policy_gradient, device='', reuse=False):
        AbstractNetwork.__init__(self, "qgen", device=device)

        # Create the scope for this graph
        with tf.variable_scope(self.scope_name, reuse=reuse):

            # Misc
            self._is_training = tf.placeholder(tf.bool, name='is_training')
            batch_size = None

            #####################
            #   IMAGES
            #####################

            self.img_embedding = None
            if config['inputs']['image']:

                self._image = tf.placeholder(tf.float32, [batch_size] + config['image']["dim"], name='image')

                self.visual_features = get_image_features(image=self._image,
                                                          is_training=self._is_training,
                                                          config=config['image'])

                # Pool Image Features
                with tf.variable_scope("image_pooling"):
                    self.visual_features = get_attention(self.visual_features,
                                                         context=None,
                                                         is_training=self._is_training,
                                                         config=config["pooling"],
                                                         dropout_keep=1.,
                                                         reuse=reuse)

                if config['image']['projection_units'] > 0:
                    self.visual_features = tfc_layers.fully_connected(self.visual_features,
                                                                      num_outputs=config["image"]["projection_units"],
                                                                      activation_fn=tf.nn.relu,
                                                                      reuse=reuse,
                                                                      scope="image_projection_units")

            else:
                self.visual_features = None

            #####################
            #   DIALOGUE
            #####################

            self._dialogue = tf.placeholder(tf.int32, [batch_size, None], name='dialogue')
            self._seq_length_dialogue = tf.placeholder(tf.int32, [batch_size], name='seq_length_dialogue')

            input_dialogue = self._dialogue[:, :-1]  # Ignore stop token
            target_dialogue = self._dialogue[:, 1:]  # Ignore start token

            with tf.variable_scope('word_embedding', reuse=reuse):
                self.dialogue_emb_weights = tf.get_variable("dialogue_embedding_encoder",
                                                            shape=[num_words, config["dialogue"]["word_embedding_dim"]])

            word_emb_dialogue = tf.nn.embedding_lookup(params=self.dialogue_emb_weights, ids=input_dialogue)

            if config['inputs']['image']:
                visual_features = tf.expand_dims(self.visual_features, axis=1)
                visual_features = tf.tile(visual_features, multiples=[1, tf.reduce_max(self._seq_length_dialogue) - 1, 1])

                visword_emb_dialogue = tf.concat([word_emb_dialogue, visual_features], axis=-1)
            else:
                visword_emb_dialogue = word_emb_dialogue

            #####################
            #   DECODER
            #####################

            self._question_mask = tf.sequence_mask(lengths=self._seq_length_dialogue - 1, dtype=tf.float32)  # -1 : remove start token at decoding time
            self._answer_mask = tf.placeholder(tf.float32, [batch_size, None], name='answer_mask')[:, 1:]  # -1 : remove start token at decoding time

            assert config['dialogue']["cell"] != "lstm", "LSTM are not yet supported for the decoder"
            self.decoder_cell = rnn.create_cell(cell=config['dialogue']["cell"],
                                                num_units=config['dialogue']["rnn_units"],
                                                layer_norm=config["dialogue"]["layer_norm"],
                                                reuse=reuse)

            self.decoder_projection_layer = tf.layers.Dense(num_words, use_bias=False)

            training_helper = tfc_seq.TrainingHelper(inputs=visword_emb_dialogue,  # The question is the target
                                                     sequence_length=self._seq_length_dialogue - 1)  # -1 : remove start token at decoding time

            # Define RNN states
            self.zero_states = self.decoder_cell.zero_state(tf.shape(self._seq_length_dialogue)[0], dtype=tf.float32)
            decoder = BasicDecoderWithState(cell=self.decoder_cell,
                                            helper=training_helper,
                                            initial_state=self.zero_states,
                                            output_layer=self.decoder_projection_layer)

            (self.decoder_outputs, self.decoder_states, _), _, _ = tfc_seq.dynamic_decode(decoder, maximum_iterations=None)

            #####################
            #   LOSS
            #####################

            # ignore answers while computing cross entropy or applying reward
            mask = (self._question_mask - self._answer_mask)

            # compute the softmax for evaluation
            with tf.variable_scope('ml_loss'):

                self.ml_loss = tfc_seq.sequence_loss(logits=self.decoder_outputs,
                                                     targets=target_dialogue,
                                                     weights=mask,
                                                     average_across_timesteps=True,
                                                     average_across_batch=True)

                self.softmax_output = tf.nn.softmax(self.decoder_outputs, name="softmax")
                self.argmax_output = tf.argmax(self.decoder_outputs, axis=2)

                self.loss = self.ml_loss

            # Compute policy gradient
            if policy_gradient:

                self._cum_rewards = tf.placeholder(tf.float32, shape=[batch_size, None], name='cum_reward')[:, 1:]

                with tf.variable_scope('rl_baseline'):
                    decoder_out = tf.stop_gradient(self.decoder_states)  # take the LSTM output (and stop the gradient!)

                    baseline_hidden = tfc_layers.fully_connected(decoder_out,
                                                                 num_outputs=int(int(decoder_out.get_shape()[-1]) / 4),
                                                                 activation_fn=tf.nn.relu,
                                                                 scope='baseline_hidden',
                                                                 reuse=reuse)

                    baseline_out = tfc_layers.fully_connected(baseline_hidden,
                                                              num_outputs=1,
                                                              activation_fn=None,
                                                              scope='baseline',
                                                              reuse=reuse)
                    self.baseline = tf.squeeze(baseline_out, axis=-1)

                    self.baseline_loss = tf.square(self._cum_rewards - self.baseline)
                    self.baseline_loss *= mask

                    self.baseline_loss = tf.reduce_sum(self.baseline_loss, axis=1)
                    self.baseline_loss = tf.reduce_mean(self.baseline_loss, axis=0)

                with tf.variable_scope('policy_gradient_loss'):
                    self.log_of_policy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.decoder_outputs, labels=target_dialogue)

                    self.policy_gradient_loss = tf.multiply(self.log_of_policy, self._cum_rewards - self.baseline)  # score function
                    self.policy_gradient_loss *= mask

                    self.policy_gradient_loss = tf.reduce_sum(self.policy_gradient_loss, axis=1)  # sum over the dialogue trajectory
                    self.policy_gradient_loss = tf.reduce_mean(self.policy_gradient_loss, axis=0)  # reduce over minibatch dimension

                    self.loss = self.policy_gradient_loss

    def create_sampling_graph(self, start_token, stop_token, max_tokens):

        batch_size = tf.shape(self._dialogue)[0]

        def embedding(idx):
            emb = tf.nn.embedding_lookup(params=self.dialogue_emb_weights, ids=idx)
            emb = tf.concat([emb, self.visual_features], axis=-1)
            return emb

        sample_helper = tfc_seq.SampleEmbeddingHelper(embedding=embedding,
                                                      start_tokens=tf.fill([batch_size], start_token),
                                                      end_token=stop_token)

        decoder = tfc_seq.BasicDecoder(
            self.decoder_cell, sample_helper, self.zero_states,
            output_layer=self.decoder_projection_layer)

        (_, sample_id), _, seq_length = tfc_seq.dynamic_decode(decoder, maximum_iterations=max_tokens)

        return sample_id, seq_length

    def create_greedy_graph(self, start_token, stop_token, max_tokens):

        batch_size = tf.shape(self._dialogue)[0]

        def embedding(idx):
            emb = tf.nn.embedding_lookup(params=self.dialogue_emb_weights, ids=idx)
            emb = tf.concat([emb, self.visual_features], axis=-1)
            return emb

        greedy_helper = tfc_seq.GreedyEmbeddingHelper(embedding=embedding,
                                                      start_tokens=tf.fill([batch_size], start_token),
                                                      end_token=stop_token)

        decoder = tfc_seq.BasicDecoder(
            self.decoder_cell, greedy_helper, self.zero_states,
            output_layer=self.decoder_projection_layer)

        (_, sample_id), _, seq_length = tfc_seq.dynamic_decode(decoder, maximum_iterations=max_tokens)

        return sample_id, seq_length

    def create_beam_graph(self, start_token, stop_token, max_tokens, k_best):

        # create k_beams
        decoder_initial_state = tf.contrib.seq2seq.tile_batch(
            self.zero_states, multiplier=k_best)

        def embedding(idx):
            emb = tf.nn.embedding_lookup(params=self.dialogue_emb_weights, ids=idx)

            visual_features = tf.expand_dims(self.visual_features, axis=1)
            visual_features = tf.tile(visual_features, multiples=[1, tf.shape(idx)[1], 1])

            emb = tf.concat([emb, visual_features], axis=-1)
            return emb

        # Define a beam-search decoder
        batch_size = tf.shape(self._dialogue)[0]
        decoder = tfc_seq.BeamSearchDecoder(
            cell=self.decoder_cell,
            embedding=embedding,
            start_tokens=tf.fill([batch_size], start_token),
            end_token=stop_token,
            initial_state=decoder_initial_state,
            beam_width=k_best,
            output_layer=self.decoder_projection_layer,
            length_penalty_weight=0.0)

        (_, sample_id), _, seq_length = tfc_seq.dynamic_decode(decoder, maximum_iterations=max_tokens)

        return sample_id, seq_length

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.loss


if __name__ == "__main__":
    import json

    with open("../../../../config/qgen/config.rnn.json", 'rb') as f_config:
        config = json.load(f_config)

    network = QGenNetworkRNN(config["model"], num_words=111, policy_gradient=True)

    network.create_sampling_graph(start_token=1, stop_token=2, max_tokens=10)
    network.create_greedy_graph(start_token=1, stop_token=2, max_tokens=10)
    network.create_beam_graph(start_token=1, stop_token=2, max_tokens=10, k_best=5)
