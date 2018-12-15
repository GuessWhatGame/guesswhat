import tensorflow as tf

import tensorflow.contrib.layers as tfc_layers

from neural_toolbox import rnn

from generic.tf_factory.image_factory import get_image_features
from generic.tf_factory.fusion_factory import get_fusion_mechanism
from generic.tf_factory.attention_factory import get_attention

from generic.tf_utils.abstract_network import AbstractNetwork
from guesswhat.models.qgen.qgen_utils import *

from neural_toolbox.reading_unit import create_reading_unit, create_film_layer_with_reading_unit
from neural_toolbox.film_stack import FiLM_Stack


class QGenNetworkDecoder(AbstractNetwork):

    def __init__(self, config, num_words, policy_gradient, device='', reuse=False):
        AbstractNetwork.__init__(self, "qgen", device=device)

        # Create the scope for this graph
        with tf.variable_scope(self.scope_name, reuse=reuse):

            # Misc
            self._is_training = tf.placeholder(tf.bool, name='is_training')
            batch_size = None

            dropout_keep_scalar = float(config.get("dropout_keep_prob", 1.0))
            dropout_keep = tf.cond(self._is_training,
                                   lambda: tf.constant(dropout_keep_scalar),
                                   lambda: tf.constant(1.0))

            #####################
            #   DIALOGUE
            #####################

            self._dialogue = tf.placeholder(tf.int32, [batch_size, None], name='dialogue')
            self._seq_length_dialogue = tf.placeholder(tf.int32, [batch_size], name='seq_length_dialogue')

            with tf.variable_scope('word_embedding', reuse=reuse):
                self.dialogue_emb_weights = tf.get_variable("dialogue_embedding_encoder",
                                                            shape=[num_words, config["dialogue"]["word_embedding_dim"]])

            word_emb_dialogue = tf.nn.embedding_lookup(params=self.dialogue_emb_weights, ids=self._dialogue)

            word_emb_dialogue = tf.nn.dropout(word_emb_dialogue, dropout_keep)
            self.dialogue_states, self.dialogue_last_states = \
                rnn.rnn_factory(inputs=word_emb_dialogue,
                                seq_length=self._seq_length_dialogue,
                                cell=config['dialogue']["cell"],
                                num_hidden=config['dialogue']["rnn_units"],
                                bidirectional=config["dialogue"]["bidirectional"],
                                max_pool=config["dialogue"]["max_pool"],
                                layer_norm=config["dialogue"]["layer_norm"],
                                reuse=reuse)

            #####################
            #   IMAGES
            #####################

            self.img_embedding = None
            if config['inputs']['image']:

                self._image = tf.placeholder(tf.float32, [batch_size] + config['image']["dim"], name='image')

                self.visual_features = get_image_features(image=self._image,
                                                          is_training=self._is_training,
                                                          config=config['image'])

                if "film_block" in config:

                    with tf.variable_scope("image_reading_unit", reuse=reuse):
                        reading_unit = create_reading_unit(last_state=self.dialogue_last_states,
                                                           states=self.dialogue_states,
                                                           seq_length=self._seq_length_dialogue,
                                                           config=config["film_input"]["reading_unit"],
                                                           reuse=reuse)

                        film_layer_fct = create_film_layer_with_reading_unit(reading_unit)

                    with tf.variable_scope("image_film_stack", reuse=reuse):
                        film_stack = FiLM_Stack(image=self.visual_features,
                                                film_input=[],
                                                film_layer_fct=film_layer_fct,
                                                is_training=self._is_training,
                                                config=config["film_block"],
                                                reuse=reuse)

                        self.visual_features = film_stack.get()

                # Pool Image Features
                with tf.variable_scope("image_pooling"):
                    self.visual_features = get_attention(self.visual_features, self.dialogue_last_states,
                                                         is_training=self._is_training,
                                                         config=config["pooling"],
                                                         dropout_keep=dropout_keep,
                                                         reuse=reuse)

            else:
                self.visual_features = None

            # fuse vision/language
            with tf.variable_scope("multimodal_fusion"):
                self.visdiag_embedding = get_fusion_mechanism(input1=self.dialogue_last_states,
                                                              input2=self.visual_features,
                                                              config=config["fusion"],
                                                              dropout_keep=dropout_keep)

            #####################
            #   TARGET QUESTION
            #####################

            self._question = tf.placeholder(tf.int32, [batch_size, None], name='question')
            self._seq_length_question = tf.placeholder(tf.int32, [batch_size], name='seq_length_question')

            input_question = self._question[:, :-1]  # Ignore start token
            target_question = self._question[:, 1:]  # Ignore stop token

            self._question_mask = tf.sequence_mask(lengths=self._seq_length_question - 1, dtype=tf.float32)  # -1 : remove start token a decoding time

            if config["dialogue"]["share_decoder_emb"]:
                self.question_emb_weights = self.dialogue_emb_weights
            else:
                self.question_emb_weights = tf.get_variable("question_embedding_encoder",
                                                            shape=[num_words, config["dialogue"]["word_embedding_dim"]])
            self.word_emb_question = tf.nn.embedding_lookup(params=self.question_emb_weights, ids=input_question)
            self.word_emb_question = tf.nn.dropout(self.word_emb_question, dropout_keep)

            #####################
            #   DECODER
            #####################

            self.decoder_cell = rnn.create_cell(cell=config['decoder']["cell"],
                                                num_units=int(self.visdiag_embedding.shape[-1]),
                                                layer_norm=config["decoder"]["layer_norm"],
                                                reuse=reuse)

            self.decoder_projection_layer = tf.layers.Dense(num_words, use_bias=False)

            training_helper = tfc_seq.TrainingHelper(inputs=self.word_emb_question,  # The question is the target
                                                     sequence_length=self._seq_length_question - 1)  # -1 : remove start token at decoding time
            decoder = BasicDecoderWithState(cell=self.decoder_cell,
                                            helper=training_helper,
                                            initial_state=self.visdiag_embedding,
                                            output_layer=self.decoder_projection_layer)

            (self.decoder_outputs, self.decoder_states, _), _, _ = tfc_seq.dynamic_decode(decoder, maximum_iterations=None)

            #####################
            #   LOSS
            #####################

            # compute the softmax for evaluation
            with tf.variable_scope('ml_loss'):

                self.ml_loss = tfc_seq.sequence_loss(logits=self.decoder_outputs,
                                                     targets=target_question,
                                                     weights=self._question_mask,
                                                     average_across_timesteps=True,
                                                     average_across_batch=True)

                self.softmax_output = tf.nn.softmax(self.decoder_outputs, name="softmax")
                self.argmax_output = tf.argmax(self.decoder_outputs, axis=2)

                self.loss = self.ml_loss

            # Compute policy gradient
            if policy_gradient:
                self._cum_rewards = tf.placeholder(tf.float32, shape=[batch_size, None], name='cum_reward')[:, 1:]

                with tf.variable_scope('rl_baseline'):
                    baseline_input = tf.stop_gradient(self.decoder_states)

                    baseline_hidden = tfc_layers.fully_connected(baseline_input,
                                                                 num_outputs=int(int(baseline_input.get_shape()[-1]) / 4),
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
                    self.baseline_loss *= self._question_mask

                    self.baseline_loss = tf.reduce_sum(self.baseline_loss, axis=0)
                    self.baseline_loss = tf.reduce_mean(self.baseline_loss, axis=1)

                with tf.variable_scope('policy_gradient_loss'):
                    self.log_of_policy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.decoder_outputs, labels=target_question)

                    self.policy_gradient_loss = tf.multiply(self.log_of_policy, self._cum_rewards - self.baseline)  # score function
                    self.policy_gradient_loss *= self._question_mask

                    self.policy_gradient_loss = tf.reduce_sum(self.policy_gradient_loss, axis=1)  # sum over the dialogue trajectory
                    self.policy_gradient_loss = tf.reduce_mean(self.policy_gradient_loss, axis=0)  # reduce over minibatch dimension

                    self.loss = self.policy_gradient_loss

    def create_sampling_graph(self, start_token, stop_token, max_tokens):

        batch_size = tf.shape(self._dialogue)[0]
        sample_helper = tfc_seq.SampleEmbeddingHelper(embedding=self.question_emb_weights,
                                                      start_tokens=tf.fill([batch_size], start_token),
                                                      end_token=stop_token)

        decoder = tfc_seq.BasicDecoder(
            self.decoder_cell, sample_helper, self.visdiag_embedding,
            output_layer=self.decoder_projection_layer)

        (_, sample_id), _, seq_length = tfc_seq.dynamic_decode(decoder, maximum_iterations=max_tokens)

        return sample_id, seq_length

    def create_greedy_graph(self, start_token, stop_token, max_tokens):

        batch_size = tf.shape(self._dialogue)[0]
        greedy_helper = tfc_seq.GreedyEmbeddingHelper(embedding=self.question_emb_weights,
                                                      start_tokens=tf.fill([batch_size], start_token),
                                                      end_token=stop_token)

        decoder = tfc_seq.BasicDecoder(
            self.decoder_cell, greedy_helper, self.visdiag_embedding,
            output_layer=self.decoder_projection_layer)

        (_, sample_id), _, seq_length = tfc_seq.dynamic_decode(decoder, maximum_iterations=max_tokens)

        return sample_id, seq_length

    def create_beam_graph(self, start_token, stop_token, max_tokens, k_best):

        # create k_beams
        decoder_initial_state = tf.contrib.seq2seq.tile_batch(
            self.visdiag_embedding, multiplier=k_best)

        # Define a beam-search decoder
        batch_size = tf.shape(self._dialogue)[0]
        decoder = tfc_seq.BeamSearchDecoder(
            cell=self.decoder_cell,
            embedding=self.question_emb_weights,
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

    with open("../../../../config/qgen/config.baseline.json", 'rb') as f_config:
        config = json.load(f_config)

    network = QGenNetworkDecoder(config["model"], num_words=111, policy_gradient=True)

    network.create_sampling_graph(start_token=1, stop_token=2, max_tokens=10)
    network.create_greedy_graph(start_token=1, stop_token=2, max_tokens=10)
    network.create_beam_graph(start_token=1, stop_token=2, max_tokens=10, k_best=5)
