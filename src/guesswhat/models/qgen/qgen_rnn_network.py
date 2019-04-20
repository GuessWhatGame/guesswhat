import tensorflow as tf

import tensorflow.contrib.layers as tfc_layers

from neural_toolbox import rnn

from generic.tf_factory.image_factory import get_image_features
from generic.tf_factory.attention_factory import get_attention
from generic.tf_utils.abstract_network import AbstractNetwork

from guesswhat.models.qgen.qgen_utils import *


class QGenNetworkRNN(AbstractNetwork):

    def __init__(self, config, num_words, rl_module=None, device='', reuse=False):
        AbstractNetwork.__init__(self, "qgen", device=device)

        self.rl_module = rl_module

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
            self.zero_states = self.decoder_cell.zero_state(tf.shape(self._seq_length_dialogue)[0], dtype=tf.float32)

            self.decoder_states, self.decoder_final_state = tf.nn.dynamic_rnn(
                cell=self.decoder_cell,
                inputs=visword_emb_dialogue,
                dtype=tf.float32,
                initial_state=self.zero_states,
                sequence_length=self._seq_length_dialogue,
                scope="decoder")

            self.decoder_projection_layer = tf.layers.Dense(num_words)
            self.decoder_outputs = self.decoder_projection_layer(self.decoder_states)

            # Prepare sampling graph TODO: make it clean
            self._new_answer = tf.placeholder_with_default(
                tf.zeros_like(self._seq_length_dialogue), shape=[batch_size], name='new_answer')

            #####################
            #   LOSS
            #####################

            # ignore answers while computing cross entropy or applying reward
            self.mask = (self._question_mask - self._answer_mask)

            # compute the softmax for evaluation
            with tf.variable_scope('ml_loss'):

                self.ml_loss = tfc_seq.sequence_loss(logits=self.decoder_outputs,
                                                     targets=target_dialogue,
                                                     weights=self.mask,
                                                     average_across_timesteps=True,
                                                     average_across_batch=True)

                self.softmax_output = tf.nn.softmax(self.decoder_outputs, name="softmax")
                self.argmax_output = tf.argmax(self.decoder_outputs, axis=2)

                self.loss = self.ml_loss

            # Compute policy gradient
            if self.rl_module is not None:

                # Step 1: compute the state-value function
                self._cum_rewards = tf.placeholder(tf.float32, shape=[batch_size, None], name='cum_reward')[:, 1:]

                # Step 2: compute the state-value function
                value_state = self.decoder_states
                if self.rl_module.stop_gradient:
                    value_state = tf.stop_gradient(self.decoder_states)
                v_hidden_units = int(int(value_state.get_shape()[-1]) / 4)

                with tf.variable_scope('value_function'):
                    self.value_function = tf.keras.models.Sequential()
                    self.value_function.add(tf.layers.Dense(units=v_hidden_units,
                                                            activation=tf.nn.relu,
                                                            input_shape=(int(value_state.get_shape()[-1]),),
                                                            name="value_function_hidden"))
                    self.value_function.add(tf.layers.Dense(units=1,
                                                            activation=None,
                                                            name="value_function"))
                    self.value_function.add(tf.keras.layers.Reshape((-1,)))

                # Step 3: compute the RL loss (reinforce, A3C, PPO etc.)
                self.loss = rl_module(cum_rewards=self._cum_rewards,
                                      value_function=self.value_function(value_state),
                                      policy_state=self.decoder_outputs,
                                      actions=target_dialogue,
                                      action_mask=self.mask)

    def create_sampling_graph(self, start_token, stop_token, max_tokens):

        def embedding(idx):
            emb = tf.nn.embedding_lookup(params=self.dialogue_emb_weights, ids=idx)
            emb = tf.concat([emb, self.visual_features], axis=-1)
            return emb

        sample_helper = tfc_seq.SampleEmbeddingHelper(embedding=embedding,
                                                      start_tokens=self._new_answer,
                                                      end_token=stop_token)

        start_dialogue = tf.equal(self._new_answer, start_token)
        initial_state = tf.where(start_dialogue, self.zero_states, self.decoder_final_state)

        decoder = BasicDecoderWithState(cell=self.decoder_cell,
                                        helper=sample_helper,
                                        initial_state=initial_state,
                                        output_layer=self.decoder_projection_layer)

        (outputs, states, sample_id), _, seq_length = tfc_seq.dynamic_decode(decoder, maximum_iterations=max_tokens)

        state_value = sample_id * 0
        if self.rl_module is not None:
            state_value = self.value_function(states)

        return sample_id, seq_length, state_value

    def create_greedy_graph(self, start_token, stop_token, max_tokens):

        def embedding(idx):
            emb = tf.nn.embedding_lookup(params=self.dialogue_emb_weights, ids=idx)
            emb = tf.concat([emb, self.visual_features], axis=-1)
            return emb

        greedy_helper = tfc_seq.GreedyEmbeddingHelper(embedding=embedding,
                                                      start_tokens=self._new_answer,
                                                      end_token=stop_token)

        start_dialogue = tf.equal(self._new_answer, start_token)
        initial_state = tf.where(start_dialogue, self.zero_states, self.decoder_final_state)

        decoder = BasicDecoderWithState(cell=self.decoder_cell,
                                        helper=greedy_helper,
                                        initial_state=initial_state,
                                        output_layer=self.decoder_projection_layer)

        (outputs, states, sample_id), _, seq_length = tfc_seq.dynamic_decode(decoder, maximum_iterations=max_tokens)

        state_value = sample_id * 0
        if self.rl_module is not None:
            state_value = self.value_function(states)

        return sample_id, seq_length, state_value

    def create_beam_graph(self, start_token, stop_token, max_tokens, k_best):

        # create k_beams
        start_dialogue = tf.equal(self._new_answer, start_token)
        decoder_initial_state = tf.where(start_dialogue, self.zero_states, self.decoder_final_state)
        decoder_initial_state = tf.contrib.seq2seq.tile_batch(decoder_initial_state, multiplier=k_best)

        def embedding(idx):
            emb = tf.nn.embedding_lookup(params=self.dialogue_emb_weights, ids=idx)

            visual_features = tf.expand_dims(self.visual_features, axis=1)
            visual_features = tf.tile(visual_features, multiples=[1, tf.shape(idx)[1], 1])

            emb = tf.concat([emb, visual_features], axis=-1)
            return emb

        # Define a beam-search decoder
        decoder = tfc_seq.BeamSearchDecoder(
            cell=self.decoder_cell,
            embedding=embedding,
            start_tokens=self._new_answer,
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

    @staticmethod
    def is_seq2seq():
        return False


if __name__ == "__main__":
    import json

    with open("../../../../config/qgen/config.rnn.json", 'rb') as f_config:
        config = json.load(f_config)

    network = QGenNetworkRNN(config["model"], num_words=111, policy_gradient=True)

    network.create_sampling_graph(start_token=1, stop_token=2, max_tokens=10)
    network.create_greedy_graph(start_token=1, stop_token=2, max_tokens=10)
    network.create_beam_graph(start_token=1, stop_token=2, max_tokens=10, k_best=5)
