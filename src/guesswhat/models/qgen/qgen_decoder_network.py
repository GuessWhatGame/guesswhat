import tensorflow as tf

from neural_toolbox import rnn

from generic.tf_factory.image_factory import get_image_features
from generic.tf_factory.fusion_factory import get_fusion_mechanism
from generic.tf_factory.attention_factory import get_attention

from generic.tf_utils.abstract_network import AbstractNetwork
from guesswhat.models.qgen.qgen_utils import *

from neural_toolbox.reading_unit import create_reading_unit, create_film_layer_with_reading_unit
from neural_toolbox.film_stack import FiLM_Stack


class QGenNetworkDecoder(AbstractNetwork):

    def __init__(self, config, num_words, rl_module, device='', reuse=False):
        AbstractNetwork.__init__(self, "qgen", device=device)

        self.rl_module = rl_module

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

            self.mask = tf.sequence_mask(lengths=self._seq_length_question - 1, dtype=tf.float32)  # -1 : remove start token a decoding time

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

            assert config['decoder']["cell"] != "lstm", "LSTM are not yet supported for the decoder"
            self.decoder_cell = rnn.create_cell(cell=config['decoder']["cell"],
                                                num_units=int(self.visdiag_embedding.shape[-1]),
                                                layer_norm=config["decoder"]["layer_norm"],
                                                reuse=reuse)

            self.decoder_states, _ = tf.nn.dynamic_rnn(
                cell=self.decoder_cell,
                inputs=self.word_emb_question,
                dtype=tf.float32,
                initial_state=self.visdiag_embedding,
                sequence_length=self._seq_length_question - 1,
                scope="decoder")

            self.decoder_projection_layer = tf.layers.Dense(num_words)
            self.decoder_outputs = self.decoder_projection_layer(self.decoder_states)

            #####################
            #   LOSS
            #####################

            # compute the softmax for evaluation
            with tf.variable_scope('ml_loss'):

                self.ml_loss = tfc_seq.sequence_loss(logits=self.decoder_outputs,
                                                     targets=target_question,
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
                self._cum_rewards *= self.mask

                # Step 2: compute the state-value function
                value_state = self.decoder_states
                if self.rl_module.stop_gradient:
                    value_state = tf.stop_gradient(self.decoder_states)
                v_num_hidden_units = int(int(value_state.get_shape()[-1]) / 4)

                with tf.variable_scope('value_function'):
                    self.value_function = tf.keras.models.Sequential()
                    self.value_function.add(tf.layers.Dense(units=v_num_hidden_units,
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
                                      actions=target_question,
                                      action_mask=self.mask)

    def create_sampling_graph(self, start_token, stop_token, max_tokens):

        batch_size = tf.shape(self._dialogue)[0]
        sample_helper = tfc_seq.SampleEmbeddingHelper(embedding=self.question_emb_weights,
                                                      start_tokens=tf.fill([batch_size], start_token),
                                                      end_token=stop_token)

        decoder = BasicDecoderWithState(cell=self.decoder_cell,
                                        helper=sample_helper,
                                        initial_state=self.visdiag_embedding,
                                        output_layer=self.decoder_projection_layer)

        (_, states, sample_id), _, seq_length = tfc_seq.dynamic_decode(decoder, maximum_iterations=max_tokens)

        state_value = sample_id * 0
        if self.rl_module is not None:
            state_value = self.value_function(states)

        return sample_id, seq_length, state_value

    def create_greedy_graph(self, start_token, stop_token, max_tokens):

        batch_size = tf.shape(self._dialogue)[0]
        greedy_helper = tfc_seq.GreedyEmbeddingHelper(embedding=self.question_emb_weights,
                                                      start_tokens=tf.fill([batch_size], start_token),
                                                      end_token=stop_token)

        decoder = BasicDecoderWithState(cell=self.decoder_cell,
                                        helper=greedy_helper,
                                        initial_state=self.visdiag_embedding,
                                        output_layer=self.decoder_projection_layer)

        (_, states, sample_id), _, seq_length = tfc_seq.dynamic_decode(decoder, maximum_iterations=max_tokens)

        state_value = sample_id * 0
        if self.rl_module is not None:
            state_value = self.value_function(states)

        return sample_id, seq_length, state_value

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

    @staticmethod
    def is_seq2seq():
        return False


if __name__ == "__main__":
    import json

    with open("../../../../config/qgen/config.baseline.json", 'rb') as f_config:
        config = json.load(f_config)

    network = QGenNetworkDecoder(config["model"], num_words=111, policy_gradient=True)

    network.create_sampling_graph(start_token=1, stop_token=2, max_tokens=10)
    network.create_greedy_graph(start_token=1, stop_token=2, max_tokens=10)
    network.create_beam_graph(start_token=1, stop_token=2, max_tokens=10, k_best=5)
