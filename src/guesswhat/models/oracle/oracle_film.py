import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers

import neural_toolbox.ft_utils as ft_utils
import neural_toolbox.rnn as rnn

from generic.tf_factory.image_factory import get_image_features
from generic.tf_factory.attention_factory import get_attention
from generic.tf_utils.abstract_network import ResnetModel
from generic.utils.config import get_recursively

from neural_toolbox.film_stack import FiLM_Stack
from neural_toolbox.reading_unit import create_reading_unit, create_film_layer_with_reading_unit


class FiLM_Oracle(ResnetModel):
    def __init__(self, config, no_words, no_answers, reuse=False, device=''):
        ResnetModel.__init__(self, "oracle", device=device)

        with tf.variable_scope(self.scope_name, reuse=reuse):

            self.batch_size = None
            self._is_training = tf.placeholder(tf.bool, name="is_training")

            dropout_keep_scalar = float(config["dropout_keep_prob"])
            dropout_keep = tf.cond(self._is_training,
                                   lambda: tf.constant(dropout_keep_scalar),
                                   lambda: tf.constant(1.0))

            #####################
            #   QUESTION
            #####################

            self._question = tf.placeholder(tf.int32, [self.batch_size, None], name='question')
            self._seq_length = tf.placeholder(tf.int32, [self.batch_size], name='seq_length')
            self._answer = tf.placeholder(tf.int64, [self.batch_size, no_answers], name='answer')

            word_emb = tfc_layers.embed_sequence(
                ids=self._question,
                vocab_size=no_words,
                embed_dim=config["question"]["word_embedding_dim"],
                scope="word_embedding",
                reuse=reuse)

            if config["question"]['glove']:
                self._glove = tf.placeholder(tf.float32, [None, None, 300], name="glove")
                word_emb = tf.concat([word_emb, self._glove], axis=2)

            word_emb = tf.nn.dropout(word_emb, dropout_keep)
            self.rnn_states, self.last_rnn_states = rnn.rnn_factory(
                inputs=word_emb,
                seq_length=self._seq_length,
                cell=config["question"]["cell"],
                num_hidden=config["question"]["rnn_state_size"],
                bidirectional=config["question"]["bidirectional"],
                max_pool=config["question"]["max_pool"],
                layer_norm=config["question"]["layer_norm"],
                reuse=reuse)

            #####################
            #   SIDE INPUTS
            #####################

            # Category
            if any(get_recursively(config, "category", no_field_recursive=True)):
                self._category = tf.placeholder(tf.int32, [self.batch_size], name='category')

                cat_emb = tfc_layers.embed_sequence(
                    ids=self._category,
                    vocab_size=config['category']["n_categories"] + 1,
                    embed_dim=config['category']["embedding_dim"],
                    scope="category_embedding",
                    reuse=reuse)
                cat_emb = tf.nn.dropout(cat_emb, dropout_keep)
            else:
                cat_emb = None

            # Spatial
            if any(get_recursively(config, "spatial", no_field_recursive=True)):
                self._spatial = tf.placeholder(tf.float32, [self.batch_size, 8], name='spatial')
                spatial_emb = tfc_layers.fully_connected(self._spatial,
                                                         num_outputs=config["spatial"]["no_mlp_units"],
                                                         activation_fn=tf.nn.relu,
                                                         reuse=reuse,
                                                         scope="spatial_upsampling")
                spatial_emb = tf.nn.dropout(spatial_emb, dropout_keep)
            else:
                spatial_emb = None

            self.classifier_input = []

            #####################
            #   IMAGES / CROP
            #####################

            for visual_str in ["image", "crop"]:

                # Check whether to use the visual input
                if config["inputs"][visual_str]:

                    # Load Image Features
                    visual_features = tf.placeholder(tf.float32, shape=[self.batch_size] + config[visual_str]["dim"], name=visual_str)
                    with tf.variable_scope(visual_str, reuse=reuse):
                        visual_features = get_image_features(image=visual_features,
                                                             config=config[visual_str],
                                                             is_training=self._is_training)

                    # Modulate Image Features
                    if "film_input" in config:

                        # Retrieve configuration
                        film_config = config["film_input"]
                        block_config = config["film_block"]

                        # Load object mask
                        mask = tf.placeholder(tf.float32, visual_features.get_shape()[:3], name='{}_mask'.format(visual_str))
                        mask = tf.expand_dims(mask, axis=-1)

                        # Perform the actual modulation
                        with tf.variable_scope("{}_modulation".format(visual_str)):

                            extra_context = []
                            with tf.variable_scope("{}_film_input".format(visual_str), reuse=reuse):

                                if film_config["category"]:
                                    extra_context.append(cat_emb)

                                if film_config["spatial"]:
                                    extra_context.append(spatial_emb)

                                if film_config["mask"]:
                                    mask_dim = int(visual_features.get_shape()[1]) * int(visual_features.get_shape()[2])
                                    flat_mask = tf.reshape(mask, shape=[-1, mask_dim])
                                    extra_context.append(flat_mask)

                            with tf.variable_scope("{}_reading_cell".format(visual_str)):

                                reading_unit = create_reading_unit(last_state=self.last_rnn_states,
                                                                   states=self.rnn_states,
                                                                   seq_length=self._seq_length,
                                                                   config=film_config["reading_unit"],
                                                                   reuse=reuse)

                                film_layer_fct = create_film_layer_with_reading_unit(reading_unit)

                            with tf.variable_scope("{}_film_stack".format(visual_str), reuse=reuse):

                                def append_extra_features(features, config):
                                    if config["spatial_location"]:  # add the pixel location as two additional feature map
                                        features = ft_utils.append_spatial_location(features)
                                    if config["mask"]:  # add the mask on the object as one additional feature map
                                        features = tf.concat([features, mask], axis=3)
                                    return features

                                film_stack = FiLM_Stack(image=visual_features,
                                                        film_input=extra_context,
                                                        film_layer_fct=film_layer_fct,
                                                        is_training=self._is_training,
                                                        config=block_config,
                                                        append_extra_features=append_extra_features,
                                                        reuse=reuse)

                                visual_features = film_stack.get()

                    # Pool Image Features
                    if len(visual_features.get_shape()) > 2:
                        with tf.variable_scope("{}_pooling".format(visual_str)):
                            visual_features = get_attention(visual_features, self.last_rnn_states,
                                                            is_training=self._is_training,
                                                            config=config["pooling"],
                                                            dropout_keep=dropout_keep,
                                                            reuse=reuse)

                    self.classifier_input.append(visual_features)

            #####################
            #   FINAL LAYER
            #####################

            with tf.variable_scope("classifier", reuse=reuse):

                if config["classifier"]["inputs"]["question"]:
                    self.classifier_input.append(self.last_rnn_states)

                if config["classifier"]["inputs"]["category"]:
                    self.classifier_input.append(cat_emb)

                if config["classifier"]["inputs"]["spatial"]:
                    self.classifier_input.append(spatial_emb)

                assert len(self.classifier_input) > 0, "Please provide some inputs for the classifier!!!"
                self.classifier_input = tf.concat(self.classifier_input, axis=1)

                self.hidden_state = tfc_layers.fully_connected(self.classifier_input,
                                                               num_outputs=config["classifier"]["no_mlp_units"],
                                                               activation_fn=tf.nn.relu,
                                                               reuse=reuse,
                                                               scope="classifier_hidden_layer")

                self.hidden_state = tf.nn.dropout(self.hidden_state, dropout_keep)
                self.out = tfc_layers.fully_connected(self.hidden_state,
                                                      num_outputs=no_answers,
                                                      activation_fn=None,
                                                      reuse=reuse,
                                                      scope="classifier_softmax_layer")

            #####################
            #   Loss
            #####################

            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.out, labels=self._answer, name='cross_entropy')
            self.loss = tf.reduce_mean(self.cross_entropy)

            self.softmax = tf.nn.softmax(self.out, name='answer_prob')
            self.prediction = tf.argmax(self.out, axis=1, name='predicted_answer')  # no need to compute the softmax

            self.success = tf.equal(self.prediction, tf.argmax(self._answer, axis=1))  # no need to compute the softmax

            with tf.variable_scope('accuracy'):
                self.accuracy = tf.equal(self.prediction, tf.argmax(self._answer, axis=1))
                self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

            tf.summary.scalar('accuracy', self.accuracy)

            print('Model... build!')

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy


if __name__ == "__main__":

    import json
    with open("../../../config/referit/config.2.json", 'r') as f_config:
        config = json.load(f_config)

    get_recursively(config, "spatial", no_field_recursive=True)

    FiLM_Oracle(config["model"], no_words=354, no_answers=3)
