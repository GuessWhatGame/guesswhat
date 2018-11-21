import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers

from neural_toolbox import rnn

from generic.tf_factory.image_factory import get_image_features
from generic.tf_factory.attention_factory import get_attention
from generic.tf_factory.fusion_factory import get_fusion_mechanism
from generic.tf_utils.abstract_network import AbstractNetwork


class GuesserNetwork(AbstractNetwork):
    def __init__(self, config, num_words, device='', reuse=False):
        AbstractNetwork.__init__(self, "guesser", device=device)

        batch_size = None

        self._is_training = tf.placeholder(tf.bool, name="is_training")

        dropout_keep_scalar = float(config['regularizer'].get("dropout_keep_prob", 1.0))
        dropout_keep = tf.cond(self._is_training,
                               lambda: tf.constant(dropout_keep_scalar),
                               lambda: tf.constant(1.0))

        with tf.variable_scope(self.scope_name, reuse=reuse):

            #####################
            #   DIALOGUE
            #####################

            self._dialogue = tf.placeholder(tf.int32, [batch_size, None], name='dialogue')
            self._seq_length = tf.placeholder(tf.int32, [batch_size], name='seq_length')

            word_emb = tfc_layers.embed_sequence(ids=self._dialogue,
                                                 vocab_size=num_words,
                                                 embed_dim=config["question"]["word_embedding_dim"],
                                                 scope="word_embedding",
                                                 reuse=reuse)

            _, self.dialogue_embedding = rnn.rnn_factory(inputs=word_emb,
                                                         seq_length=self._seq_length,
                                                         cell=config['question']["cell"],
                                                         num_hidden=config['question']["rnn_units"],
                                                         bidirectional=config["question"]["bidirectional"],
                                                         max_pool=config["question"]["max_pool"],
                                                         layer_norm=config["question"]["layer_norm"],
                                                         reuse=reuse)

            #####################
            #   IMAGES
            #####################

            self.img_embedding = None
            if 'image' in config['image']:

                self._image = tf.placeholder(tf.float32, [batch_size] + config['image']["dim"], name='image')

                self.img_embedding = get_image_features(image=self._image,
                                                        is_training=self._is_training,
                                                        config=config['image'])

                if len(self.img_embedding.get_shape()) > 2:
                    with tf.variable_scope("image_pooling"):
                        self.img_embedding = get_attention(self.img_embedding, self.dialogue_embedding,
                                                           is_training=self._is_training,
                                                           config=config["pooling"],
                                                           dropout_keep=dropout_keep,
                                                           reuse=reuse)

            #####################
            #   FUSION
            #####################

            self.visdiag_embedding = get_fusion_mechanism(input1=self.dialogue_embedding,
                                                          input2=self.img_embedding,
                                                          config=config["fusion"],
                                                          dropout_keep=dropout_keep)

            visdiag_dim = int(self.visdiag_embedding.get_shape()[-1])

            #####################
            #   OBJECTS
            #####################

            self._num_object = tf.placeholder(tf.int32, [batch_size], name='obj_seq_length')
            self._obj_cats = tf.placeholder(tf.int32, [batch_size, None], name='obj_cat')
            self._obj_spats = tf.placeholder(tf.float32, [batch_size, None, 8], name='obj_spat')

            self.object_cats_emb = tfc_layers.embed_sequence(ids=self._obj_cats,
                                                             vocab_size=config['category']["n_categories"] + 1,  # we add the unkwown category
                                                             embed_dim=config['category']["embedding_dim"],
                                                             scope="cat_embedding",
                                                             reuse=reuse)

            self.objects_input = tf.concat([self.object_cats_emb, self._obj_spats], axis=2)
            self.flat_objects_inp = tf.reshape(self.objects_input, [-1, config['cat_emb_dim'] + config['spat_dim']])

            with tf.variable_scope('obj_mlp'):
                h1 = tfc_layers.fully_connected(self.flat_objects_inp,
                                                num_outputs=config['obj_mlp_units'],
                                                activation_fn=tf.nn.relu,
                                                scope='l1')

                h2 = tfc_layers.fully_connected(h1,
                                                num_outputs=visdiag_dim,
                                                activation_fn=tf.nn.relu,
                                                scope='l2')

                obj_embs = tf.reshape(h2, [-1, tf.shape(self._obj_cats)[1], visdiag_dim])

            #####################
            #   SCORES
            #####################

            self.scores = obj_embs * tf.expand_dims(visdiag_dim, axis=-1)
            self.scores = tf.reshape(self.scores, [-1, tf.shape(self._obj_cats)[1]])

            with tf.variable_scope('object_mask', reuse=reuse):

                object_mask = tf.sequence_mask(self._num_object)
                score_mask_values = float("-inf") * tf.ones_like(self.scores)

                self.score_masked = tf.where(object_mask, self.scores, score_mask_values)

            #####################
            #   LOSS
            #####################

            # Targets
            self._targets = tf.placeholder(tf.int32, [batch_size], name="target_index")

            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._targets, logits=self.score_masked)
            self.loss = tf.reduce_mean(self.loss)

            self.selected_object = tf.argmax(self.score_masked, axis=1)

            with tf.variable_scope('accuracy'):
                self.accuracy = tf.equal(self.selected_object, tf.cast(self._targets,  tf.int64))
                self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))


    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return 1. - self.error

