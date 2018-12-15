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

        with tf.variable_scope(self.scope_name, reuse=reuse):

            self._is_training = tf.placeholder(tf.bool, name="is_training")

            dropout_keep_scalar = float(config["dropout_keep_prob"])
            dropout_keep = tf.cond(self._is_training,
                                   lambda: tf.constant(dropout_keep_scalar),
                                   lambda: tf.constant(1.0))

            #####################
            #   DIALOGUE
            #####################

            self._dialogue = tf.placeholder(tf.int32, [batch_size, None], name='dialogue')
            self._seq_length = tf.placeholder(tf.int32, [batch_size], name='seq_length_dialogue')

            word_emb = tfc_layers.embed_sequence(ids=self._dialogue,
                                                 vocab_size=num_words,
                                                 embed_dim=config["question"]["word_embedding_dim"],
                                                 scope="word_embedding",
                                                 reuse=reuse)

            if config["question"]['glove']:
                self._glove = tf.placeholder(tf.float32, [None, None, 300], name="glove")
                word_emb = tf.concat([word_emb, self._glove], axis=2)

            _, self.dialogue_embedding = rnn.rnn_factory(inputs=word_emb,
                                                         seq_length=self._seq_length,
                                                         cell=config['question']["cell"],
                                                         num_hidden=config['question']["rnn_units"],
                                                         bidirectional=config["question"]["bidirectional"],
                                                         max_pool=config["question"]["max_pool"],
                                                         layer_norm=config["question"]["layer_norm"],
                                                         reuse=reuse)

            #####################
            #   IMAGE
            #####################

            self.img_embedding = None
            if config['inputs']['image']:

                self._image = tf.placeholder(tf.float32, [batch_size] + config['image']["dim"], name='image')

                # get image
                self.img_embedding = get_image_features(image=self._image,
                                                        is_training=self._is_training,
                                                        config=config['image'])

                # pool image feature if needed
                if len(self.img_embedding.get_shape()) > 2:
                    with tf.variable_scope("image_pooling"):
                        self.img_embedding = get_attention(self.img_embedding, self.dialogue_embedding,
                                                           is_training=self._is_training,
                                                           config=config["pooling"],
                                                           dropout_keep=dropout_keep,
                                                           reuse=reuse)

                # fuse vision/language
                self.visdiag_embedding = get_fusion_mechanism(input1=self.dialogue_embedding,
                                                              input2=self.img_embedding,
                                                              config=config.get["fusion"],
                                                              dropout_keep=dropout_keep)
            else:
                self.visdiag_embedding = self.dialogue_embedding

            visdiag_dim = int(self.visdiag_embedding.get_shape()[-1])

            #####################
            #   OBJECTS
            #####################

            self._num_object = tf.placeholder(tf.int32, [batch_size], name='obj_seq_length')
            self._obj_cats = tf.placeholder(tf.int32, [batch_size, None], name='obj_cat')
            self._obj_spats = tf.placeholder(tf.float32, [batch_size, None, 8], name='obj_spat')

            cats_emb = tfc_layers.embed_sequence(ids=self._obj_cats,
                                                 vocab_size=config['category']["n_categories"] + 1,  # we add the unknown category
                                                 embed_dim=config['category']["embedding_dim"],
                                                 scope="cat_embedding",
                                                 reuse=reuse)

            spatial_emb = tfc_layers.fully_connected(self._obj_spats,
                                                     num_outputs=config["spatial"]["no_mlp_units"],
                                                     activation_fn=tf.nn.relu,
                                                     reuse=reuse,
                                                     scope="spatial_upsampling")

            self.objects_input = tf.concat([cats_emb, spatial_emb], axis=2)
            self.objects_input = tf.nn.dropout(self.objects_input, dropout_keep)

            with tf.variable_scope('obj_mlp'):
                h1 = tfc_layers.fully_connected(self.objects_input,
                                                num_outputs=config["object"]['no_mlp_units'],
                                                activation_fn=tf.nn.relu,
                                                scope='l1')
                h1 = tf.nn.dropout(h1, dropout_keep)

                obj_embeddings = tfc_layers.fully_connected(h1,
                                                            num_outputs=visdiag_dim,
                                                            activation_fn=tf.nn.relu,
                                                            scope='l2')

            #####################
            #   SCORES
            #####################

            self.scores = obj_embeddings * tf.expand_dims(self.visdiag_embedding, axis=1)
            self.scores = tf.reduce_sum(self.scores, axis=2)

            # remove max for stability (trick)
            self.scores -= tf.reduce_max(self.scores, axis=1, keepdims=True)

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
            self.softmax = tf.nn.softmax(self.score_masked)

            with tf.variable_scope('accuracy'):
                self.accuracy = tf.equal(self.selected_object, tf.cast(self._targets, tf.int64))
                self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy


if __name__ == "__main__":

    import json
    with open("../../../../config/guesser/config.config.baseline.json", 'rb') as f_config:
        config = json.loads(f_config.read().decode('utf-8'))

    network = GuesserNetwork(config["model"], num_words=352)
    print("Ok!!")
