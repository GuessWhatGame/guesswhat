import tensorflow as tf

from generic.tensorflow import rnn, mlp, attention
from generic.tensorflow.abstract_model import AbstractModel


class OracleNetwork(AbstractModel):

    def __init__(self, config, num_words, log=print, device='', reuse=False):
        AbstractModel.__init__(self, "oracle", device=device)
        self.config = config

        with tf.variable_scope(self.scope_name, reuse=reuse):
            embeddings = []
            self.batch_size = None

            # QUESTION
            self._is_training = tf.placeholder(tf.bool, name="is_training")
            self._question = tf.placeholder(tf.int32, [self.batch_size, None], name='question')
            self._seq_length = tf.placeholder(tf.int32, [self.batch_size], name='seq_length')

            word_emb = mlp.get_embedding(self._question,
                                           n_words=num_words,
                                           n_dim=int(config['model']['question']["embedding_dim"]),
                                           scope="word_embedding")

            lstm_states, _ = rnn.variable_length_LSTM(word_emb,
                                                   num_hidden=int(config['model']['question']["no_LSTM_hiddens"]),
                                                   seq_length=self._seq_length)
            embeddings.append(lstm_states)

            # CATEGORY
            if config['inputs']['category']:
                self._category = tf.placeholder(tf.int32, [self.batch_size], name='category')

                cat_emb = mlp.get_embedding(self._category,
                                              int(config['model']['category']["n_categories"]) + 1,  # we add the unkwon category
                                              int(config['model']['category']["embedding_dim"]),
                                              scope="cat_embedding")
                embeddings.append(cat_emb)
                log("Input: Category")

            # SPATIAL
            if config['inputs']['spatial']:
                self._spatial = tf.placeholder(tf.float32, [self.batch_size, 8], name='spatial')
                embeddings.append(self._spatial)
                log("Input: Spatial")

            # CROP
            if config['inputs']['crop']:
                self._crop_fc8 = tf.placeholder(tf.float32, [self.batch_size, 1000], name='crop_fc8')
                embeddings.append(self._crop_fc8)
                log("Input: Crop")

            # IMAGE
            if config['inputs']['picture']:
                self._picture_fc8 = tf.placeholder(tf.float32, [self.batch_size, 1000], name='picture_fc8')
                embeddings.append(self._picture_fc8)
                log("Input: Image")

            # Compute the final embedding
            emb = tf.concat(embeddings, 1)

            # OUTPUT
            num_classes = 3
            self._answer = tf.placeholder(tf.float32, [self.batch_size, num_classes], name='answer')

            with tf.variable_scope('mlp'):
                num_hiddens = config['model']['MLP']['num_hiddens']
                l1 = mlp.fully_connected(emb, num_hiddens, activation='relu', scope='l1')

                self._pred = mlp.fully_connected(l1, num_classes, activation='softmax', scope='softmax')
                self._best_pred = tf.argmax(self._pred, axis=1)

            self.loss = tf.reduce_mean(mlp.cross_entropy(self._pred, self._answer))
            self.error = tf.reduce_mean(mlp.error(self._pred, self._answer))


            print('Model... Oracle build!')

    def get_outputs(self):
        return [self.loss, self.error]


