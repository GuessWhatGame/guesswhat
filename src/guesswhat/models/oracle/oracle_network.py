import tensorflow as tf

from generic.tf_models import rnn, utils, attention

from generic.tf_utils.abstract_network import AbstractNetwork


class OracleNetwork(AbstractNetwork):

    def __init__(self, config, num_words, device='', reuse=False):
        AbstractNetwork.__init__(self, "oracle", device=device)

        with tf.variable_scope(self.scope_name, reuse=reuse):
            embeddings = []
            self.batch_size = None

            # QUESTION
            self._is_training = tf.placeholder(tf.bool, name="is_training")
            self._question = tf.placeholder(tf.int32, [self.batch_size, None], name='question')
            self._seq_length = tf.placeholder(tf.int32, [self.batch_size], name='seq_length')

            word_emb = utils.get_embedding(self._question,
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

                cat_emb = utils.get_embedding(self._category,
                                              int(config['model']['category']["n_categories"]) + 1,  # we add the unkwon category
                                              int(config['model']['category']["embedding_dim"]),
                                              scope="cat_embedding")
                embeddings.append(cat_emb)
                print("Input: Category")

            # SPATIAL
            if config['inputs']['spatial']:
                self._spatial = tf.placeholder(tf.float32, [self.batch_size, 8], name='spatial')
                embeddings.append(self._spatial)
                print("Input: Spatial")


            # IMAGE
            if config['inputs']['image']:
                self._image = tf.placeholder(tf.float32, [self.batch_size] + config['model']['image']["dim"], name='image')

                if len(config['model']["image"]["dim"]) == 1:
                    self.image_out = self._image
                else:
                    self.image_out = attention.attention_factory(self._image, lstm_states, config['model']["image"]["attention"])

                embeddings.append(self.image_out)
                print("Input: Image")


            # CROP
            if config['inputs']['crop']:
                self._crop = tf.placeholder(tf.float32, [self.batch_size] + config['model']['crop']["dim"], name='crop')

                if len(config['model']["crop"]["dim"]) == 1:
                    self.crop_out = self._crop
                else:
                    self.crop_out = attention.attention_factory(self._crop, lstm_states, config['model']['crop']["attention"])

                embeddings.append(self.crop_out)
                print("Input: Crop")


            # Compute the final embedding
            emb = tf.concat(embeddings, axis=1)

            # OUTPUT
            num_classes = 3
            self._answer = tf.placeholder(tf.float32, [self.batch_size, num_classes], name='answer')

            with tf.variable_scope('mlp'):
                num_hiddens = config['model']['MLP']['num_hiddens']
                l1 = utils.fully_connected(emb, num_hiddens, activation='relu', scope='l1')

                self.pred = utils.fully_connected(l1, num_classes, activation='softmax', scope='softmax')
                self.best_pred = tf.argmax(self.pred, axis=1)

            self.loss = tf.reduce_mean(utils.cross_entropy(self.pred, self._answer))
            self.error = tf.reduce_mean(utils.error(self.pred, self._answer))


            print('Model... Oracle build!')

    def get_outputs(self):
        return [self.loss, self.error]


