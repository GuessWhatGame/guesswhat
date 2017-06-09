import tensorflow as tf

from generic.tf_utils.abstract_network import AbstractNetwork


from generic.tf_models import rnn, utils


class GuesserNetwork(AbstractNetwork):
    def __init__(self, config, num_words, device='', reuse=False):
        AbstractNetwork.__init__(self, "guesser", device=device)

        mini_batch_size = None

        with tf.variable_scope(self.scope_name, reuse=reuse):

            # Dialogues
            self.dialogues = tf.placeholder(tf.int32, [mini_batch_size, None], name='dialogues')
            self.seq_length = tf.placeholder(tf.int32, [mini_batch_size], name='seq_length')

            # Objects
            self.obj_mask = tf.placeholder(tf.float32, [mini_batch_size, None], name='obj_mask')
            self.obj_cats = tf.placeholder(tf.int32, [mini_batch_size, None], name='obj_cats')
            self.obj_spats = tf.placeholder(tf.float32, [mini_batch_size, None, config['spat_dim']], name='obj_spats')

            # Targets
            self.targets = tf.placeholder(tf.int32, [mini_batch_size], name="targets_index")



            self.object_cats_emb = utils.get_embedding(
                self.obj_cats,
                config['no_categories'] + 1,
                config['cat_emb_dim'],
                scope='cat_embedding')

            self.objects_input = tf.concat([self.object_cats_emb, self.obj_spats], axis=2)
            self.flat_objects_inp = tf.reshape(self.objects_input, [-1, config['cat_emb_dim'] + config['spat_dim']])

            with tf.variable_scope('obj_mlp'):
                h1 = utils.fully_connected(
                    self.flat_objects_inp,
                    n_out=config['obj_mlp_units'],
                    activation='relu',
                    scope='l1')
                h2 = utils.fully_connected(
                    h1,
                    n_out=config['dialog_emb_dim'],
                    activation='relu',
                    scope='l2')

            obj_embs = tf.reshape(h2, [-1, tf.shape(self.obj_cats)[1], config['dialog_emb_dim']])

            # Compute the word embedding
            input_words = utils.get_embedding(self.dialogues,
                                              n_words=num_words,
                                              n_dim=config['word_emb_dim'],
                                              scope="input_word_embedding")

            last_states, _ = rnn.variable_length_LSTM(input_words,
                                               num_hidden=config['num_lstm_units'],
                                               seq_length=self.seq_length)

            last_states = tf.reshape(last_states, [-1, config['num_lstm_units'], 1])
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

            self.softmax = masked_softmax(scores, self.obj_mask)
            self.selected_object = tf.argmax(self.softmax, axis=1)

            self.loss = tf.reduce_mean(utils.cross_entropy(self.softmax, self.targets))
            self.error = tf.reduce_mean(utils.error(self.softmax, self.targets))

    def get_outputs(self):
        return [self.loss, self.error]





    # def find_object(self, sess, dialogue, seq_length, ground_data):
    #     """Inputs:
    #     High level method that return whether the guesser manage to find the correct object according a dialogue and game information
    #
    #     Example
    #     --------
    #     {'question': [[1, 500, 3, 5, 2], [1, 48, 12, 2, 4]],
    #      'seq_length': [5, 4] length of the sequence (=first padding token(4) or array shape)
    #      'ground_data', {
    #         obj_mask : [[1,1,1,1], [1,1,,1,0]], #4 object / 3 objects
    #         obj_spats : np.array((2,8)), # spatial info by object
    #         obj_spats : [[10,22,11,10], [5,10,10]], # obj cat Ex [person,dog,cat,person],[kite,person,person]
    #         object_indices : [3,0] # indices for correct object, e.g. person:1 / kite:0
    #         },
    #      'spatial': [[-0.5, 0.8, 0.7, 0.5, 0.4, 0.56, -0.3, -0.1]],
    #      'seq_length': [5]}
    #     """
    #
    #     # TO avoid code duplication, we can:
    #     # - create a predict method
    #     # guesser_input = dict(ground_data) # shallow copy
    #     # guesser_input["question"] = dialogue
    #     # guesser_input["seq_length"] = seq_length
    #     # selected_object = self.predict(sess, guesser_input) # return predicted_object (or softmax)
    #     # found = (selected_object == ground_data["targets"])
    #     # OR found = (np.argmax(selected_object, axis=1) == ground_data["targets"]) if softmax
    #
    #     selected_object, softmax = sess.run([self.selected_object, self.softmax], feed_dict={
    #         self.dialogues: dialogue,
    #         self.seq_length: seq_length,
    #         self.mask: ground_data["obj_mask"],
    #         self.obj_spats: ground_data["obj_spats"],
    #         self.obj_cats: ground_data["obj_cats"],
    #     })
    #
    #     found = (selected_object == ground_data["targets"])
    #
    #     return found, softmax
    #
    #
