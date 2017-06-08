import tensorflow as tf
from  generic.tf_utils.abstract_network import AbstractNetwork

from generic.tf_models import rnn, utils, attention



class QGenNetworkLSTM(AbstractNetwork):

    #TODO: add dropout
    def __init__(self, config, num_words, policy_gradient, device='', reuse=False):
        AbstractNetwork.__init__(self, "qgen", device=device)

        # Create the scope for this graph
        with tf.variable_scope(self.scope_name, reuse=reuse):

            mini_batch_size = None

            # Picture
            self.image = tf.placeholder(tf.float32, [mini_batch_size] + config['image']["dim"], name='images')

            # Question
            self.dialogues = tf.placeholder(tf.int32, [mini_batch_size, None], name='dialogues')
            self.answer_mask = tf.placeholder(tf.float32, [mini_batch_size, None], name='answer_mask')  # 1 if keep and (1 q/a 1) for (START q/a STOP)
            self.padding_mask = tf.placeholder(tf.float32, [mini_batch_size, None], name='padding_mask')
            self.seq_length = tf.placeholder(tf.int32, [mini_batch_size], name='seq_length')

            # Rewards
            self.cum_rewards = tf.placeholder(tf.float32, shape=[mini_batch_size, None], name='cum_reward')

            # DECODER Hidden state (for beam search)
            zero_state = tf.zeros([1, config['num_lstm_units']])  # default LSTM state is a zero-vector
            zero_state = tf.tile(zero_state, [tf.shape(self.image)[0], 1])  # trick to do a dynamic size 0 tensors

            self.decoder_zero_state_c = tf.placeholder_with_default(zero_state, [mini_batch_size, config['num_lstm_units']])
            self.decoder_zero_state_h = tf.placeholder_with_default(zero_state, [mini_batch_size, config['num_lstm_units']])
            decoder_initial_state = tf.contrib.rnn.LSTMStateTuple(c=self.decoder_zero_state_c, h=self.decoder_zero_state_h)

            # Misc
            self.is_training = tf.placeholder(tf.bool, name='is_training')

            # remove last token
            input_dialogues = self.dialogues[:, :-1]
            input_seq_length = self.seq_length - 1

            # remove first token(=start token)
            rewards = self.cum_rewards[:, 1:]
            target_words = self.dialogues[:, 1:]

            # to understand the padding:
            # input
            #   <start>  is   it   a    blue   <?>   <yes>   is   it  a    car  <?>   <no>   <stop_dialogue>
            # target
            #    is      it   a   blue   <?>    -      is    it   a   car  <?>   -   <stop_dialogue>  -



            # image processing
            if len(config["image"]["dim"]) == 1:
                self.image_out = self.image
            else:
                self.image_out = attention.attention_factory(self.image, None, "none") #TODO: improve by using the previous lstm state?


            # Reduce the embedding size of the image
            with tf.variable_scope('picture_embedding'):
                picture_emb = utils.fully_connected(self.image_out,
                                                    config['picture_embedding_size'])
                picture_emb = tf.expand_dims(picture_emb, 1)
                picture_emb = tf.tile(picture_emb, [1, tf.shape(input_dialogues)[1], 1])

            # Compute the question embedding
            input_words = utils.get_embedding(
                input_dialogues,
                n_words=num_words,
                n_dim=config['word_embedding_size'],
                scope="word_embedding")

            # concat word embedding and picture embedding
            decoder_input = tf.concat([input_words, picture_emb], axis=2, name="concat_full_embedding")


            # encode one word+picture
            decoder_lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                    config['num_lstm_units'],
                    layer_norm=False,
                    dropout_keep_prob=1.0,
                    reuse=reuse)


            self.decoder_output, self.decoder_state = tf.nn.dynamic_rnn(
                cell=decoder_lstm_cell,
                inputs=decoder_input,
                dtype=tf.float32,
                initial_state=decoder_initial_state,
                sequence_length=input_seq_length,
                scope="word_decoder")  # TODO: use multi-layer RNN

            max_sequence = tf.reduce_max(self.seq_length)

            # compute the softmax for evaluation
            with tf.variable_scope('decoder_output'):
                flat_decoder_output = tf.reshape(self.decoder_output, [-1, decoder_lstm_cell.output_size])
                flat_mlp_output = utils.fully_connected(flat_decoder_output, num_words)

                # retrieve the batch/dialogue format
                mlp_output = tf.reshape(flat_mlp_output, [tf.shape(self.seq_length)[0], max_sequence - 1, num_words])  # Ignore th STOP token

                self.softmax_output = tf.nn.softmax(mlp_output, name="softmax")
                self.argmax_output = tf.argmax(mlp_output, axis=2)

                self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mlp_output, labels=target_words)


            # Compute policy gradient
            if policy_gradient:

                with tf.variable_scope('rl_baseline'):
                    decoder_out = tf.stop_gradient(self.decoder_output)  # take the LSTM output (and stop the gradient!)

                    flat_decoder_output = tf.reshape(decoder_out, [-1, decoder_lstm_cell.output_size])  #
                    flat_h1 = utils.fully_connected(flat_decoder_output, n_out=config["baseline_no_hidden"], activation='relu', scope='baseline_hidden')
                    flat_baseline = utils.fully_connected(flat_h1, 1, activation='relu', scope='baseline_out')

                    self.baseline = tf.reshape(flat_baseline, [tf.shape(self.seq_length)[0], max_sequence-1])
                    self.baseline *= self.answer_mask[:, 1:]
                    self.baseline *= self.padding_mask[:, 1:]


                with tf.variable_scope('policy_gradient_loss'):

                    # Compute log_prob
                    self.log_of_policy = tf.identity(self.cross_entropy_loss)
                    self.log_of_policy *= self.answer_mask[:, 1:]  # remove answers (<=> predicted answer has maximum reward) (ignore the START token in the mask)
                    # No need to use padding mask as the discounted_reward is already zero once the episode terminated

                    # Policy gradient loss
                    rewards *= self.answer_mask[:, 1:]
                    self.score_function = tf.multiply(self.log_of_policy, rewards - self.baseline)  # score function

                    self.baseline_loss = tf.reduce_sum(tf.square(rewards - self.baseline))

                    self.policy_gradient_loss = tf.reduce_sum(self.score_function, axis=1)  # sum over the dialogue trajectory
                    self.policy_gradient_loss = tf.reduce_mean(self.policy_gradient_loss, axis=0)  # reduce over minibatch dimension

                    self.loss = self.policy_gradient_loss

            else:
                # compute the maximum likelihood loss
                with tf.variable_scope('ml_loss'):

                    ml_loss = tf.identity(self.cross_entropy_loss)
                    ml_loss *= self.answer_mask[:, 1:]  # remove answers (ignore the <stop> token)
                    ml_loss *= self.padding_mask[:, 1:]  # remove padding (ignore the <start> token)

                    # Count number of unmask elements
                    count = tf.reduce_sum(self.padding_mask) - tf.reduce_sum(1 - self.answer_mask[:, :-1]) - 1  # no_unpad - no_qa - START token

                    ml_loss = tf.reduce_sum(ml_loss, axis=1)  # reduce over dialogue dimension
                    ml_loss = tf.reduce_sum(ml_loss, axis=0)  # reduce over minibatch dimension
                    self.ml_loss = ml_loss / count  # Normalize

                    self.loss = self.ml_loss

            # Final optimization !
            # def optimize(loss, variables, config, optimizer):
            #     clip_val = config['optimizer']['clip_val']
            #     gvs = optimizer.compute_gradients(loss, var_list=variables)
            #     clipped_gvs = [(tf.clip_by_norm(grad, clip_val), var) for grad, var in gvs]
            #
            #     return optimizer.apply_gradients(clipped_gvs)

            # lrt = config['optimizer']['learning_rate']
            # with tf.variable_scope('ml_optimizer'):
            #     variables = [v for v in tf.trainable_variables() if 'rl_baseline' not in v.name]
            #     self.ml_optimize = optimize(self.ml_loss, variables, config, tf.train.AdamOptimizer(learning_rate=lrt))

            # # We directly minimize the approximate score function and we let Tensorflow compute the gradient
            # with tf.variable_scope('policy_gradient_optimizer'):
            #     pg_variables = [v for v in tf.trainable_variables() if 'rl_baseline' not in v.name]
            #     baseline_variables = [v for v in tf.trainable_variables() if 'rl_baseline' in v.name]
            #     self.pg_optimize = optimize(self.policy_gradient_loss, pg_variables, config, tf.train.GradientDescentOptimizer(learning_rate=lrt))
            #     self.baseline_optimize = optimize(self.baseline_loss, baseline_variables, config, tf.train.GradientDescentOptimizer(learning_rate=1e-3))

    def get_outputs(self):
        return [self.loss]



