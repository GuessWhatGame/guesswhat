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
            self.images = tf.placeholder(tf.float32, [mini_batch_size] + config['image']["dim"], name='images')

            # Question
            self.dialogues = tf.placeholder(tf.int32, [mini_batch_size, None], name='dialogues')
            self.answer_mask = tf.placeholder(tf.float32, [mini_batch_size, None], name='answer_mask')  # 1 if keep and (1 q/a 1) for (START q/a STOP)
            self.padding_mask = tf.placeholder(tf.float32, [mini_batch_size, None], name='padding_mask')
            self.seq_length = tf.placeholder(tf.int32, [mini_batch_size], name='seq_length')

            # Rewards
            self.cum_rewards = tf.placeholder(tf.float32, shape=[mini_batch_size, None], name='cum_reward')

            # DECODER Hidden state (for beam search)
            zero_state = tf.zeros([1, config['num_lstm_units']])  # default LSTM state is a zero-vector
            zero_state = tf.tile(zero_state, [tf.shape(self.images)[0], 1])  # trick to do a dynamic size 0 tensors

            self.decoder_zero_state_c = tf.placeholder_with_default(zero_state, [mini_batch_size, config['num_lstm_units']], name="state_c")
            self.decoder_zero_state_h = tf.placeholder_with_default(zero_state, [mini_batch_size, config['num_lstm_units']], name="state_h")
            decoder_initial_state = tf.contrib.rnn.LSTMStateTuple(c=self.decoder_zero_state_c, h=self.decoder_zero_state_h)

            # Misc
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.greedy = tf.placeholder_with_default(False, shape=(), name="greedy") # use for graph
            self.samples = None

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
                self.image_out = self.images
            else:
                self.image_out = attention.attention_factory(self.images, None, "none") #TODO: improve by using the previous lstm state?


            # Reduce the embedding size of the image
            with tf.variable_scope('picture_embedding'):
                self.picture_emb = utils.fully_connected(self.image_out,
                                                    config['picture_embedding_size'])
                picture_emb = tf.expand_dims(self.picture_emb, 1)
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

            # Compute policy gradient
            if policy_gradient:

                with tf.variable_scope('rl_baseline'):
                    decoder_out = tf.stop_gradient(self.decoder_output)  # take the LSTM output (and stop the gradient!)

                    flat_decoder_output = tf.reshape(decoder_out, [-1, decoder_lstm_cell.output_size])  #
                    flat_h1 = utils.fully_connected(flat_decoder_output, n_out=100, activation='relu', scope='baseline_hidden')
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






    def get_outputs(self):
        return [self.loss]

    def build_sampling_graph(self, config, tokenizer, max_length=12):

        if self.samples is not None:
            return

        # define stopping conditions
        def stop_cond(states_c, states_h, tokens, seq_length, stop_indicator):

            has_unfinished_dialogue = tf.less(tf.shape(tf.where(stop_indicator))[0],tf.shape(stop_indicator)[0]) # TODO use "any" instead of checking shape
            has_not_reach_size_limit = tf.less(tf.reduce_max(seq_length), max_length)

            return tf.logical_and(has_unfinished_dialogue,has_not_reach_size_limit)


        # define one_step sampling
        with tf.variable_scope(self.scope_name):
            stop_token = tf.constant(tokenizer.stop_token)
            stop_dialogue_token = tf.constant(tokenizer.stop_dialogue)

        def step(prev_state_c, prev_state_h, tokens, seq_length, stop_indicator):
            input = tf.gather(tokens, tf.shape(tokens)[0] - 1)

            # Look for new finish dialogue
            is_stop_token = tf.equal(input, stop_token)
            is_stop_dialogue_token = tf.equal(input, stop_dialogue_token)
            is_stop = tf.logical_or(is_stop_token, is_stop_dialogue_token)
            stop_indicator = tf.logical_or(stop_indicator, is_stop)  # flag to false new finished dialogue

            # increment seq_length when the dialogue is not over
            seq_length = tf.where(stop_indicator, seq_length, tf.add(seq_length, 1))

            # compute the next words. TODO: factorize with qgen.. but how?!
            with tf.variable_scope(self.scope_name, reuse=True):
                word_emb = utils.get_embedding(
                    input,
                    n_words=tokenizer.no_words,
                    n_dim=config['word_embedding_size'],
                    scope="word_embedding",
                    reuse=True)

                inp_emb = tf.concat([word_emb, self.picture_emb], axis=1)
                with tf.variable_scope("word_decoder"):
                    lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                        config['num_lstm_units'],
                        layer_norm=False,
                        dropout_keep_prob=1.0,
                        reuse=True)

                    state = tf.contrib.rnn.LSTMStateTuple(c=prev_state_c, h=prev_state_h)
                    out, state = lstm_cell(inp_emb, state)

                    # store/update the state when the dialogue is not finished (after sampling the <?> token)
                    cond = tf.greater_equal(seq_length, tf.subtract(tf.reduce_max(seq_length), 1))
                    state_c = tf.where(cond, state.c, prev_state_c)
                    state_h = tf.where(cond, state.h, prev_state_h)


                with tf.variable_scope('decoder_output'):
                    output = utils.fully_connected(state_h, tokenizer.no_words, reuse=True)

                    sampled_tokens = tf.cond(self.greedy,
                                             lambda: tf.argmax(output, 1),
                                             lambda: tf.reshape(tf.multinomial(output, 1), [-1])
                                             )
                    sampled_tokens = tf.cast(sampled_tokens, tf.int32)

            tokens = tf.concat([tokens, tf.expand_dims(sampled_tokens, 0)], axis=0) # check axis!

            return state_c, state_h, tokens, seq_length, stop_indicator


        # initialialize sequences
        batch_size = tf.shape(self.seq_length)[0]
        seq_length = tf.fill([batch_size], 0)
        stop_indicator = tf.fill([batch_size], False)

        transpose_dialogue = tf.transpose(self.dialogues, perm=[1,0])

        self.samples = tf.while_loop(stop_cond, step, [self.decoder_zero_state_c,
                                                       self.decoder_zero_state_h,
                                                       transpose_dialogue,
                                                       seq_length,
                                                       stop_indicator],
                                     shape_invariants=[self.decoder_zero_state_c.get_shape(),
                                                       self.decoder_zero_state_h.get_shape(),
                                                       tf.TensorShape([None, None]),
                                                       seq_length.get_shape(),
                                                       stop_indicator.get_shape()])

