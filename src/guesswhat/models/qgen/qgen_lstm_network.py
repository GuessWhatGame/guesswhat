import collections
import copy
import json
import numpy as np
import os
import tensorflow as tf


from guesswhat.models.utils import get_embedding, fully_connected
from tqdm import *

#TODO add seq_length
BeamToken = collections.namedtuple('BeamToken', ['path', 'word_id', 'decoder_state',
                                   'score', 'prev_beam'])
# The Beam token is the key element of question generation
#    - path are the outputed words
#    - word_id are the next word inputs
#    - decoder_state is the state of the decoder after outputed path[-1]
#    - score is the \sum log(prob(w)) (beam search only)
#    - prev_beam chain the previous beam (to store the hidden state for each outputed words)




# Warning modify the input (use defensive copy?)
def clear_after_stop_dialogue(dialogues, tokenizer):
    no_questions = []
    answer_indices = []
    stop_indices = []
    for i, dialogue in enumerate(dialogues):
        stop_dialogue_index = get_index(dialogue.tolist(), tokenizer.stop_dialogue, default=len(dialogue)-1)
        answers_index = [j for j,token in enumerate(dialogue[:stop_dialogue_index]) if token in tokenizer.answers]
        if answers_index:
            no_questions.append(len(answers_index))
            answer_indices.append(answers_index)
            dialogues[i] = dialogue[:stop_dialogue_index+1]
            stop_indices.append(stop_dialogue_index)
        else:
            dialogues[i] = []
            no_questions.append(0)
            answer_indices.append([])
            stop_indices.append(0)


    return dialogues, no_questions, answer_indices, stop_indices


def list_to_padded_tokens(dialogues, tokenizer):

    # compute the length of the dialogue
    seq_length = [len(d) for d in dialogues]

    # Get dialogue numpy max size
    batch_size = len(dialogues)
    max_seq_length = max(seq_length)

    # Initialize numpy array (Idea re-use previous numpy array to optimize memory consumption)
    padded_tokens = np.full((batch_size, max_seq_length), tokenizer.padding_token, dtype=np.int32)

    # fill the padded array with word_id
    for i, (one_path, l) in enumerate(zip(dialogues, seq_length)):
       padded_tokens[i, 0:l] = one_path

    return padded_tokens, seq_length


# TODO move to index
def get_index(l, index, default=-1):
    try:
        return l.index(index)
    except ValueError:
        return default


def unloop_beam_serie(cur_beam):

    # Build the full beam sequence by using the chain-list structure
    sequence = [cur_beam]
    while cur_beam.prev_beam is not None:
        cur_beam = cur_beam.prev_beam
        sequence.append(cur_beam)

    return sequence[::-1]  # reverse sequence


class QGenNetworkLSTM(object):
    # TODO use regularization (dropout?)
    def __init__(self, config, use_baseline):

        self.scope_name = "qgen"
        self.config = config

        # Create the scope for this graph
        with tf.variable_scope(self.scope_name):

            #self.is_training = tf.placeholder(tf.bool, name='is_training')
            mini_batch_size = None
            no_words = config['model']['no_words']

            # PICTURE
            self.picture_fc8 = tf.placeholder(tf.float32,  [mini_batch_size, config['model']['fc8_dim']], name='picture_fc8')

            # QUESTION
            self.dialogues = tf.placeholder(tf.int32, [mini_batch_size, None], name='question')
            self.answer_mask = tf.placeholder(tf.float32, [mini_batch_size, None], name='answer_mask')  # 1 if keep and (1 q/a 1) for (START q/a STOP)
            self.padding_mask = tf.placeholder(tf.float32, [mini_batch_size, None], name='padding_mask')
            self.seq_length = tf.placeholder(tf.int32, [mini_batch_size], name='seq_length')

            # REWARDS
            self.cum_rewards = tf.placeholder(tf.float32, shape=[mini_batch_size, None], name='cum_reward')
            zero_baseline = tf.zeros_like(self.cum_rewards, dtype=tf.float32)
            self.baselines =  tf.placeholder_with_default(zero_baseline, shape=[mini_batch_size, None], name='baselines')

            # DECODER Hidden state (for beam search)
            zero_state = tf.zeros([1, config['model']['num_lstm_units']])  # default LSTM state is a zero-vector
            zero_state = tf.tile(zero_state, [tf.shape(self.picture_fc8)[0], 1])  # trick to do a dynamic size 0 tensors

            self.decoder_zero_state_c = tf.placeholder_with_default(zero_state, [mini_batch_size, config['model']['num_lstm_units']])
            self.decoder_zero_state_h = tf.placeholder_with_default(zero_state, [mini_batch_size, config['model']['num_lstm_units']])
            decoder_initial_state = tf.nn.rnn_cell.LSTMStateTuple(c=self.decoder_zero_state_c, h=self.decoder_zero_state_h)

            # remove last token
            input_dialogues = self.dialogues[:, :-1]
            input_seq_length = self.seq_length - 1

            # remove first token(=start token)
            rewards = self.cum_rewards[:, 1:]
            baselines = self.baselines[:, 1:]
            target_words = self.dialogues[:, 1:]

            # Reduce the embedding size of the image
            with tf.variable_scope('picture_embedding'):
                picture_emb = fully_connected(self.picture_fc8,
                                              config['model']['picture_embedding_size'])
                picture_emb = tf.expand_dims(picture_emb, 1)
                picture_emb = tf.tile(picture_emb, [1, tf.shape(input_dialogues)[1], 1])

            # Compute the question embedding
            input_words = get_embedding(
                input_dialogues,
                n_words=no_words,
                n_dim=config['model']['word_embedding_size'],
                scope="word_embedding")

            # concat word embedding and picture embedding
            decoder_input = tf.concat(2, [input_words, picture_emb], name="concat_full_embedding")
            # full_encoder_input = input_words

            # encode one word+picture
            decoder_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                num_units=config['model']['num_lstm_units'],
                forget_bias=1.0,
                state_is_tuple=True)


            self.decoder_output, self.decoder_state = tf.nn.dynamic_rnn(
                cell=decoder_lstm_cell,
                inputs=decoder_input,
                dtype=tf.float32,
                initial_state=decoder_initial_state,
                sequence_length=input_seq_length,
                scope="word_decoder")

            max_sequence = tf.reduce_max(self.seq_length)

            # compute the softmax for evaluation
            with tf.variable_scope('decoder_output'):
                flat_decoder_output = tf.reshape(self.decoder_output, [-1, decoder_lstm_cell.output_size])
                flat_mlp_output = fully_connected(flat_decoder_output, no_words)

                # retrieve the batch/dialogue format
                mlp_output = tf.reshape(flat_mlp_output, [tf.shape(self.seq_length)[0], max_sequence-1, no_words])  # Ignore th STOP token

                self.softmax_output = tf.nn.softmax(mlp_output, name="softmax")
                self.argmax_output = tf.argmax(self.softmax_output, axis=2)

                self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(mlp_output, target_words)

            # compute the maximum likelihood loss
            with tf.variable_scope('ml_loss'):
                ml_loss = tf.identity(self.cross_entropy_loss)
                ml_loss *= self.answer_mask[:, 1:]  # remove answers (ignore the STOP token)
                ml_loss *= self.padding_mask[:, 1:]  # remove padding (ignore the START token)

                # Count number of unmask elements
                count = tf.reduce_sum(self.padding_mask) - tf.reduce_sum(1 - self.answer_mask[:, :-1]) - 1  # no_unpad - no_qa - START token

                ml_loss = tf.reduce_sum(ml_loss, axis=1)  # reduce over dialogue dimension
                ml_loss = tf.reduce_sum(ml_loss, axis=0)  # reduce over minibatch dimension
                self.ml_loss = ml_loss / count  # Normalize

            # Compute policy gradient

            if use_baseline:
                with tf.variable_scope('rl_baseline'):
                    decoder_out = self.decoder_output
                    flat_decoder_output = tf.stop_gradient(tf.reshape(decoder_out, [-1, decoder_lstm_cell.output_size]))
                    flat_h1 = fully_connected(flat_decoder_output, 100, activation='relu', scope='fc1')
                    flat_baseline = fully_connected(flat_h1, 1, scope='fc2')
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



            # Final optimization !
            def optimize(loss, variables, config, optimizer):
                clip_val = config['optimizer']['clip_val']
                gvs = optimizer.compute_gradients(loss, var_list=variables)
                clipped_gvs = [(tf.clip_by_norm(grad, clip_val), var) for grad, var in gvs]

                return optimizer.apply_gradients(clipped_gvs)

            lrt = config['optimizer']['learning_rate']
            with tf.variable_scope('ml_optimizer'):
                variables = [v for v in tf.trainable_variables() if 'rl_baseline' not in v.name]
                self.ml_optimize = optimize(self.ml_loss, variables, config, tf.train.AdamOptimizer(learning_rate=lrt))

            # We directly minimize the approximate score function and we let Tensorflow compute the gradient
            with tf.variable_scope('policy_gradient_optimizer'):
                pg_variables = [v for v in tf.trainable_variables() if 'rl_baseline' not in v.name]
                baseline_variables = [v for v in tf.trainable_variables() if 'rl_baseline' in v.name]
                self.pg_optimize = optimize(self.policy_gradient_loss, pg_variables, config, tf.train.GradientDescentOptimizer(learning_rate=lrt))
                self.baseline_optimize = optimize(self.baseline_loss, baseline_variables, config, tf.train.GradientDescentOptimizer(learning_rate=1e-3))

    @classmethod
    def from_exp_identifier(cls, identifier, exp_dir):
        file = os.path.join(exp_dir, 'experiments.jsonl')
        with open(file, 'rb') as f:
            config = None
            for line in f:
                s = json.loads(line.decode('utf-8'))
                if s['identifier'] == identifier:
                    config = s['config']
        assert(config is not None), "Couldn't find QGen config"
        return cls(config)

    def create_initial_beam(self, batch, tokenizer, decoder_state=None, batch_size=1):

        if "padded_tokens" in batch:
            batch_size = batch["padded_tokens"].shape[0]
            initial_words_id = batch["padded_tokens"]
        else:
            initial_words_id = [[tokenizer.start_token]] * batch_size

        if decoder_state is None:
            decoder_state = tf.nn.rnn_cell.LSTMStateTuple(
                c=np.zeros((batch_size, int(self.decoder_zero_state_c.get_shape()[1]))),
                h=np.zeros((batch_size, int(self.decoder_zero_state_h.get_shape()[1]))))

        return BeamToken(
            path=[[] for _ in range(batch_size)],
            word_id=initial_words_id,
            decoder_state=decoder_state,
            score=0,  # initial probability is 1. If we apply the log trick log(1) = 0
            prev_beam=None
        )

    def evaluate_ml(self, sess, iterator):
        loss = 0
        N = 0
        for N, batch in enumerate(tqdm(iterator)):
            l, _ = sess.run([self.ml_loss, self.ml_optimize],
                            feed_dict={
                                self.picture_fc8: batch['picture_fc8'],
                                self.dialogues: batch['padded_tokens'],
                                self.answer_mask: batch['answer_mask'],
                                self.padding_mask: batch['padding_mask'],
                                self.seq_length: batch['seq_length']
                            })
            loss += l
        loss /= (N + 1)

        return loss

    def build_sample_graph(self, max_length=12, greedy=False):
        batch_size = tf.shape(self.picture_fc8)[0]
        self.stop_token = tf.placeholder(tf.int64, shape=())
        self.stop_dialogue_token = tf.placeholder(tf.int64, shape=())
        self.start_tokens = tf.placeholder(tf.int64, shape=[None])

        self.state_c = tf.placeholder(tf.float32, [None, self.config['model']['num_lstm_units']])
        self.state_h = tf.placeholder(tf.float32, [None, self.config['model']['num_lstm_units']])

        tokens = tf.expand_dims(self.start_tokens, 0)
        seq_length = tf.fill([batch_size], 0)

        with tf.variable_scope("qgen", reuse=True):
            with tf.variable_scope("picture_embedding"):
                picture_emb = fully_connected(self.picture_fc8,
                                              self.config['model']['picture_embedding_size'],
                                              reuse=True)

        stop_indicator = tf.fill([batch_size], False)
        def stop_cond(states_c, states_h, tokens, seq_length, stop_indicator):
            return tf.logical_and(tf.less(tf.shape(tf.where(stop_indicator))[0], tf.shape(stop_indicator)[0]), tf.less(tf.reduce_max(seq_length), max_length))

        def step(state_c, state_h, tokens, seq_length, stop_indicator):
            input = tf.gather(tokens, tf.shape(tokens)[0] - 1)

            is_stop_token = tf.equal(input, self.stop_token)
            is_stop_dialogue_token = tf.equal(input, self.stop_dialogue_token)
            is_stop = tf.logical_or(is_stop_token, is_stop_dialogue_token)
            stop_indicator = tf.logical_or(stop_indicator, is_stop)

            seq_length = tf.select(stop_indicator, seq_length, tf.add(seq_length, 1))

            with tf.variable_scope("qgen", reuse=True):
                word_emb = get_embedding(
                    input,
                    n_words=self.config['model']['no_words'],
                    n_dim=self.config['model']['word_embedding_size'],
                    scope="word_embedding",
                    reuse=True)

                inp_emb = tf.concat(1, [word_emb, picture_emb])
                with tf.variable_scope("word_decoder"):
                    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                            num_units=self.config['model']['num_lstm_units'],
                            forget_bias=1.0,
                            state_is_tuple=True)
                    state = tf.nn.rnn_cell.LSTMStateTuple(c=state_c, h=state_h)
                    out, state = lstm_cell(inp_emb, state)
                    cond = tf.greater_equal(seq_length, tf.subtract(tf.reduce_max(seq_length), 1))
                    state_c = tf.select(cond, state.c, state_c)
                    state_h = tf.select(cond, state.h, state_h)

                with tf.variable_scope('decoder_output'):
                    output = fully_connected(state_h, self.config['model']['no_words'], reuse=True)
                    if greedy:
                        sampled_tokens = tf.argmax(output, 1)
                    else:
                        sampled_tokens = tf.reshape(tf.multinomial(output, 1), [-1])

            tokens = tf.concat(0, [tokens, tf.expand_dims(sampled_tokens, 0)])

            return state_c, state_h, tokens, seq_length, stop_indicator

        self.samples = tf.while_loop(stop_cond, step, [self.state_c,
                                                       self.state_h,
                                                       tokens,
                                                       seq_length,
                                                       stop_indicator],
                                     shape_invariants=[self.state_c.get_shape(),
                                                       self.state_h.get_shape(),
                                                       tf.TensorShape([None, None]),
                                                       seq_length.get_shape(),
                                                       stop_indicator.get_shape()])

    def sample(self, sess, picture_fc8, start_tokens, tokenizer, state_c=None, state_h=None):
        if state_c is None or state_h is None:
            tmp = sess.run(self.samples, feed_dict={
                self.picture_fc8: picture_fc8,
                self.start_tokens: start_tokens,
                self.stop_token: tokenizer.stop_token,
                self.stop_dialogue_token: tokenizer.stop_dialogue
            })
        else:
            tmp = sess.run(self.samples, feed_dict={
                self.picture_fc8: picture_fc8,
                self.start_tokens: start_tokens,
                self.stop_token: tokenizer.stop_token,
                self.stop_dialogue_token: tokenizer.stop_dialogue,
                self.state_c: state_c,
                self.state_h: state_h
            })

        return {'state_c': tmp[0],
                'state_h': tmp[1],
                'tokens': tmp[2],
                'seq_length': tmp[3],
                'stop_indicator': tmp[4]}

    def generate_next_question(self, sess, batch, tokenizer, initial_beam=None, greedy=False, keep_trajectory=False, max_depth=7):

        batch_size = batch["picture_fc8"].shape[0]
        current_beam = None

        if initial_beam is None:
            initial_beam = self.create_initial_beam(batch, tokenizer=tokenizer, batch_size=batch_size)

        # Select the correct output and sampler to be greedy or not (memory optimization)
        if greedy:
            output_list = [self.argmax_output, self.decoder_state]
            sample_word_fct = lambda argmax: [[x] for x in argmax[:, -1]]
        else:
            output_list = [self.softmax_output, self.decoder_state]
            sample_word_fct = lambda softmax: [[np.random.choice(prob[-1].shape[0], 1, p=prob[-1])[0]] for prob in softmax]  # Sample according softmax output

        # Initialize the question generation loop
        rollout_sequence = initial_beam.path
        rollout_words = initial_beam.word_id

        rollout_decoder_state = initial_beam.decoder_state
        previous_rollout_decoder_state = rollout_decoder_state


        # Look for the padding token to know the sequence length
        rollout_seq_length = []
        for one_word_input in rollout_words:
            rollout_seq_length.append(get_index(list(one_word_input),
                                                index=tokenizer.padding_token,  # The first padding index equals the sequence_length
                                                default=len(one_word_input)+1))   # +1 for the dummy STOP token

        # Compute a softmax sampling starting with the trajectory token
        for depth in range(max_depth):

            # Append a dummy STOP token to compute a single step
            rollout_words = np.concatenate((rollout_words, [[tokenizer.stop_token]] * batch_size), axis=1)

            # Compute one forward step
            rollout_out, rollout_decoder_state = sess.run(output_list,
                                                          feed_dict={
                                                                  self.picture_fc8: batch["picture_fc8"],
                                                                  self.dialogues: rollout_words,
                                                                  self.seq_length: rollout_seq_length,  # [word_id, STOP]
                                                                  self.decoder_zero_state_c: rollout_decoder_state.c,
                                                                  self.decoder_zero_state_h: rollout_decoder_state.h
                                                                            })

            # Sample the next words according the softmax policy
            # From this point the sequence will always be [word_id, STOP]
            rollout_words = sample_word_fct(rollout_out)
            rollout_seq_length = [2] * batch_size  # [word_id, STOP]

            # Store the rollout (ignore tokens after the STOP token and keep the previous hidden state)
            for i, cur_word_id, sequence in zip(range(batch_size), rollout_words, rollout_sequence):

                if len(sequence) == 0 or sequence[-1] not in [tokenizer.stop_token, tokenizer.stop_dialogue, tokenizer.padding_token]:
                    rollout_sequence[i].append(cur_word_id[0])
                else:
                    rollout_sequence[i].append(tokenizer.padding_token)
                    rollout_words[i] = [tokenizer.stop_token]
                    rollout_decoder_state.c[i] = previous_rollout_decoder_state.c[i] #TODO the state structure should be abstracted
                    rollout_decoder_state.h[i] = previous_rollout_decoder_state.h[i]

            # Store trajectory
            current_beam = BeamToken(
                    path=copy.deepcopy(rollout_sequence),  # remove the reference to the list of the previous beam
                    word_id=copy.deepcopy(rollout_words),
                    decoder_state=rollout_decoder_state,
                    score=0,
                    prev_beam=current_beam if keep_trajectory else None  # Keep trace of the previous beam if we want to keep the trajectory
            )

            previous_rollout_decoder_state = rollout_decoder_state

            # Stop generating question if all sequence has a STOP token
            for sequence in rollout_sequence:
                if sequence[-1] not in [tokenizer.stop_token, tokenizer.stop_dialogue, tokenizer.padding_token]:
                    break
            else:
                break

        # Build the full beam sequence by using the chain-list structure
        final_sequence = unloop_beam_serie(current_beam)

        return final_sequence

    def eval_one_beam_search(self, sess, one_sample, tokenizer, initial_beam=None, max_depth=7, k_best=5, keep_trajectory=False):

        # Check that one_sample has a single element and there is no padding in word history

        if initial_beam is None:
            assert one_sample["picture_fc8"].shape[0] == 1
            assert "padded_tokens" not in one_sample or one_sample["padded_tokens"][0][-1] != tokenizer.padding_token

            # Create the first beam token that will generate the sequence
            initial_beam = self.create_initial_beam(one_sample, batch_size=1, tokenizer=tokenizer)

        to_evaluate = [initial_beam]


        memory = []
        for depth in range(max_depth):

            # evaluate all the current tokens
            for beam_token in to_evaluate:

                # if token is final token, directly put it into memory
                if beam_token.word_id[0][-1] in [tokenizer.stop_token, tokenizer.stop_dialogue]:
                    memory.append(beam_token)
                    continue

                # Append a dummy STOP token to fit HQGen constraint (can also be a PADDING)
                dialogue_history = np.concatenate((beam_token.word_id, [[tokenizer.stop_token]]), axis=1)

                # evaluate next_step
                softmax, decoder_state = sess.run([self.softmax_output, self.decoder_state],
                                                  feed_dict={
                                                      self.picture_fc8: one_sample["picture_fc8"],
                                                      self.dialogues: dialogue_history,
                                                      self.seq_length: [dialogue_history.shape[1]],
                                                      self.decoder_zero_state_c: beam_token.decoder_state.c,
                                                      self.decoder_zero_state_h: beam_token.decoder_state.h
                                                  })

                # softmax.shape = (batch_size, seq_length, output_size)
                # Reshape tensor (remove 1 size batch)
                softmax = softmax[0, -1]

                # put into memory the k-best tokens of this sample
                k_best_word_indices = np.argpartition(softmax, -k_best)[-k_best:]
                for word_id in k_best_word_indices:
                    memory.append(
                        BeamToken(
                            path=[beam_token.path[0] + [word_id]],
                            word_id=[[word_id]],
                            decoder_state=decoder_state,
                            score=beam_token.score + np.log(softmax[word_id]),  # log trick
                            prev_beam=beam_token if keep_trajectory else None  # Keep trace of the previous beam if we want to keep the trajectory
                        ))

            # retrieve best beams in memory
            scores = [beam.score / len(beam.path[0]) for beam in memory]
            k_best_word_indices = np.argpartition(scores, -k_best)[-k_best:]
            to_evaluate = [memory[i] for i in k_best_word_indices]

            # reset memory
            memory = []

        # Pick the best beam
        final_scores = [beam.score / len(beam.path[0]) for beam in to_evaluate]
        best_beam_index = np.argmax(final_scores)
        best_beam = to_evaluate[best_beam_index]

        # Build the full beam sequence by using the chain-list structure
        final_sequence = unloop_beam_serie(best_beam)

        # provide the full sequence
        full_path = final_sequence[-1].path[0]
        if "padded_tokens" in one_sample:
            full_path = list(one_sample["padded_tokens"][0]) + full_path

        return final_sequence, full_path


if __name__ == '__main__':

    with open("../../../config/qgen/config_fc8.json", 'r') as f_config:
        config = json.load(f_config)

    import guesswhat.data_provider as provider
    tokenizer = provider.GWTokenizer('../../../data/dict.json')
    config['model']['no_words'] = tokenizer.no_words

    network = QGenNetworkLSTM(config)

    # save_path = "/home/sequel/fstrub/rl_dialogue/tmp/qgen/{}"
    #
    # saver = tf.train.Saver()
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     saver.save(sess, save_path.format('params.ckpt'))
    #
    # tf.reset_default_graph()
    #
    # network = QGenNetworkLSTM(config)
    # saver = tf.train.Saver()
    with tf.Session() as sess:
        # saver.restore(sess, save_path.format('params.ckpt'))
        sess.run(tf.global_variables_initializer())

        pictures = np.zeros((2, 1000))

        dialogues = np.array([
                [2, 5, 3, 23,  5, 9, 3, 24, 4, 4],  # 2: start / 3: stop / 4: padding
                [2, 5, 6,  3, 24, 8, 9, 9, 3, 23],
            ])

        answer_mask = np.array([
                        [1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
                        [1., 1., 1., 1., 0., 1., 1., 1., 1., 0.]
            ])

        padding_mask = np.array([
                        [1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
        ])

        seq_length = np.array([8, 10])

        sess.run(network.ml_loss,
                 feed_dict={
                     network.picture_fc8: pictures,
                     network.dialogues: dialogues,
                     network.answer_mask: answer_mask,
                     network.padding_mask: padding_mask,
                     network.seq_length: seq_length
                 })

        one_sample = {
            "picture_fc8": np.zeros((1, 1000)),
        }
        network.eval_one_beam_search(sess, one_sample, tokenizer, max_depth=3)

        one_sample = {
            "picture_fc8": np.zeros((1, 1000)),
            "padded_tokens": np.array([[2, 5, 6, 3, 24, 8, 9, 9, 3, 23]])
        }

        network.eval_one_beam_search(sess, one_sample, tokenizer, max_depth=3)

        batch = {
            "picture_fc8": np.zeros((2, 1000)),
        }
        network.generate_next_question(sess, batch, tokenizer, max_depth=3)
        network.generate_next_question(sess, batch, tokenizer, greedy=True, max_depth=3)

        batch = {
            "picture_fc8": np.zeros((2, 1000)),
            "padded_tokens": dialogues
        }
        network.generate_next_question(sess, batch, tokenizer, max_depth=3)
        network.generate_next_question(sess, batch, tokenizer, greedy=True, max_depth=3)

        one_sample = {
            "picture_fc8": np.zeros((1, 1000)),
            "padded_tokens": np.array([[0, 2, 5, 6, 1, 23]])
        }

        trajectory = network.generate_next_question(sess, one_sample, tokenizer, greedy=True, max_depth=3, keep_trajectory=True)
