import tensorflow as tf
import numpy as np
import collections
from tqdm import tqdm
import copy

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
