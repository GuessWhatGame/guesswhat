import tensorflow as tf
import numpy as np
import collections

from generic.tf_utils.evaluator import Evaluator

# The Beam token is the key element of question generation
#    - path are the outputed words
#    - word_id are the next word inputs
#    - decoder_state is the state of the decoder after outputed path[-1]
#    - score is the \sum log(prob(w)) (beam search only)
#    - prev_beam chain the previous beam (to store the hidden state for each outputed words)

BeamToken = collections.namedtuple('BeamToken', ['path', 'word_id', 'decoder_state',
                                                 'score', 'prev_beam'])
def unloop_beam_serie(cur_beam):

    # Build the full beam sequence by using the chain-list structure
    sequence = [cur_beam]
    while cur_beam.prev_beam is not None:
        cur_beam = cur_beam.prev_beam
        sequence.append(cur_beam)

    return sequence[::-1]  # reverse sequence


def create_initial_beam(decoder_state_size, batch_size=1):

    decoder_state = tf.contrib.rnn.LSTMStateTuple(
        c=np.zeros((1, decoder_state_size)),
        h=np.zeros((1, decoder_state_size))
    )

    return BeamToken(
        path=[[] for _ in range(batch_size)],
        word_id=[[] for _ in range(batch_size)],
        decoder_state=decoder_state,
        score=0,  # initial probability is 1. If we apply the log trick log(1) = 0
        prev_beam=None
    )


class QGenBSWrapper(object):
    def __init__(self, qgen, tokenizer, max_length, k_best):

        self.qgen = qgen

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.k_best = k_best

        self.beam = None

        self.state_size = int(qgen.decoder_zero_state_c.get_shape()[1])

        self.max_length = max_length

    def initialize(self, sess):
        pass

    def reset(self, batch_size):
        self.beam = [create_initial_beam(decoder_state_size=self.state_size,batch_size=1)
                     for _ in range(batch_size)]


    def sample_next_question(self, sess, prev_answers, game_data):

        for i, one_beam in enumerate(self.beam):

            # split the batch into mini_batch of size 1
            one_sample = {}
            for k, val in game_data.items():
                one_sample[k] = [val[i]]

            # Prepare beam by appending answer and removing previous path
            one_beam.word_id[0].append(prev_answers[i][0])
            one_beam.path[0] = list()

            # Execute beam search
            new_beam = self.eval_one_beam_search(sess, one_sample, one_beam)

            # Store current beam (with LSTM state)
            self.beam[i] = new_beam

        # Compute output
        questions =  [b.path[0] for b in self.beam]
        seq_length = [len(q) for q in questions]

        padded_questions = np.full((len(self.beam), max(seq_length)), fill_value=self.tokenizer.padding_token)
        for i, (q, l) in enumerate(zip(questions, seq_length)):
            padded_questions[i, :l] = q

        return padded_questions, questions, seq_length



    # Legacy code: TODO refactor and use tf graph
    def eval_one_beam_search(self, sess, one_sample, initial_beam, keep_trajectory=False):

        to_evaluate = [initial_beam]

        memory = []
        for depth in range(self.max_length):

            # evaluate all the current tokens
            for beam_token in to_evaluate:

                # if token is final token, directly put it into memory
                if beam_token.word_id[0][-1] in [self.tokenizer.stop_token, self.tokenizer.stop_dialogue]:
                    memory.append(beam_token)
                    continue

                # Append a dummy STOP token to fit HQGen constraint (can also be a PADDING)
                dialogue_history = np.concatenate((beam_token.word_id, [[self.tokenizer.stop_token]]), axis=1)

                # evaluate next_step
                softmax, decoder_state = sess.run([self.qgen.softmax_output, self.qgen.decoder_state],
                                                  feed_dict={
                                                      self.qgen.images: one_sample["images"],
                                                      self.qgen.dialogues: dialogue_history,
                                                      self.qgen.seq_length: [dialogue_history.shape[1]],
                                                      self.qgen.decoder_zero_state_c: beam_token.decoder_state.c,
                                                      self.qgen.decoder_zero_state_h: beam_token.decoder_state.h
                                                  })

                # Reshape tensor (remove 1 size batch)
                softmax = softmax[0, -1]

                # put into memory the k-best tokens of this sample
                k_best_word_indices = np.argpartition(softmax, -self.k_best)[-self.k_best:]
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
            k_best_word_indices = np.argpartition(scores, -self.k_best)[-self.k_best:]
            to_evaluate = [memory[i] for i in k_best_word_indices]

            # reset memory
            memory = []

        # Pick the best beam
        final_scores = [beam.score / len(beam.path[0]) for beam in to_evaluate]
        best_beam_index = np.argmax(final_scores)
        best_beam = to_evaluate[best_beam_index]

        return best_beam




