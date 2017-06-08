import tensorflow as tf
import numpy as np
import collections
import copy



BeamToken = collections.namedtuple('BeamToken', ['path', 'word_id', 'decoder_state',
                                                 'score', 'prev_beam'])


# The Beam token is the key element of question generation
#    - path are the outputed words
#    - word_id are the next word inputs
#    - decoder_state is the state of the decoder after outputed path[-1]
#    - score is the \sum log(prob(w)) (beam search only)
#    - prev_beam chain the previous beam (to store the hidden state for each outputed words)


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


def create_initial_beam(qgen, batch, tokenizer, decoder_state=None, batch_size=1):

    if "padded_tokens" in batch:
        batch_size = batch["padded_tokens"].shape[0]
        initial_words_id = batch["padded_tokens"]
    else:
        initial_words_id = [[tokenizer.start_token]] * batch_size

    if decoder_state is None:
        decoder_state = tf.contrib.rnn.LSTMStateTuple(
            c=np.zeros((batch_size, int(qgen.decoder_zero_state_c.get_shape()[1]))),
            h=np.zeros((batch_size, int(qgen.decoder_zero_state_h.get_shape()[1]))))

    return BeamToken(
        path=[[] for _ in range(batch_size)],
        word_id=initial_words_id,
        decoder_state=decoder_state,
        score=0,  # initial probability is 1. If we apply the log trick log(1) = 0
        prev_beam=None
    )




# Poor quality code: TODO refactor
def eval_one_beam_search(sess, qgen, one_sample, tokenizer, initial_beam=None, max_depth=12, k_best=20, keep_trajectory=False):

    # Check that one_sample has a single element and there is no padding in word history

    if initial_beam is None:
        assert one_sample["images"].shape[0] == 1
        assert "padded_tokens" not in one_sample or one_sample["padded_tokens"][0][-1] != tokenizer.padding_token

        # Create the first beam token that will generate the sequence
        initial_beam = create_initial_beam(qgen, one_sample, batch_size=1, tokenizer=tokenizer)

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
            softmax, decoder_state = sess.run([qgen.softmax_output, qgen.decoder_state],
                                              feed_dict={
                                                  qgen.picture_fc8: one_sample["images"],
                                                  qgen.dialogues: dialogue_history,
                                                  qgen.seq_length: [dialogue_history.shape[1]],
                                                  qgen.decoder_zero_state_c: beam_token.decoder_state.c,
                                                  qgen.decoder_zero_state_h: beam_token.decoder_state.h
                                              })

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





# legacy code for beam search and sampling
######################################################

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
