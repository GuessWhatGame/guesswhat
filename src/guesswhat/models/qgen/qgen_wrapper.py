import numpy as np
import re

from guesswhat.models.qgen.qgen_sampling_wrapper import QGenSamplingWrapper
from guesswhat.models.qgen.qgen_beamsearch_wrapper import QGenBSWrapper

# This is very ugly code that must be refactored.
# To avoid breaking future code, we hide the implementation behind this Decorator
# Implementation of sampling was updated for speed reason while eam search rely ion legacy code
# Therefore, their internal implementation differs. that iw why we put a wrapper to hide technical detail in the looper

class QGenWrapper(object):
    def __init__(self, qgen, tokenizer, max_length, k_best):

        self.sampling_wrapper = QGenSamplingWrapper(qgen, tokenizer, max_length)
        self.bs_wrapper = QGenBSWrapper(qgen, tokenizer, max_length, k_best)
        self.qgen = qgen

    def initialize(self, sess):
        self.sampling_wrapper.initialize(sess)
        self.bs_wrapper.initialize(sess)

    def reset(self, batch_size):
        self.sampling_wrapper.reset(batch_size)
        self.bs_wrapper.reset(batch_size)

    def sample_next_question(self, sess, prev_answers, game_data, mode):

        if mode == "sampling":
            return self.sampling_wrapper.sample_next_question(sess, prev_answers, game_data, greedy=False)
        elif mode == "greedy":
            return self.sampling_wrapper.sample_next_question(sess, prev_answers, game_data, greedy=True)
        elif mode == "beam_search":
            return self.bs_wrapper.sample_next_question(sess, prev_answers, game_data)
        else:
            assert False, "Invalid samppling mode: {}".format(mode)


class QGenUserWrapper(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def initialize(self, sess):
        pass

    def reset(self, batch_size):
        pass

    def sample_next_question(self, _, prev_answers, game_data, **__):

        if prev_answers[0] == self.tokenizer.start_token:
            print("Type the character '(S)top' when you want to guess the object")
        else:
            print("A :", self.tokenizer.decode(prev_answers[0]))

        print()
        while True:
            question = input('Q: ')
            if question != "":
                break

        # Stop the dialogue
        if question == "S" or question == "Stop":
            tokens = [self.tokenizer.stop_dialogue]

        # Stop the question (add stop token)
        else:
            question = re.sub('\?', '', question) # remove question tags if exist
            question +=  " ?"
            tokens = self.tokenizer.apply(question)

        return [tokens], np.array([tokens]), [len(tokens)]