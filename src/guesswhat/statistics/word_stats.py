
import collections

import re
import numpy as np
from guesswhat.statistics.abstract_plotter import *

import pandas as pd
import seaborn as sns

stopwords = ["a", "an", "is", "it", "the", "does", "do", "are", "you", "that",
             "they", "doe", "this", "there", "hi", "his", "her", "its", "picture", "can", "he", "she", "bu", "us",
             "photo"]


class WordStat(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(WordStat, self).__init__(path, self.__class__.__name__, suffix)

        questions = []
        word_list = []
        word_counter = collections.Counter()

        for game in games:
            # split questions into words
            for q in game.questions:
                questions.append(q)
                q = re.sub('[?]', '', q)
                words = re.findall(r'\w+', q)
                word_list.append(words)

                for w in words:
                    word_counter[w.lower()] += 1

        size_voc = 0
        size_voc_3 = 0
        no_words = 0
        for key, val in word_counter.items():
            no_words += val
            size_voc += 1
            if val >= 3:
                size_voc_3 += 1

        logger.info("Number of dialogues: " + str(len(games)))
        logger.info("Number of words:     " + str(no_words))
        logger.info("voc size:            " + str(size_voc))
        logger.info("voc size (occ >3):   " + str(size_voc_3))
        logger.info("50 most common words")
        logger.info(word_counter.most_common(50))



    # override matplotlib methods
    def plot(self):
        pass

    # override matplotlib methods
    def save_as_pdf(self):
        pass



