
import collections

import re
import numpy as np
from guesswhat.statistics.abstract_plotter import *

import pandas as pd
import seaborn as sns

stopwords = ["a", "an", "is", "it", "the", "does", "do", "are", "you", "that",
             "they", "doe", "this", "there", "hi", "his", "her", "its", "picture", "can", "he", "she", "bu", "us",
             "photo"]


class WordCoocurence(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(WordCoocurence, self).__init__(path, self.__class__.__name__, suffix)

        questions = []
        word_counter = collections.Counter()

        NO_WORDS_TO_DISPLAY = 50

        for game in games:
            # split questions into words
            for q in game.questions:
                questions.append(q)
                q = re.sub('[?]', '', q)
                words = re.findall(r'\w+', q)

                for w in words:
                    word_counter[w.lower()] += 1


        # compute word co-coocurrence
        common_words = word_counter.most_common(NO_WORDS_TO_DISPLAY)
        common_words = [pair[0] for pair in common_words]
        corrmat = np.zeros((NO_WORDS_TO_DISPLAY, NO_WORDS_TO_DISPLAY))

        # compute the correlation matrices
        for i, question in enumerate(questions):
            for word in question:
                if word in common_words:
                    for other_word in question:
                        if other_word in common_words:
                            if word != other_word:
                                corrmat[common_words.index(word)][common_words.index(other_word)] += 1.

        # Display the cor matrix
        df = pd.DataFrame(data=corrmat, index=common_words, columns=common_words)
        f = sns.clustermap(df, standard_scale=0, col_cluster=False, row_cluster=True, cbar_kws={"label": "co-occurence"})
        f.ax_heatmap.xaxis.tick_top()

        plt.setp(f.ax_heatmap.get_xticklabels(), rotation=90)
        plt.setp(f.ax_heatmap.get_yticklabels(), rotation=0)






