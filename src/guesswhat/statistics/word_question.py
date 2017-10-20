
import re
import numpy as np
from guesswhat.statistics.abstract_plotter import *

import seaborn as sns

class WordVsQuestion(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(WordVsQuestion, self).__init__(path, self.__class__.__name__, suffix)


        w_by_q = []
        for game in games:
            for q in game.questions:
                q = re.sub('[?]', '', q)
                words = re.findall(r'\w+', q)
                w_by_q.append(len(words))

        sns.set_style("whitegrid", {"axes.grid": False})

        # ratio question/words
        f = sns.distplot(w_by_q, norm_hist=True, kde=False, bins=np.arange(2.5, 15.5, 1), color="g")

        f.set_xlabel("Number of words", {'size': '14'})
        f.set_ylabel("Ratio of questions", {'size': '14'})
        f.set_xlim(2.5, 14.5)
        f.set_ylim(bottom=0)











