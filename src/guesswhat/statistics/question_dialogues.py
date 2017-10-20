
import re
import numpy as np
from guesswhat.statistics.abstract_plotter import *

import seaborn as sns

class QuestionVsDialogue(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(QuestionVsDialogue, self).__init__(path, self.__class__.__name__, suffix)

        q_by_d = []
        for game in games:
            q_by_d.append(len(game.questions))

        sns.set_style("whitegrid", {"axes.grid": False})


        #ratio question/dialogues
        f = sns.distplot(q_by_d, norm_hist =True, kde=False, bins=np.arange(0.5, 25.5, 1))
        f.set_xlim(0.5,25.5)
        f.set_ylim(bottom=0)

        f.set_xlabel("Number of questions", {'size':'14'})
        f.set_ylabel("Ratio of dialogues", {'size':'14'})


