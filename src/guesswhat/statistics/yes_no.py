import collections

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



from guesswhat.statistics.abstract_plotter import *

class YesNo(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(YesNo, self).__init__(path, self.__class__.__name__, suffix)

        # basic storage for statistics
        yes_no = collections.defaultdict(list)
        number_yesno = collections.defaultdict(int)

        MAX = 15

        for i, game in enumerate(games):

            if game.status == "incomplete":
                continue

            yn = []
            for a in game.answers:

                a = a.lower()
                if a == "yes":
                    number_yesno["yes"] +=1
                    yn.append(1)
                elif a == "no":
                    number_yesno["no"] += 1
                    yn.append(0)
                else:
                    number_yesno["n/a"] += 1
                    yn.append(0.5)

            no_question = len(game.answers)
            yes_no[no_question].append(yn)


        sns.set(style="whitegrid")
        max_no_question = min(MAX, max(yes_no.keys())) + 1

        fig = None
        for key, yn in yes_no.items():

            no_question = int(key)
            yn_mean = np.array(yn).mean(axis=0)

            if no_question < max_no_question :
                fig = sns.regplot(x=np.arange(1, no_question + 1, 1), y=yn_mean, lowess=True, scatter=False)

        #dummy legend
        sns.regplot(x=np.array([-1]), y=np.array([-1]), scatter=False, line_kws={'linestyle':'-'}, label="Ratio yes-no",ci=None, color="g")
        fig.legend(loc="best", fontsize='x-large')

        fig.set_xlim(1, max_no_question)
        fig.set_ylim(0.1, 1)
        fig.set_xlabel("Number of questions", {'size': '14'})
        fig.set_ylabel('Ratio yes-no', {'size': '14'})

