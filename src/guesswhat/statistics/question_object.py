
import numpy as np
from guesswhat.statistics.abstract_plotter import *

import seaborn as sns
import matplotlib.pyplot as plt

import collections

class QuestionVsObject(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(QuestionVsObject, self).__init__(path, self.__class__.__name__, suffix)

        ratio_q_object = []
        for game in games:

            no_object = len(game.objects)
            no_question = len(game.questions)

            ratio_q_object.append([no_object,no_question])

        ratio_q_object = np.array(ratio_q_object)

        sns.set(style="white")

        x = np.linspace(3, 20, 80)
        counter = collections.defaultdict(list)

        for k, val in ratio_q_object:
            counter[k] += [val]

        arr = np.zeros( [4, 21])
        for k, val in counter.items():
            if len(val) > 0:
                arr[0,k] = k
                arr[1,k] = np.mean(val)

                # Std
                arr[2, k] = np.std(val)

                # confidence interval 95%
                arr[3,k] = 1.95*np.std(val)/np.sqrt(len(val))


        #plt.plot(arr[0,:],arr[1,:] , 'b.', label="Human behavior")
        sns.regplot(x=ratio_q_object[:, 0], y=ratio_q_object[:, 1], x_ci=None, x_bins=20, order=4,  label="Human behavior", marker="o", line_kws={'linestyle':'-'})
        plt.fill_between(x=arr[0,:], y1=arr[1,:]-arr[2,:], y2=arr[1,:]+arr[2,:], alpha=0.2)

        sns.regplot    (x=x, y=np.log2(x), order=6, scatter=False, label="y = log2(x)", line_kws={'linestyle':'--'})
        f = sns.regplot(x=x, y=x         , order=1, scatter=False, label="y = x"      , line_kws={'linestyle':'--'})

        f.legend(loc="best", fontsize='x-large')
        f.set_xlim(3,20)
        f.set_ylim(0,20)
        f.set_xlabel("Number of objects", {'size':'14'})
        f.set_ylabel("Number of questions", {'size':'14'})






