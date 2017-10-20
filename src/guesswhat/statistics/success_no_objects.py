

import collections
import numpy as np
from guesswhat.statistics.abstract_plotter import *

import seaborn as sns
import pandas as pd

class SuccessNoObject(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(SuccessNoObject, self).__init__(path, self.__class__.__name__, suffix)

        status_count = collections.Counter()
        status_list = []

        objects = []

        for game in games:

            status_count[game.status] += 1
            status_list.append(game.status)

            objects.append(len(game.objects))


        sns.set(style="whitegrid", color_codes=True)

        success = np.array([s == "success" for s in status_list]) + 0
        failure = np.array([s == "failure" for s in status_list]) + 0
        incomp  = np.array([s == "incomplete" for s in status_list]) + 0



        if sum(incomp) > 0:
            columns = ['No objects', 'Success', 'Failure', 'Incomplete']
            data = np.array([objects, success, failure, incomp]).transpose()
        else:
            columns = ['No objects', 'Success', 'Failure']
            data = np.array([objects, success, failure]).transpose()


        df = pd.DataFrame(data, columns=columns)
        df = df.convert_objects(convert_numeric=True)
        df = df.groupby('No objects').sum()
        f = df.plot(kind="bar", stacked=True, width=1, alpha=0.3, color=["g", "r", "b"])

        sns.regplot(x=np.array([0]), y=np.array([0]), scatter=False, line_kws={'linestyle':'--'}, label="% Success",ci=None, color="b")


        #f.set_xlim(0.5,18.5)
        #f.set_ylim(0,25000)
        f.set_xlabel("Number of objects", {'size':'14'})
        f.set_ylabel("Number of dialogues", {'size':'14'})
        f.legend(loc="best", fontsize='large')



        ###########################


        success = np.array([s == "success" for s in status_list])
        failure = np.array([s == "failure" for s in status_list])
        incomp  = np.array([s == "incomplete" for s in status_list])


        objects = np.array(objects)
        rng = range(3, 22)
        histo_success = np.histogram(objects[success], bins=rng)
        histo_failure = np.histogram(objects[failure], bins=rng)
        histo_incomp  = np.histogram(objects[incomp], bins=rng)


        normalizer = histo_success[0] + histo_failure[0] + histo_incomp[0]
        histo_success = 1.0*histo_success[0] / normalizer
        histo_failure = 1.0*histo_failure[0] / normalizer
        histo_incomp  = 1.0*histo_incomp[0]  / normalizer


        ax2 = f.twinx()

        curve = np.ones(len(normalizer))-histo_failure-histo_incomp
        f = sns.regplot(x=np.linspace(1, 20, 18), y=curve, order=3, scatter=False, line_kws={'linestyle':'--'},ci=None, truncate=False, color="b")
        ax2.set_ylim(0,1)
        ax2.grid(None)
        ax2.set_ylabel("Success ratio", {'size':'14'})



