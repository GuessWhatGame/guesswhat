
import collections
import numpy as np
from guesswhat.statistics.abstract_plotter import *

import seaborn as sns
import pandas as pd

class SuccessDialogueLength(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(SuccessDialogueLength, self).__init__(path, self.__class__.__name__, suffix)

        status_list = []
        status_count = collections.defaultdict(int)
        length_list = []

        for game in games:

            length_list.append(len(game.questions))

            status_count[game.status] += 1
            status_list.append(game.status)


        success = np.array([s == "success" for s in status_list]) + 0
        failure = np.array([s == "failure" for s in status_list]) + 0
        incomp  = np.array([s == "incomplete" for s in status_list]) + 0

        sns.set_style("whitegrid", {"axes.grid": False})

        if sum(incomp) > 0:
            columns = ['Size of Dialogues', 'Success', 'Failure', 'Incomplete']
            data = np.array([length_list, success, failure, incomp]).transpose()
        else:
            columns = ['Size of Dialogues', 'Success', 'Failure']
            data = np.array([length_list, success, failure]).transpose()

        df = pd.DataFrame(data, columns=columns)
        df = df.convert_objects(convert_numeric=True)
        df = df.groupby('Size of Dialogues').sum()
        df = df.div(df.sum(axis=1), axis=0)
        #df = df.sort_values(by='Success')
        f = df.plot(kind="bar", stacked=True, width=1, alpha=0.3)

        f.set_xlim(-0.5,29.5)

        plt.xlabel("Size of Dialogues", {'size':'14'})
        plt.ylabel("Success ratio", {'size':'14'})
