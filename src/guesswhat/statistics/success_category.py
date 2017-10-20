
import collections
import numpy as np
from guesswhat.statistics.abstract_plotter import *

import seaborn as sns
import pandas as pd

class SuccessCategory(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(SuccessCategory, self).__init__(path, self.__class__.__name__, suffix)

        status_list = []
        status_count = collections.defaultdict(int)

        category_list = []
        category_count = collections.Counter()

        for game in games:

                category = game.object.category
                category_count[category] += 1
                category_list.append(category)

                status_count[game.status] += 1
                status_list.append(game.status)


        success = np.array([s == "success" for s in status_list]) + 0
        failure = np.array([s == "failure" for s in status_list]) + 0
        incomp  = np.array([s == "incomplete" for s in status_list]) + 0



        sns.set(style="white", color_codes=True)

        if sum(incomp) > 0:
            columns = ['Category', 'Success', 'Failure', 'Incomplete']
            data = np.array([category_list, success, failure, incomp]).transpose()
        else:
            columns = ['Category', 'Success', 'Failure']
            data = np.array([category_list, success, failure]).transpose()

        df = pd.DataFrame(data, columns=columns)
        df = df.convert_objects(convert_numeric=True)
        df = df.groupby('Category').sum()
        df = df.div(df.sum(axis=1), axis=0)
        df = df.sort_values(by='Success')
        df.plot(kind="bar", stacked=True, width=1, alpha=0.3, figsize=(14,6), color=["g", "r", "b"])
        sns.set(style="whitegrid")

        plt.xlabel("Categories", {'size':'14'})
        plt.ylabel("Success ratio", {'size':'14'})





