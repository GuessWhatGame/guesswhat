
import collections
import numpy as np
from guesswhat.statistics.abstract_plotter import *

import seaborn as sns

class SuccessPosition(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(SuccessPosition, self).__init__(path, self.__class__.__name__, suffix)
        x_bin = 7
        y_bin = 7

        success_sum = np.zeros((x_bin+1, y_bin+1))
        total_sum = np.zeros((x_bin+1, y_bin+1))

        for game in games:

            bbox = game.object.bbox
            picture = game.image

            x = int(bbox.x_center / picture.width * x_bin)
            y = int(bbox.y_center / picture.height * y_bin)

            total_sum[x][y] += 1.0

            if game.status == "success":
                success_sum[x][y] += 1.0

        ratio = 1.0 * success_sum / total_sum


        sns.set(style="whitegrid")


        # Draw the heatmap with the mask and correct aspect ratio
        f = sns.heatmap(ratio, robust=True, linewidths=.5, cbar_kws={"label" : "% Success"}, xticklabels=False, yticklabels=False)
        f.set_xlabel("normalized image width", {'size':'14'})
        f.set_ylabel("normalized image height", {'size':'14'})
        f.legend(loc="upper left", fontsize='x-large')



