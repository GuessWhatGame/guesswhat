
import re
import numpy as np
from guesswhat.statistics.abstract_plotter import *

import seaborn as sns

class DialogueSampler(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(DialogueSampler, self).__init__(path, self.__class__.__name__, suffix)

        for i, game in enumerate(games[:100]):

            logger.info("Dialogue {} : ".format(i))
            logger.info(" - picture : http://mscoco.org/images/{}".format(game.image.id))
            logger.info(" - object category : {}".format(game.object.category))
            logger.info(" - object position : {}, {}".format(game.object.bbox.x_center,game.object.bbox.y_center))
            logger.info(" - question/answers :")
            for q, a in zip(game.questions, game.answers):
                logger.info('  > ' + q + ' ? ' + a)
            logger.info('  #> ' + game.status)
            logger.info("")


    def save_as_pdf(self):
        pass


    def plot(self):
        pass