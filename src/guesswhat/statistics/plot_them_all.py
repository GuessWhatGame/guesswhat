import argparse

import logging
from logging.handlers import RotatingFileHandler

import matplotlib
matplotlib.use('Agg')

from guesswhat.data_provider.guesswhat_dataset import Dataset

from guesswhat.statistics.dialogue_sampler import *
from guesswhat.statistics.yes_no import *
from guesswhat.statistics.word_stats import *
from guesswhat.statistics.word_coocurrence import *
from guesswhat.statistics.word_cloud import *
from guesswhat.statistics.word_question import *
from guesswhat.statistics.success_area import *
from guesswhat.statistics.success_category import *
from guesswhat.statistics.success_no_objects import *
from guesswhat.statistics.success_position import *
from guesswhat.statistics.sucess_dialogue_length import *
from guesswhat.statistics.question_object import *
from guesswhat.statistics.question_dialogues import *



def create_logger(save_path, name):

    logger = logging.getLogger()
    # Debug = write everything
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    file_handler = RotatingFileHandler(save_path + '/' + name + '.stats.log', 'a', 1000000, 1)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(logging.DEBUG)
    logger.addHandler(steam_handler)

    return logger

# List all the class to plot
prototypes = [
    DialogueSampler,
    YesNo,
    WordStat,
    WordVsQuestion,
    # WordCoocurence, # Buggy
    SuccessDialogueLength,
    SuccessPosition,
    SuccessNoObject,
    SuccessCategory,
    SuccessArea,
    QuestionVsObject,
    QuestionVsDialogue,
    WordCloud
]


def save_plots(in_dir, out_dir, name, ignore_incomplete):


    stat_logger = create_logger(out_dir, name)
    dataset = Dataset(in_dir, name)

    if ignore_incomplete:
        dataset.games = [g for g in dataset.games if g.status == "success" or g.status == "failure"]

    for prototype in prototypes:
        p = prototype(out_dir, dataset.games, stat_logger, name)
        p.save_as_pdf()


if __name__ == '__main__':


    parser = argparse.ArgumentParser('Plotter options!')

    parser.add_argument("-data_dir", type=str, help="Directory with data", required=True)
    parser.add_argument("-out_dir", type=str, help="Output directory", required=True)
    parser.add_argument("-name", type=str, help="Output directory", required=True)
    parser.add_argument("-ignore_incomplete", type=bool, default=True, help="Ignore incomplete games in the dataset")

    args = parser.parse_args()

    save_plots(in_dir=args.data_dir,
               out_dir=args.out_dir,
               name=args.name,
               ignore_incomplete=args.ignore_incomplete)






