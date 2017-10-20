
import json
from pprint import pprint
import itertools
import collections

import wordcloud as wc

import re
import sys


from guesswhat.statistics.abstract_plotter import *

stopwords = ["a", "an", "is", "it", "the", "does", "do", "are", "you", "that",
             "they", "doe", "this", "there", "hi", "his", "her", "its", "picture", "can", "he", "she", "bu", "us", "photo"]

class WordCloud(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(WordCloud, self).__init__(path, self.__class__.__name__, suffix)

        questions = []

        for game in games:
            questions.append(game.questions)
        questions = list(itertools.chain(*questions))


        # split questions into words
        word_list = []
        word_counter = collections.Counter()
        for q in questions:
            q = re.sub('[?]', '', q)
            words = re.findall(r'\w+', q)
            word_list.append(words)

            for w in words:
                word_counter[w.lower()] += 1


        word_list = list(itertools.chain(*word_list))
        pprint(word_counter)

        def color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
            color_list =["green",'blue', 'brown', "red", 'white', "black", "yellow", "color", "orange", "pink"]
            people_list  =['people', 'person', "he", "she", "human", "man", "woman", "guy", 'alive', "girl", "boy", "head", 'animal']
            prep = ['on', "in", 'of', 'to', "with", "by", "at", "or", "and", "from"]
            number = ['one', "two", "three", "four", "five", "six", "first", "second", "third", "half"]
            spatial = ["top", "left", "right", "side", "next", "front", "middle", "foreground", "bottom", "background",
                       "near", "behind", "back", "at", "row", "far", "whole", "closest"]
            verb=["wearing", "have", "can", "holding", "sitting", "building", "standing", "see"]
            obj = ["hand","table", 'car', "food", "plate", "shirt", "something", "thing", "object",
                   "light", "hat", "tree", "bag", "book", "sign", "bottle", "glass", "bus", "wall", "vehicle",
                   "chair", "dog", "cat", "windows", "boat", "item", "shelf", "horse", "furniture", "water", "camera", "bike",
                   "train", "window", "bowl", "plant", "ball", "cup", ]
            misc = [ 'visible', "made", "part", "piece", "all"]

            if word in color_list: return 'rgb(0, 102, 204)' #blue
            if word in people_list: return  'rgb(255, 0, 0)' #red
            if word in prep: return 'rgb(0, 153, 0)' #green
            if word in number: return 'rgb(255, 128, 0)' #orange
            if word in spatial: return 'rgb(204, 0, 102)' #purple
            if word in verb: return 'rgb(0, 204, 102)' #turquoise
            if word in obj: return 'rgb(64, 64, 64)' #grey
            if word in misc: return 'rgb(102, 102, 0)' #yellow
            else:
                logger.warning("Unexpected in cloud of words : " + word)
                return 'rgb(0, 0, 0)'


        # take relative word frequencies into account, lower max_font_size
        wordcloud = wc.WordCloud(background_color="white", color_func=color_func, max_font_size=40, max_words=80,
                              stopwords=stopwords, prefer_horizontal=1, width=400, height=200)\
            .generate(" ".join(str(x) for x in word_list))

        plt.figure()
        plt.imshow(wordcloud)
        plt.axis("off")

