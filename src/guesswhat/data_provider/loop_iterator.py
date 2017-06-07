# TODO: TEMPO file need to factorize... I am the first obeto complain when there is some CODE duplication ;P

import collections
import math
import numpy as np
import random
from multiprocessing import Queue
from threading import Thread
from guesswhat.data_provider.image_preprocessors import get_spatial_feat


def create_batch(buffer):

    batch_size = len(buffer)
    batch = {
        'picture_fc8': np.zeros((batch_size, 1000), dtype=np.float32),
        'targets': np.zeros((batch_size,), dtype=np.int32),
        'targets_category': np.zeros((batch_size,), dtype=np.int32),
        'targets_spatial': np.zeros((batch_size, 8), dtype=np.float32)  # magic number
    }

    # Tokenize question and answers into one list
    obj_spats = []
    obj_cats = []

    # For debugging
    debug = []

    # Keep track of where the answers are in this token list

    for i, game in enumerate(buffer):

        # Picture fc8 features
        batch['picture_fc8'][i, :] = game.picture.fc8

        # Add objects: spatial features + categories
        spats = [get_spatial_feat(obj.bbox, game.picture.width, game.picture.height) for obj in game.objects]
        cats = [obj.category_id for obj in game.objects]

        # Pick one random object in the picture
        random_index = random.randint(0,len(game.objects)-1)
        target_object = game.objects[random_index]
        batch['targets'][i] = random_index
        batch['targets_category'][i] = cats[random_index]
        batch['targets_spatial'][i] = spats[random_index]

        # Store all objects info
        obj_cats.append(cats)
        obj_spats.append(spats)

        # Debugging
        debug.append((target_object.category, (target_object.bbox.x_center, target_object.bbox.y_center), game.picture.url))


    # Pad objects to maximum number of objects
    max_objects = max([len(o) for o in obj_spats])
    object_categories = np.zeros((batch_size, max_objects), dtype=np.float32)
    object_spats = np.zeros((batch_size, max_objects, 8), dtype=np.float32)  # WARNING magic number
    mask = np.zeros((batch_size, max_objects), dtype=np.float32)
    for i in range(batch_size):
        object_spats[i, :len(obj_spats[i]), :] = obj_spats[i]
        object_categories[i, :len(obj_cats[i])] = obj_cats[i]
        mask[i, :len(obj_spats[i])] = 1.0

    batch['obj_spats'] = object_spats
    batch['obj_cats'] = object_categories
    batch['obj_mask'] = mask

    batch['debug'] = debug


    return batch


def run_prefetch(queue, game_list, batch_size, sources):
    """Loops over dataset and puts preprocessed batches into queue."""

    def push_buffer(_buf):
        batch = create_batch(_buf)
        queue.put(batch)

    buffer = []
    for game in game_list:
        buffer.append(game)

        if len(buffer) == batch_size:
            push_buffer(buffer)
            buffer = []

    # push_buffer(buffer) -> bug with the guesser (Need to have a look on it)

    queue.put(StopIteration)


class LoopIterator:
    """Provides an iterator over the dataset."""
    sources = ('qas', 'cat', 'spat', 'object_id')

    def __init__(self, dataset, batch_size, no_sample=None, sources=None):

        # Step 1 : extract potential game
        potential_game_dico = {}
        for game in dataset.games:
            potential_game_dico[game.picture.id] = game

        potential_game_list = [game for game in potential_game_dico.values()]
        random.shuffle(potential_game_list)

        if no_sample is not None:
            potential_game_list = potential_game_list[:no_sample]

        self.n_examples = len(potential_game_list)
        self.batch_size = batch_size

        self.n_batches = int(math.ceil(1.*self.n_examples / self.batch_size))
        self.queue = Queue(10)

        self.thread = Thread(target=run_prefetch,
                             args=(self.queue, potential_game_list, self.batch_size, sources))
        self.thread.daemon = True
        self.thread.start()

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.queue.get(block=True)
        if batch == StopIteration:
            raise StopIteration
        return batch

    # trick for python 2.X
    def next(self):
        return self.__next__()


if __name__ == '__main__':
    from dataset import Dataset
    from nlp_preprocessors import GWTokenizer
    dataset = Dataset('/data/lisa/data/guesswhat',  'valid')
    tokenizer = GWTokenizer('/data/lisa/data/guesswhat/dict.json')
    testIterator = LoopIterator(dataset, batch_size=64)

    for batch in testIterator:
        print(batch[''])
