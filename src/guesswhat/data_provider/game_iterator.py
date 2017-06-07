import math
import numpy as np
import random
from multiprocessing import Queue
from threading import Thread

from guesswhat.data_provider.image_preprocessors import get_spatial_feat


def create_batch(buffer, tokenizer):
    batch_size = len(buffer)
    batch = {
        'picture_fc8': np.zeros((batch_size, 1000), dtype=np.float32),
        'targets': np.zeros((batch_size,), dtype=np.int32),
        'targets_category': np.zeros((batch_size,), dtype=np.int32),
        'targets_spatial': np.zeros((batch_size, 8), dtype=np.float32),  # magic number
        'raw': []
    }


    # Tokenize question and answers into one list
    dialog_tokens = []
    obj_spats = []
    obj_cats = []
    obj_targets = []

    # For debugging
    debug = []

    # Keep track of where the answers are in this token list
    all_answer_indices = []

    for i, game in enumerate(buffer):

        batch['raw'].append(game)

        # Flattened question answers
        q_tokens = [tokenizer.apply(q) for q in game.questions]
        a_tokens = [tokenizer.apply(a, is_answer=True) for a in game.answers]

        tokens = [tokenizer.start_token]  # Add start token
        answer_indices = []
        cur_index = 0
        for q_tok, a_tok in zip(q_tokens, a_tokens):
            tokens += q_tok
            tokens += a_tok

            # Compute index of answer in the full dialogue
            answer_indices += [cur_index + len(q_tok) + 1]
            cur_index = answer_indices[-1]

        tokens += [tokenizer.stop_dialogue] # Add STOP token


        dialog_tokens.append(tokens)
        all_answer_indices.append(answer_indices)

        # Picture fc8 features
        if game.picture.fc8 is not None:
            batch['picture_fc8'][i, :] = game.picture.fc8

        # Add objects: spatial features + categories
        spats = [get_spatial_feat(obj.bbox, game.picture.width,
                                  game.picture.height) for obj in game.objects]
        obj_spats.append(spats)
        cats = [obj.category_id for obj in game.objects]
        obj_cats.append(cats)

        # Look for the index of the object_id
        ind = None
        for j, obj in enumerate(game.objects):
            if obj.id == game.object_id:
                ind = j
                break
        assert ind is not None
        batch['targets'][i] = ind
        batch['targets_category'][i] = game.object.category_id
        batch['targets_spatial'][i] = get_spatial_feat(game.object.bbox, game.picture.width, game.picture.height)

        #Debgging
        debug.append((game.object.category, (game.object.bbox.x_center, game.object.bbox.y_center), game.picture.url))


    # Pad sequences to maximum length sequence
    seq_length = [len(l) for l in dialog_tokens]
    max_length = max(seq_length)

    padded_tokens = np.ones((batch_size, max_length), dtype=np.int32)*tokenizer.padding_token
    answer_mask = np.ones((batch_size, max_length), dtype=np.float32)
    padding_mask = np.ones((batch_size, max_length), dtype=np.float32)

    for i in range(batch_size):
        padded_tokens[i, :len(dialog_tokens[i])] = dialog_tokens[i]
        padding_mask[i, (seq_length[i]+1):] = 0.
        answer_mask[i, all_answer_indices[i]] = 0.

    # Pad objects to maximum number of objects
    max_objects = max([len(o) for o in obj_spats])
    object_categories = np.zeros((batch_size, max_objects), dtype=np.float32)
    object_spats = np.zeros((batch_size, max_objects, 8), dtype=np.float32)  # WARNING magic number
    mask = np.zeros((batch_size, max_objects), dtype=np.float32)
    for i in range(batch_size):
        object_spats[i, :len(obj_spats[i]), :] = obj_spats[i]
        object_categories[i, :len(obj_cats[i])] = obj_cats[i]
        mask[i, :len(obj_spats[i])] = 1.0

    batch['padded_tokens'] = padded_tokens
    batch['answer_mask'] = answer_mask
    batch['padding_mask'] = padding_mask
    batch['seq_length'] = np.array(seq_length)
    batch['obj_spats'] = object_spats
    batch['obj_cats'] = object_categories
    batch['obj_mask'] = mask

    batch['debug'] = debug

    return batch


def run_prefetch(queue, games, tokenizer, batch_size, sources, status, pad_to_batch_size):
    """Loops over dataset and puts preprocessed batches into queue."""

    buffer = []
    for game in games:

        # status=('success', 'failure', 'incomplete')
        if game.status in status:
            buffer.append(game)

            if len(buffer) == batch_size:
                batch = create_batch(buffer, tokenizer)
                buffer = []
                queue.put(batch)

    buffer_len = len(buffer)
    if buffer_len > 0:
        if pad_to_batch_size:
            for _ in range(batch_size - buffer_len):
                rand_index = np.random.randint(len(games))
                buffer.append(games[rand_index])
        batch = create_batch(buffer, tokenizer)
        queue.put(batch)

    queue.put(StopIteration)


class GameIterator:
    """Provides an iterator over the dataset."""
    sources = ('qas', 'cat', 'spat', 'object_id')

    def __init__(self, dataset, tokenizer, shuffle,
                 batch_size, status, sources=None, pad_to_batch_size=False):


        # Compute the number of games (according the status)
        filtered_games = [game for game in dataset.games if game.status in status]
        self.n_examples = len(filtered_games)
        self.batch_size = batch_size
        self.n_batches = int(math.ceil(1.*self.n_examples / self.batch_size))

        if shuffle:
            random.shuffle(filtered_games)

        self.queue = Queue(10)
        self.thread = Thread(target=run_prefetch,
                             args=(self.queue, filtered_games, tokenizer,
                                   self.batch_size, sources, status, pad_to_batch_size))
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
    dataset = Dataset('../../data',  'valid')
    tokenizer = GWTokenizer('../../data/dict.json')
    testIterator = GameIterator(dataset, tokenizer, batch_size=32, shuffle=True, status=("success",))

    from tqdm import tqdm

    i = 0
    for batch in tqdm(testIterator):
        i+= 1

    print("---------")
    print(i)
    print("---------")
