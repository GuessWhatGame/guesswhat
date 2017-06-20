import numpy as np
import collections
import random
import copy

from generic.data_provider.batchifier import AbstractBatchifier

from generic.data_provider.image_preprocessors import get_spatial_feat
from generic.data_provider.nlp_utils import padder, padder_3d


class LooperBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, sources, train, **kwargs):
        self.tokenizer = tokenizer
        self.sources = sources
        self.train = train
        self.kwargs = kwargs

    def filter(self, games):

        # Step 1 : extract potential game
        if self.train:
            potential_game_dico = {}
            for game in games:
                potential_game_dico[game.picture.id] = game

            potential_game_list = [game for game in potential_game_dico.values()]
            random.shuffle(potential_game_list)

            return copy.deepcopy(potential_game_list) # deep copy to perserve the original dataset
        else:
            return games

    def apply(self, games):

        batch = collections.defaultdict(list)
        batch_size = len(games)

        for i, game in enumerate(games):

            batch['raw'].append(game)

            # Add objects: spatial features + categories (Guesser)
            obj_spats = [get_spatial_feat(obj.bbox, game.picture.width, game.picture.height) for obj in game.objects]
            obj_cats = [obj.category_id for obj in game.objects]

            batch['obj_spats'].append(obj_spats)
            batch['obj_cats'].append(obj_cats)

            # Pick one random object in the game: TODO clean a bit
            if self.train:
                random_index = random.randint(0, len(game.objects) - 1)
            else:
                random_index = game.objects.index(game.object)

            target_object = game.objects[random_index]

            # update the game with the target object
            game.object = target_object
            game.object_id = target_object.id

            batch['targets_index'].append(random_index)
            batch['targets_spatial'].append(obj_spats[random_index])
            batch['targets_category'].append(obj_cats[random_index])

            batch['debug'].append((target_object.category, (target_object.bbox.x_center, target_object.bbox.y_center), game.picture.url))

            # image
            img = game.picture.get_image()
            if img is not None:
                if "images" not in batch:  # initialize an empty array for better memory consumption
                    batch["images"] = np.zeros((batch_size,) + img.shape)
                batch["images"][i] = img


        # Pad objects
        batch['obj_spats'], obj_length = padder_3d(batch['obj_spats'])
        batch['obj_cats'], obj_length = padder(batch['obj_cats'])

        # Compute the object mask
        max_objects = max(obj_length)
        batch['obj_mask'] = np.zeros((batch_size, max_objects), dtype=np.float32)
        for i in range(batch_size):
            batch['obj_mask'][i, :obj_length[i]] = 1.0

        return batch

