import numpy as np
import collections
import random
import copy

from generic.data_provider.batchifier import AbstractBatchifier


class LooperBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, generate_new_games):
        self.tokenizer = tokenizer
        self.generate_new_games = generate_new_games

    def filter(self, games):

        if self.generate_new_games:

            # Create one game per image
            new_games_dico = {}
            for game in games:
                new_games_dico[game.image.id] = game

            games = [game for game in new_games_dico.values()]
            random.shuffle(games)

        return games

    def split(self, games):

        new_games = []
        for i, g in enumerate(games):

            # Defensive copy
            g = copy.deepcopy(g)

            # Reset game data
            g.dialogue_id = i
            g.questions = []
            g.question_ids = []
            g.answers = []
            g.status = "incomplete"
            g.is_full_dialogue = False

            # Pick random new object
            if self.generate_new_games:
                random_index = random.randint(0, len(g.objects) - 1)
                g.object = g.objects[random_index]
                g.object_id = g.object.id

            new_games.append(g)

        return new_games

    def apply(self, games):

        batch = collections.defaultdict(list)
        batch_size = len(games)

        for i, game in enumerate(games):

            batch['raw'].append(game)

            # image
            img = game.image.get_image()
            if img is not None:
                if "image" not in batch:  # initialize an empty array for better memory consumption
                    batch["image"] = np.zeros((batch_size,) + img.shape)
                batch["image"][i] = img

        return batch

