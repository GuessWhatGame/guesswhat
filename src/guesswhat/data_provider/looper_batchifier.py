import random
import copy
import collections
from generic.data_provider.batchifier import AbstractBatchifier


class LooperBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, generate_new_games):
        self.tokenizer = tokenizer
        self.generate_new_games = generate_new_games

    def filter(self, games):

        if self.generate_new_games:

            # Create one game per image
            new_games_dict = {}
            for game in games:
                new_games_dict[game.image.id] = game

            games = [game for game in new_games_dict.values()]
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
            g.user_data = {"has_stop_token": False}

            # Pick random new object
            if self.generate_new_games:
                random_index = random.randint(0, len(g.objects) - 1)
                g.object = g.objects[random_index]

            new_games.append(g)

        return new_games

    def apply(self, games, skip_targets=False):

        batch = collections.defaultdict(list)
        batch["raw"] = games

        # Optim to preload image in memory using an external thread
        # Note that the memory must be manually free once the batch is consumed! (game.flush())
        for i, game in enumerate(games):
            game.bufferize()

        return batch
