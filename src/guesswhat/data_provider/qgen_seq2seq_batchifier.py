import numpy as np
import collections
from itertools import chain

from generic.data_provider.batchifier import AbstractBatchifier, BatchifierSplitMode, batchifier_split_helper

from generic.data_provider.nlp_utils import padder, padder_3d
import copy


class Seq2SeqBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, sources, status=list(), glove=None):
        self.tokenizer = tokenizer
        self.sources = sources
        self.status = status
        self.glove = glove

    def split(self, games):

        games = batchifier_split_helper(games, split_mode=BatchifierSplitMode.DialogueHistory)

        games_with_end_of_dialogue = []

        end_of_dialogue = self.tokenizer.decode([self.tokenizer.stop_dialogue])
        for g in games:
            games_with_end_of_dialogue.append(g)

            if g.is_full_dialogue:
                game_with_eod = copy.copy(g)
                game_with_eod.questions = g.questions + [end_of_dialogue]
                game_with_eod.question_ids = g.question_ids + [-1]
                game_with_eod.answers = g.answers + ["n/a"]  # Dummy token

                games_with_end_of_dialogue.append(game_with_eod)

        games = games_with_end_of_dialogue

        return games

    def filter(self, games):

        if len(self.status) > 0:
            games = [g for g in games if g.status in self.status]

        return games

    def apply(self, games):

        batch = collections.defaultdict(list)
        batch_size = len(games)

        for i, game in enumerate(games):

            batch['raw'].append(game)

            # Encode question answers
            q_tokens = [self.tokenizer.encode(q) for q in game.questions]
            a_tokens = [self.tokenizer.encode(a, is_answer=True) for a in game.answers]

            # Flatten questions/answers except the last one
            dialogue = [self.tokenizer.start_token]  # Add start token
            for q_tok, a_tok in zip(q_tokens[:-1], a_tokens[:-1]):
                dialogue += q_tok
                dialogue += a_tok

            # Extract question to predict
            question = [self.tokenizer.start_token] + q_tokens[-1]
            if question[-1] != self.tokenizer.stop_token:
                question += [self.tokenizer.stop_token]

            batch["dialogue"].append(dialogue)
            batch["question"].append(question)

            # image
            if 'image' in self.sources:
                img = game.image.get_image()
                if "image" not in batch:  # initialize an empty array for better memory consumption
                    batch["image"] = np.zeros((batch_size,) + img.shape)
                batch["image"][i] = img

            # reward

        # Pad dialogue tokens tokens
        batch['dialogue'], batch['seq_length_dialogue'] = padder(batch['dialogue'], padding_symbol=self.tokenizer.padding_token)
        batch['question'], batch['seq_length_question'] = padder(batch['question'], padding_symbol=self.tokenizer.padding_token)

        return batch
