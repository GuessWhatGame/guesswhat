import numpy as np
import collections

from generic.data_provider.batchifier import AbstractBatchifier
from generic.data_provider.nlp_utils import padder


class RNNBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, sources, status=list(), generate=False):
        self.sources = sources
        self.status = status
        self.tokenizer = tokenizer
        self.generate = generate
        assert self.generate, "Not yet implemented!"

    def filter(self, games):

        if len(self.status) > 0:
            games = [g for g in games if g.status in self.status]

        return games

    def apply(self, games, skip_targets=False):

        batch = collections.defaultdict(list)
        batch["raw"] = games

        batch_size = len(games)

        for i, game in enumerate(games):

            # Flattened question answers
            q_tokens = [self.tokenizer.encode(q) for q in game.questions]  # Do not add <stop_token> as it create a mismatch with RL learning
            a_tokens = [self.tokenizer.encode(a, is_answer=True) for a in game.answers]

            tokens = [self.tokenizer.start_token]  # Add START token
            for q_tok, a_tok in zip(q_tokens, a_tokens):
                tokens += q_tok
                tokens += a_tok
            tokens += [self.tokenizer.stop_dialogue]  # Add STOP token

            batch["dialogue"].append(tokens)
            batch["answer_mask"].append([int(token in self.tokenizer.answers) for token in tokens])

            # image
            if 'image' in self.sources:
                img = game.image.get_image()
                if "image" not in batch:  # initialize an empty array for better memory consumption
                    batch["image"] = np.zeros((batch_size,) + img.shape)
                batch["image"][i] = img

            # reward
            if "cum_reward" in self.sources and not skip_targets:
                assert False, "Not yet implemented!"

        # Pad dialogue tokens tokens
        batch['dialogue'], batch['seq_length_dialogue'] = padder(batch['dialogue'], padding_symbol=self.tokenizer.padding_token)
        batch['answer_mask'], _ = padder(batch['answer_mask'], padding_symbol=0)

        return batch
