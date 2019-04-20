import numpy as np
import collections

from generic.data_provider.batchifier import AbstractBatchifier
from generic.data_provider.nlp_utils import padder

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

class RNNBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, sources, status=list(), supervised=False, generate=False, reward_type="MC"):
        self.sources = sources
        self.status = status
        self.tokenizer = tokenizer
        self.supervised = supervised
        self.generate = generate
        self.reward_type = reward_type

    def filter(self, games):

        if len(self.status) > 0:
            games = [g for g in games if g.status in self.status]

        return games

    def apply(self, games, skip_targets=False):

        batch = collections.defaultdict(list)
        batch["raw"] = games

        batch_size = len(games)

        for i, game in enumerate(games):

            # TODO use enum
            if self.supervised:
                dialogue_tokens = self.supervised_training_input(game)
            elif self.generate:
                dialogue_tokens, new_answer = self.generate_input(game)
                batch["new_answer"].append(new_answer)
            elif not self.supervised and not self.generate:
                dialogue_tokens = self.rl_training_input(game)
            else:
                assert False

            batch["dialogue"].append(dialogue_tokens)
            batch["answer_mask"].append([int(token in self.tokenizer.answers + [self.tokenizer.padding_token])
                                         for token in dialogue_tokens])

            # image
            if 'image' in self.sources:
                img = game.image.get_image()
                if "image" not in batch:  # initialize an empty array for better memory consumption
                    batch["image"] = np.zeros((batch_size,) + img.shape)
                batch["image"][i] = img

            # reward
            if "cum_reward" in self.sources and not skip_targets and not self.supervised:

                # if self.reward_type == "monte_carlo":
                reward = int(game.status == "success")
                batch["cum_reward"].append([reward] * len(batch["dialogue"][i]))
                #
                # if self.reward_type == "n_step":
                # n = 0
                # reward = [0] * len(batch["dialogue"][i])
                # reward[-1] = int(game.status == "success")
                #
                # values = []
                # for qv in game.user_data["state_values"]:
                #     values += qv
                #     values += [0.]
                #
                # next_values = []
                # for i, (tok, next_tok) in enumerate(zip(dialogue_tokens, dialogue_tokens[1:])):
                #     if tok == self.tokenizer.stop_token:
                #         next_values += [0.]
                #     if next_tok == self.tokenizer.stop_token:
                #         next_values += [reward[i + 2] + values[i+2]]
                #     else:
                #         next_values += [reward[i + 1] + values[i + 1]]



        # Pad dialogue tokens tokens
        batch['dialogue'], batch['seq_length_dialogue'] = padder(batch['dialogue'], padding_symbol=self.tokenizer.padding_token)
        batch['answer_mask'], _ = padder(batch['answer_mask'], padding_symbol=0)

        if 'cum_reward' in batch:
            batch['cum_reward'], _ = padder(batch['cum_reward'], padding_symbol=0)

        return batch

    def supervised_training_input(self, game):

        q_tokens = [self.tokenizer.encode(q, add_stop_token=True) for q in game.questions]
        a_tokens = [self.tokenizer.encode(a, is_answer=True) for a in game.answers]

        tokens = [self.tokenizer.start_token]  # Add START token
        for q_tok, a_tok in zip(q_tokens, a_tokens):
            tokens += q_tok
            tokens += a_tok
        tokens += [self.tokenizer.stop_dialogue]  # Add STOP token

        return tokens

    def rl_training_input(self, game):

        q_tokens = [self.tokenizer.encode(q, add_stop_token=False) for q in game.questions]  # Do not add <?> as it creates a mismatch with RL learning
        a_tokens = [self.tokenizer.encode(a, is_answer=True) for a in game.answers]

        # Add a dummy answer if the stop_dialogue token is present
        a_tokens.append([])

        tokens = [self.tokenizer.start_token]  # Add START token
        for q_tok, a_tok in zip(q_tokens, a_tokens):
            tokens += q_tok
            tokens += a_tok

        return tokens

    def generate_input(self, game):

        if not game.questions:
            tokens = [self.tokenizer.start_token, self.tokenizer.stop_token]
            new_answer = self.tokenizer.start_token

        else:
            q_tokens = [self.tokenizer.encode(q, add_stop_token=False) for q in game.questions]  # Do not add <?> as it creates a mismatch with RL learning
            a_tokens = [self.tokenizer.encode(a, is_answer=True) for a in game.answers]

            tokens = [self.tokenizer.start_token]  # Add START token
            for q_tok, a_tok in zip(q_tokens, a_tokens):
                tokens += q_tok
                tokens += a_tok

            new_answer = tokens[-1]
            tokens = tokens[:-1]  # the answer becomes the start token

        return tokens, new_answer
