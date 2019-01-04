import numpy as np
import collections

from generic.data_provider.batchifier import AbstractBatchifier, BatchifierSplitMode, batchifier_split_helper

from generic.data_provider.nlp_utils import padder
import copy


def compute_cumulative_rewards(reward, gamma=1):
    if not isinstance(gamma, list):
        gamma = [gamma] * len(reward)

    cum_reward = [reward[-1]]
    for i, (r, g) in enumerate(zip(reversed(reward[:-1]), reversed(gamma[:-1]))):
        cum_reward += [r + g * cum_reward[i]]

    return cum_reward[::-1]


class Seq2SeqBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, sources, status=list(), glove=None, generate=False, supervised=False):
        self.sources = sources
        self.status = status
        self.tokenizer = tokenizer
        self.glove = glove
        self.generate = generate
        self.supervised = supervised

    def filter(self, games):

        if len(self.status) > 0:
            games = [g for g in games if g.status in self.status]

        return games

    def split(self, games):

        games = batchifier_split_helper(games, split_mode=BatchifierSplitMode.DialogueHistory)

        if self.supervised:
            games_with_end_of_dialogue = []
            end_of_dialogue = self.tokenizer.decode([self.tokenizer.stop_dialogue])
            for game in games:

                if game.is_full_dialogue:

                    # Update status of dialogue
                    game.is_full_dialogue = False
                    games_with_end_of_dialogue.append(game)

                    # Add an extra question with end_of_dialogue_token
                    game_with_eod = copy.copy(game)  # Beware shallow copy!
                    game_with_eod.is_full_dialogue = True
                    game_with_eod.questions = game.questions + [end_of_dialogue]
                    game_with_eod.question_ids = game.question_ids + [max(game.question_ids) + 1]
                    game_with_eod.answers = game.answers + ["n/a"]  # Dummy token
                    games_with_end_of_dialogue.append(game_with_eod)

                else:
                    games_with_end_of_dialogue.append(game)

            games = games_with_end_of_dialogue

        return games

    def apply(self, games, skip_targets=False):

        batch = collections.defaultdict(list)
        batch["raw"] = games

        batch_size = len(games)

        for i, game in enumerate(games):

            # Encode question answers
            q_tokens = [self.tokenizer.encode(q, add_stop_token=True) for q in game.questions]
            a_tokens = [self.tokenizer.encode(a, is_answer=True) for a in game.answers]

            if self.generate:  # Add a dummy question at eval time to not ignore the last question
                q_tokens.append([])
                a_tokens.append([])

            # Flatten questions/answers except the last one
            dialogue = [self.tokenizer.start_token]  # Add start token (to avoid empty dialogue at the beginning)
            for q_tok, a_tok in zip(q_tokens[:-1], a_tokens[:-1]):
                dialogue += q_tok
                dialogue += a_tok

            # Extract question to predict
            question = [self.tokenizer.start_token] + q_tokens[-1]

            batch["dialogue"].append(dialogue)
            batch["question"].append(question)

            # image
            if 'image' in self.sources:
                img = game.image.get_image()
                if "image" not in batch:  # initialize an empty array for better memory consumption
                    batch["image"] = np.zeros((batch_size,) + img.shape)
                batch["image"][i] = img

            # reward
            if "cum_reward" in self.sources and not skip_targets and not self.generate and not self.supervised:
                # full_game = game.user_data["full_game"]
                # total_number_question = len(full_game.question_ids) - int(game.user_data["has_stop_token"])
                # number_question_left = total_number_question - len(game.question_ids)
                #  - number_question_left * 0.1
                reward = int(game.status == "success")
                batch["cum_reward"].append([reward] * len(batch["question"][i]))

        # Pad dialogue tokens tokens
        batch['dialogue'], batch['seq_length_dialogue'] = padder(batch['dialogue'], padding_symbol=self.tokenizer.padding_token)
        batch[' question'], batch['seq_length_question'] = padder(batch['question'], padding_symbol=self.tokenizer.padding_token)

        if 'cum_reward' in batch:
            batch['cum_reward'], _ = padder(batch['cum_reward'], padding_symbol=0)

        return batch
