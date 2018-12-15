import numpy as np
import re

from generic.tf_utils.evaluator import Evaluator


class QGenWrapper(object):

    def __init__(self, qgen, batchifier, tokenizer, max_length, k_best):

        self.qgen = qgen

        self.batchifier = batchifier
        self.tokenizer = tokenizer

        self.max_length = max_length

        self.ops = dict()
        self.ops["sampling"], _ = qgen.create_sampling_graph(start_token=tokenizer.start_token,
                                                             stop_token=tokenizer.stop_token,
                                                             max_tokens=max_length)

        self.ops["greedy"], _ = qgen.create_greedy_graph(start_token=tokenizer.start_token,
                                                         stop_token=tokenizer.stop_token,
                                                         max_tokens=max_length)

        self.ops["beam"], _ = qgen.create_greedy_graph(start_token=tokenizer.start_token,
                                                       stop_token=tokenizer.stop_token,
                                                       max_tokens=max_length,
                                                       k_best=k_best)

        self.evaluator = None

    def initialize(self, sess):
        self.evaluator = Evaluator(self.qgen.get_sources(sess), self.qgen.scope_name)

    def sample_next_question(self, sess, games, extra_data, mode):

        # Update batchifier sources
        sources = self.qgen.get_sources()
        sources = [s for s in sources if s not in extra_data]
        self.batchifier.sources = sources

        # create the training batch
        batch = self.batchifier.apply(games)
        batch = {**batch, **extra_data}

        # Sample
        tokens = self.evaluator.execute(sess, ouput=self.ops[mode], batch=batch)

        # Update game
        new_games = []
        for game, question_tokens in zip(games, tokens):
            game.questions.append(self.tokenizer.decode(question_tokens))
            game.question_ids.append(len(game.question_ids))

            if self.tokenizer.stop_dialogue in question_tokens:
                game.is_full_dialogue = True

            new_games.append(game)

        return new_games


class QGenUserWrapper(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def initialize(self, sess):
        pass

    def reset(self, batch_size):
        pass

    def sample_next_question(self, _, prev_answers, game_data, **__):

        if prev_answers[0] == self.tokenizer.start_token:
            print("Type the character '(S)top' when you want to guess the object")
        else:
            print("A :", self.tokenizer.decode(prev_answers[0]))

        print()
        while True:
            question = input('Q: ')
            if question != "":
                break

        # Stop the dialogue
        if question == "S" or question == "Stop":
            tokens = [self.tokenizer.stop_dialogue]

        # Stop the question (add stop token)
        else:
            question = re.sub('\?', '', question) # remove question tags if exist
            question +=  " ?"
            tokens = self.tokenizer.apply(question)

        return [tokens], np.array([tokens]), [len(tokens)]