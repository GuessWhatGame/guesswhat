
from generic.tf_utils.evaluator import Evaluator


class OracleWrapper(object):
    def __init__(self, oracle, batchifier, tokenizer):

        self.oracle = oracle
        self.evaluator = None

        self.tokenizer = tokenizer
        self.batchifier = batchifier

    def initialize(self, sess):
        self.evaluator = Evaluator(self.oracle.get_sources(sess), self.oracle.scope_name)

    def answer_question(self, sess, games, extra_data):

        # Update batchifier sources
        sources = self.oracle.get_sources()
        sources = [s for s in sources if s not in extra_data]
        self.batchifier.sources = sources

        # create the training batch
        batch = self.batchifier.apply(games)
        batch = {**batch, **extra_data}

        # Sample
        answers_indices = self.evaluator.execute(sess, ouput=self.oracle.best_pred, batch=batch)
        answers = [self.tokenizer.oracle_idx_to_answers[a] for a in answers_indices]

        # Update game
        new_games = []
        for game, answer in zip(games, answers):
            game.answers.append(self.tokenizer.decode(answer))
            new_games.append(game)

        return answers


class OracleUserWrapper(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def initialize(self, sess):
        pass

    def answer_question(self, sess, question, **_):

        # Discard question if it contains the stop dialogue token
        if self.tokenizer.stop_dialogue in question[0]:
            return [self.tokenizer.non_applicable_token]

        print()
        print("Q :", self.tokenizer.decode(question[0]))

        while True:
            answer = input('A (Yes,No,N/A): ').lower()
            if answer == "y" or answer == "yes":
                token = self.tokenizer.yes_token
                break

            elif answer == "n" or answer == "no":
                token = self.tokenizer.no_token
                break

            elif answer == "na" or answer == "n/a" or answer == "not applicable":
                token = self.tokenizer.non_applicable_token
                break

            else:
                print("Invalid answer...")

        return [token]