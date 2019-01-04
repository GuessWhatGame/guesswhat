from generic.tf_utils.evaluator import Evaluator
from generic.data_provider.batchifier import BatchifierSplitMode
import copy


class OracleWrapper(object):
    def __init__(self, oracle, batchifier, tokenizer):

        self.oracle = oracle
        self.evaluator = None

        self.tokenizer = tokenizer
        self.batchifier = batchifier

    def initialize(self, sess):
        self.evaluator = Evaluator(self.oracle.get_sources(sess), self.oracle.scope_name)

    def answer_question(self, sess, games):

        # create the training batch #TODO: hack -> to remove
        oracle_games = []
        if self.batchifier.split_mode == BatchifierSplitMode.SingleQuestion:
            for game in games:
                g = copy.copy(game)
                g.questions = [game.questions[-1]]
                g.question_ids = [game.question_ids[-1]]
                oracle_games.append(g)
        else:
            oracle_games = games

        batch = self.batchifier.apply(oracle_games, skip_targets=True)
        batch["is_training"] = False

        # Sample
        answers_index = self.evaluator.execute(sess, output=self.oracle.prediction, batch=batch)

        # Update game
        new_games = []
        for game, answer in zip(games, answers_index):
            if not game.user_data["has_stop_token"]:  # stop adding answer if dialogue is over
                game.answers.append(self.tokenizer.decode_oracle_answer(answer, sparse=True))
            new_games.append(game)

        return new_games


# TODO: refactor
# class OracleUserWrapper(object):
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
#
#     def initialize(self, sess):
#         pass
#
#     def answer_question(self, sess, question, **_):
#
#         # Discard question if it contains the stop dialogue token
#         if self.tokenizer.stop_dialogue in question[0]:
#             return [self.tokenizer.non_applicable_token]
#
#         print()
#         print("Q :", self.tokenizer.decode(question[0]))
#
#         while True:
#             answer = input('A (Yes,No,N/A): ').lower()
#             if answer == "y" or answer == "yes":
#                 token = self.tokenizer.yes_token
#                 break
#
#             elif answer == "n" or answer == "no":
#                 token = self.tokenizer.no_token
#                 break
#
#             elif answer == "na" or answer == "n/a" or answer == "not applicable":
#                 token = self.tokenizer.non_applicable_token
#                 break
#
#             else:
#                 print("Invalid answer...")
#
#         return [token]