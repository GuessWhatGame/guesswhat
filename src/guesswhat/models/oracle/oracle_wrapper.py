
from generic.tf_utils.evaluator import Evaluator

class OracleWrapper(object):
    def __init__(self, oracle, tokenizer):

        self.oracle = oracle
        self.evaluator = None
        self.tokenizer = tokenizer


    def initialize(self, sess):
        self.evaluator = Evaluator(self.oracle.get_sources(sess), self.oracle.scope_name)


    def answer_question(self, sess, question, seq_length, game_data):

        game_data["question"] = question
        game_data["seq_length"] = seq_length

        # convert dico name to fit oracle constraint
        game_data["category"] = game_data.get("targets_category", None)
        game_data["spatial"] = game_data.get("targets_spatial", None)

        # sample
        answers_indices = self.evaluator.execute(sess, output=self.oracle.best_pred, batch=game_data)

        # Decode the answers token  ['<yes>', '<no>', '<n/a>'] WARNING magic order... TODO move this order into tokenizer
        answer_dico = [self.tokenizer.yes_token, self.tokenizer.no_token, self.tokenizer.non_applicable_token]
        answers = [answer_dico[a] for a in answers_indices]  # turn indices into tokenizer_id

        return answers








