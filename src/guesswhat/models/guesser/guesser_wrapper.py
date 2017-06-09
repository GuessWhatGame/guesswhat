
from generic.tf_utils.evaluator import Evaluator

class GuesserWrapper(object):
    def __init__(self, guesser):

        self.guesser = guesser
        self.evaluator = None


    def initialize(self, sess):
        self.evaluator = Evaluator(self.guesser.get_sources(sess), self.guesser.scope_name)


    def find_object(self, sess, dialogues, seq_length, game_data):

        game_data["dialogues"] = dialogues
        game_data["seq_length"] = seq_length

        # sample
        selected_object, softmax = self.evaluator.execute(sess, output=[self.guesser.selected_object, self.guesser.softmax], batch=game_data)

        found = (selected_object == game_data["targets_index"])

        return found, softmax, selected_object








