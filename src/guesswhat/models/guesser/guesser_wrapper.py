import numpy as np

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


class GuesserUserWrapper(object):

    def __init__(self, tokenizer, img_raw_dir=None):
        self.tokenizer = tokenizer
        self.img_raw_dir = img_raw_dir

    def initialize(self, sess):
        pass

    def find_object(self, _, dialogues, __, game_data):

        # Step 1 : Display dialogue and objects
        print()
        print("Final dialogue:")
        qas = self.tokenizer.split_questions(dialogues[0])
        for qa in qas:
            print(" -",  self.tokenizer.decode(qa))

        print()
        print("Select one of the following objects")
        game = game_data["raw"][0]
        objects = game.objects
        for i, obj in enumerate(objects):
            print(" -", i, obj.category, "\t", obj.bbox)

        # Step 2 : Ask for guess
        while True:
            selected_object = input('What is your guess id? (S)how image. -->  ')

            if selected_object == "S" or selected_object.lower() == "show":
                game.show(self.img_raw_dir, display_index=True)

            elif 0 <= int(selected_object) < len(objects):
                break

        # Step 3 : Check guess
        found = (selected_object == game_data["targets_index"])
        softmax = np.zeros(len(objects))
        softmax[selected_object] = 1

        if found:
            print("Success!")
        else:
            print("Failure :(")
            print("The correct object was: {}".format(game_data["targets_index"][0]))

        print()

        return [found], [softmax], [selected_object]
