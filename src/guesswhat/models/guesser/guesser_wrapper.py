import copy

from generic.data_provider.iterator import BasicIterator
from generic.tf_utils.evaluator import Evaluator


class GuesserWrapper(object):

    def __init__(self, guesser, batchifier, tokenizer, listener):
        self.guesser = guesser
        self.batchifier = batchifier
        self.tokenizer = tokenizer
        self.listener = listener
        self.evaluator = None

    def initialize(self, sess):
        self.evaluator = Evaluator(self.guesser.get_sources(sess), self.guesser.scope_name)

    def find_object(self, sess, games):

        # the guesser may need to split the input
        iterator = BasicIterator(games,
                                 batch_size=len(games),
                                 batchifier=self.batchifier)

        # sample
        self.evaluator.process(sess, iterator, outputs=[], listener=self.listener, show_progress=False)
        results = self.listener.results()

        # Update games
        new_games = []
        for game in games:

            res = results[game.dialogue_id]

            game.id_guess_object = res["id_guess_object"]
            game.user_data.get("softmax", []).append(res["softmax"])
            game.status = "success" if res["success"] else "failure"
            game.is_full_dialogue = True

            new_games.append(game)

        return new_games


# TODO: refactor
# class GuesserUserWrapper(object):
#
#     def __init__(self, tokenizer, img_raw_dir=None):
#         self.tokenizer = tokenizer
#         self.img_raw_dir = img_raw_dir
#
#     def initialize(self, sess):
#         pass
#
#     def find_object(self, _, dialogues, __, game_data):
#
#         # Step 1 : Display dialogue and objects
#         print()
#         print("Final dialogue:")
#         qas = self.tokenizer.split_questions(dialogues[0])
#         for qa in qas:
#             print(" -",  self.tokenizer.decode(qa))
#
#         print()
#         print("Select one of the following objects")
#         game = game_data["raw"][0]
#         objects = game.objects
#         for i, obj in enumerate(objects):
#             print(" -", i, obj.category, "\t", obj.bbox)
#
#         # Step 2 : Ask for guess
#         while True:
#             selected_object = input('What is your guess id? (S)how image. -->  ')
#
#             if selected_object == "S" or selected_object.lower() == "show":
#                 game.show(self.img_raw_dir, display_index=True)
#
#             elif 0 <= int(selected_object) < len(objects):
#                 break
#
#         # Step 3 : Check guess
#         found = (selected_object == game_data["target_index"])
#         softmax = np.zeros(len(objects))
#         softmax[selected_object] = 1
#
#         if found:
#             print("Success!")
#         else:
#             print("Failure :(")
#             print("The correct object was: {}".format(game_data["target_index"][0]))
#
#         print()
#
#         return [found], [softmax], [selected_object]
