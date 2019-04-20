from generic.tf_utils.abstract_listener import EvaluatorListener
import collections
import numpy as np
import math


class OracleListener(EvaluatorListener):
    def __init__(self, tokenizer, require):
        super(OracleListener, self).__init__(require)
        self._results = None
        self._tokenizer = tokenizer
        self.reset()

    def after_batch(self, result, batch, is_training):
        for predicted_answer, game in zip(result, batch['raw']):
            qas = {
                "id": game.question_ids[-1],
                "question": game.questions[-1],
                "answer":  game.answers[-1],
                "oracle_answer": self._tokenizer.decode_oracle_answer(predicted_answer, sparse=True),
                "success": predicted_answer == game.answers[-1]
            }

            self._results[game.dialogue_id].append(qas)

    def reset(self):
        self._results = collections.defaultdict(list)

    def before_epoch(self, is_training):
        self.reset()

    def after_epoch(self, is_training):
        for k, v in self._results.items():
            # assume that the question are sorted according their id
            self._results[k] = sorted(v, key = lambda x: x["id"])

    def get_answers(self):
        return self._results


class QGenListener(EvaluatorListener):
    def __init__(self, require):
        super(QGenListener, self).__init__(require)
        self._results = None
        self.reset()
        self._first_batch = True

    def after_batch(self, result, batch, is_training):

        if not self._first_batch:
            return
        else:
            self._first_batch = False

        self._results = result

    def reset(self):
        self._results = collections.defaultdict(list)
        self._first_batch = True

    def before_epoch(self, is_training):
        self.reset()

    def get_questions(self):
        return self._results


class CropAccuracyListener(EvaluatorListener):
    def __init__(self, require):
        super(CropAccuracyListener, self).__init__(require)
        self.scores = None
        self.reset()

    def after_batch(self, result, batch, is_training):

        for i, (softmax, game) in enumerate(zip(result, batch["raw"])):
            self.scores[game.dialogue_id][game.object.id] = [softmax[1], game.user_data["is_target_object"], game.object.id]

    def reset(self):
        self.scores = collections.defaultdict(dict)

    def before_epoch(self, is_training):
        self.reset()

    def results(self):

        res = dict()

        for game_id, objects in self.scores.items():

            # Compute softmax
            norm = sum(math.exp(obj[0]) for obj in objects.values())
            softmax = {}
            for object_id, obj in objects.items():
                softmax[object_id] = math.exp(obj[0]) / norm

            # retrieve success/failure
            select_object = max(objects.values(), key=lambda v: v[0])

            res[game_id] = dict(
                success=select_object[1],
                id_guess_object=select_object[2],
                softmax=softmax
            )

        return res

    def accuracy(self):

        accuracy = 0.
        for game in self.scores.values():
            select_object = max(game.values(), key=lambda v: v[0])
            if select_object[1]:
                accuracy += 1.
        accuracy /= len(self.scores)

        return accuracy


class AccuracyListener(EvaluatorListener):
    def __init__(self, require):
        super(AccuracyListener, self).__init__(require)
        self._results = None
        self.reset()

    def after_batch(self, result, batch, is_training):
        for i, (softmax, game, target_index) in enumerate(zip(result, batch["raw"], batch['target_index'])):

            predicted_index = np.argmax(softmax)

            self._results[game.dialogue_id] = dict(
                success=(target_index == predicted_index),
                id_guess_object=game.objects[predicted_index].id,
                softmax={game.objects[i].id: prob for i, prob in enumerate(softmax[:len(game.objects)])}
            )

    def reset(self):
        self._results = {}

    def before_epoch(self, is_training):
        self.reset()

    def results(self):
        return self._results

    def accuracy(self):
        accuracy = [int(res["success"]) for res in self._results.values()]
        return 1.0 * sum(accuracy) / len(accuracy)
