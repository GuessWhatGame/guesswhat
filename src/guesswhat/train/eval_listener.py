from generic.tf_utils.abstract_listener import EvaluatorListener
import collections


class OracleListener(EvaluatorListener):
    def __init__(self, tokenizer, require):
        super(OracleListener, self).__init__(require)
        self.results = None
        self.tokenizer = tokenizer
        self.reset()

    def after_batch(self, result, batch, is_training):
        for predicted_answer, game in zip(result, batch['raw']):
            qas = {
                "id" : game.question_ids[-1],
                "question" : game.questions[-1],
                "answer" :  game.answers[-1],
                "oracle_answer": self.tokenizer.decode_oracle_answer(predicted_answer, sparse=True),
                "success" : predicted_answer == game.answers[-1]
            }

            self.results[game.dialogue_id].append(qas)

    def reset(self):
        self.results = collections.defaultdict(list)

    def before_epoch(self, is_training):
        self.reset()


    def after_epoch(self, is_training):
        for k, v in self.results.items():
            # assume that the question are sorted according their id
            self.results[k] = sorted(v, key = lambda x: x["id"])

    def get_answers(self):
        return self.results


import tensorflow as tf

class ProfilerListener(EvaluatorListener):
    def __init__(self, pctx):

        self.pctx = pctx

        builder = tf.profiler.ProfileOptionBuilder
        self.opts = builder(builder.time_and_memory()).order_by('micros').build()

        super(ProfilerListener, self).__init__(require=tf.no_op())

    def before_batch(self, result, batch, is_training):

        self.pctx.trace_next_step()
        # Dump the profile to '/tmp/train_dir' after the step.
        self.pctx.dump_next_step()

    def after_epoch(self, is_training):
        self.pctx.profiler.profile_operations(options=self.opts)





class QGenListener(EvaluatorListener):
    def __init__(self, require):
        super(QGenListener, self).__init__(require)
        self.results = None
        self.reset()
        self.first_batch = True

    def after_batch(self, result, batch, is_training):

        if not self.first_batch:
            return
        else:
            self.first_batch = False

        self.results = result

    def reset(self):
        self.results = collections.defaultdict(list)
        self.first_batch = True

    def before_epoch(self, is_training):
        self.reset()

    def get_questions(self):
        return self.results

class GuesserAccuracyListener(EvaluatorListener):
    def __init__(self, require):
        super(GuesserAccuracyListener, self).__init__(require)
        self.scores = None
        self.reset()

    def after_batch(self, result, batch, is_training):

        for i, (softmax, game) in enumerate(zip(result, batch["raw"])):
            self.scores[game.dialogue_id][game.object_id] = [softmax[1], game.is_full_dialogue]

    def reset(self):
        self.scores = collections.defaultdict(dict)

    def before_epoch(self, is_training):
        self.reset()

    def evaluate(self):

        accuracy = 0.
        for game in self.scores.values():
            select_object = max(game.values(), key=lambda v: v[0])
            if select_object[1]:
                accuracy += 1.
        accuracy /= len(self.scores)

        return accuracy