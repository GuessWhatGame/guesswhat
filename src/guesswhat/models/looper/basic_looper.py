from tqdm import tqdm
import numpy as np


from guesswhat.models.looper.tools import clear_after_stop_dialogue, list_to_padded_tokens


class BasicLooper(object):
    def __init__(self, config, oracle_wrapper, qgen_wrapper, guesser_wrapper, tokenizer, batch_size):
        self.storage = []

        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.max_no_question = config['loop']['max_question']
        self.max_depth = config['loop']['max_depth']

        self.oracle = oracle_wrapper
        self.guesser = guesser_wrapper
        self.qgen = qgen_wrapper

    def process(self, sess, iterator, mode, optimizer=list(), store_games=False):

        # initialize the wrapper
        self.qgen.initialize(sess)
        self.oracle.initialize(sess)
        self.guesser.initialize(sess)

        games = []

        score, total_elem = 0, 0
        for game_data in tqdm(iterator):

            ongoing_games = game_data["raw"]

            # Step 1: generate question/answer

            for no_question in range(self.max_no_question):

                # Step 1.1: Generate new question
                ongoing_games = self.qgen.sample_next_question(sess, ongoing_games, extra_data=game_data, mode=mode)

                # Step 1.2: Answer the question
                ongoing_games = self.oracle.answer_question(sess, ongoing_games, extra_data=game_data)

                # Step 1.3 Check if all dialogues are finished
                if all([g.is_full_dialogue for g in ongoing_games]):
                    break

            # Step 2 : Find the object
            ongoing_games = self.guesser.find_object(sess, ongoing_games, extra_data=game_data)
            games.extend(ongoing_games)

            # Step 3 : Apply gradient
            if len(optimizer) > 0:
                self.apply_policy_gradient(sess, ongoing_games, game_data=game_data, optimizer=optimizer)

            # Step 4 : Compute score
            score += sum([g.status == "success" for g in ongoing_games])

        score = 1.0 * score / iterator.n_examples

        return score, games


    def apply_policy_gradient(self, sess, games, game_data, optimizer):

        # Compute cumulative reward TODO: move into an external function

        cum_rewards = np.zeros_like(padded_dialogue, dtype=np.float32)
        for i, (end_of_dialogue, r) in enumerate(zip(seq_length, final_reward)):
            cum_rewards[i, :(end_of_dialogue - 1)] = r  # gamma = 1

        # Create answer mask to ignore the reward for yes/no tokens
        answer_mask = np.ones_like(padded_dialogue)  # quick and dirty mask -> TODO to improve
        answer_mask[padded_dialogue == self.tokenizer.yes_token] = 0
        answer_mask[padded_dialogue == self.tokenizer.no_token] = 0
        answer_mask[padded_dialogue == self.tokenizer.non_applicable_token] = 0

        # Create padding mask to ignore the reward after <stop_dialogue>
        padding_mask = np.ones_like(padded_dialogue)
        padding_mask[padded_dialogue == self.tokenizer.padding_token] = 0
        # for i in range(np.max(seq_length)): print(cum_rewards[0][i], answer_mask[0][i],self.tokenizer.decode([padded_dialogue[0][i]]))

        # Step 4.4: optim step
        qgen = self.qgen.qgen  # retrieve qgen from wrapper (dirty)

        sess.run(optimizer,
                 feed_dict={
                     qgen.images: game_data["images"],
                     qgen.dialogues: padded_dialogue,
                     qgen.seq_length: seq_length,
                     qgen.padding_mask: padding_mask,
                     qgen.answer_mask: answer_mask,
                     qgen.cum_rewards: cum_rewards,
                 })
