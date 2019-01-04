from tqdm import tqdm


class BasicLooper(object):
    def __init__(self, config, oracle_wrapper, qgen_wrapper, guesser_wrapper, tokenizer, batch_size):

        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.max_no_question = config['loop']['max_question']

        self.oracle = oracle_wrapper
        self.guesser = guesser_wrapper
        self.qgen = qgen_wrapper

    def process(self, sess, iterator, mode, optimizer=None, store_games=False):

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
                ongoing_games = self.qgen.sample_next_question(sess, ongoing_games, mode=mode)

                # Step 1.2: Answer the question
                ongoing_games = self.oracle.answer_question(sess, ongoing_games)

                # Step 1.3 Check if all dialogues are finished
                if all([g.user_data["has_stop_token"] for g in ongoing_games]):
                    break

            # Step 2 : Find the object
            ongoing_games = self.guesser.find_object(sess, ongoing_games)
            games.extend(ongoing_games)

            # Step 3 : Apply gradient
            if optimizer is not None:
                self.qgen.policy_update(sess, ongoing_games, optimizer=optimizer)

            # Step 4 : Compute score
            score += sum([g.status == "success" for g in ongoing_games])

            # Free the memory used for optimization -> DO NOT REMOVE! cd bufferize in looper_batchifier
            for game in ongoing_games:
                game.flush()

        score = 1.0 * score / iterator.n_examples

        return score, games
