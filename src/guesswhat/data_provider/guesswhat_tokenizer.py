from nltk.tokenize import TweetTokenizer
import json
import numpy as np


class GWTokenizer:
    """ """
    def __init__(self, dictionary_file):
        with open(dictionary_file, 'r') as f:
            self.word2i = json.load(f)['word2i']
        self.wpt = TweetTokenizer(preserve_case=False)

        if "<stop_dialogue>" not in self.word2i:
            self.word2i["<stop_dialogue>"] = len(self.word2i)

        self.i2word = {}
        for (k, v) in self.word2i.items():
            self.i2word[v] = k

        self.start_word = "<start>"
        self.stop_word = "?"
        self.stop_dialogue_word = "<stop_dialogue>"
        self.padding_word = "<padding>"
        self.yes_word = "<yes>"
        self.no_word = "<no>"
        self.non_applicable_word = "<n/a>"

        # Retrieve key values
        self.no_words = len(self.word2i)
        self.start_token = self.word2i[self.start_word]
        self.stop_token = self.word2i[self.stop_word]
        self.stop_dialogue = self.word2i[self.stop_dialogue_word]
        self.padding_token = self.word2i[self.padding_word]
        self.yes_token = self.word2i[self.yes_word]
        self.no_token = self.word2i[self.no_word]
        self.non_applicable_token = self.word2i[self.non_applicable_word]

        assert self.padding_token == 0, "Padding token must be equal to zero"

        self.answers = [self.yes_token, self.no_token, self.non_applicable_token]

        # TODO load from dico
        self.oracle_answers_to_idx = {"yes": 0,
                                      "no": 1,
                                      "n/a": 2}

        self.oracle_idx_to_answers = {v: k for k, v in self.oracle_answers_to_idx.items()}

    def encode(self, question, is_answer=False, add_stop_token=False):

        tokens = []
        if is_answer:
            token = self.format_answer(question)
            tokens.append(self.word2i[token])
        else:
            for token in self.wpt.tokenize(question):
                if token not in self.word2i:
                    token = '<unk>'
                tokens.append(self.word2i[token])

            if add_stop_token and tokens[-1] != self.stop_token:
                tokens += [self.stop_token]

        return tokens

    @staticmethod
    def format_answer(answer):
        return '<' + answer.lower() + '>'

    def decode(self, tokens):
        return ' '.join([self.i2word[tok] for tok in tokens])

    def split_questions(self, dialogue_tokens):

        qas = []
        qa = []
        for token in dialogue_tokens:

            assert token != self.padding_token, "Unexpected padding token"

            # check for end of dialogues
            if token == self.stop_dialogue:
                break

            if token == self.start_token:
                continue

            qa.append(token)

            # check for end of question
            if token in self.answers:
                qas.append(qa)
                qa = []

        return qas

    def encode_oracle_answer(self, answer, sparse):
        idx = self.oracle_answers_to_idx[answer.lower()]
        if sparse:
            return idx
        else:
            arr = np.zeros(len(self.oracle_answers_to_idx))
            arr[idx] = 1
            return arr

    def decode_oracle_answer(self, token, sparse):
        if sparse:
            return self.oracle_idx_to_answers[token]
        else:
            assert len(token) < len(self.oracle_answers_to_idx), "Invalid size for oracle answer"
            return self.oracle_answers_to_idx[token.argmax()]

    def tokenize_question(self, question, add_stop_token=False):
        if add_stop_token and question[-1] != self.i2word[self.stop_token]:
            question += self.i2word[self.stop_token]
        return self.wpt.tokenize(question)
