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

        # Retrieve key values
        self.no_words = len(self.word2i)
        self.start_token = self.word2i["<start>"]
        self.stop_token = self.word2i["?"]
        self.stop_dialogue = self.word2i["<stop_dialogue>"]
        self.padding_token = self.word2i["<padding>"]
        self.yes_token = self.word2i["<yes>"]
        self.no_token = self.word2i["<no>"]
        self.non_applicable_token = self.word2i["<n/a>"]

        assert self.padding_token == 0, "Padding token must be equal to zero"

        self.answers = [self.yes_token, self.no_token, self.non_applicable_token]

        # TODO load from dico
        self.oracle_answers_to_idx = {"yes": 0,
                                      "no": 1,
                                      "n/a": 2}

        self.oracle_idx_to_answers = {v: k for k, v in self.oracle_answers_to_idx.items()}

    def encode(self, question, is_answer=False):

        tokens = []
        if is_answer:
            token = self.format_answer(question)
            tokens.append(self.word2i[token])
        else:
            for token in self.wpt.tokenize(question):
                if token not in self.word2i:
                    token = '<unk>'
                tokens.append(self.word2i[token])

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

    def tokenize_question(self, question):
        return self.wpt.tokenize(question)
