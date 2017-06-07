from nltk.tokenize import TweetTokenizer
import re
import os
import json

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

        self.answers = [self.yes_token, self.no_token, self.non_applicable_token]

    """
    Input: String
    Output: List of tokens
    """
    def apply(self, question, is_answer=False):

        tokens = []
        if is_answer:
            token = '<' + question.lower() + '>'
            tokens.append(self.word2i[token])
        else:
            for token in self.wpt.tokenize(question):
                if token not in self.word2i:
                    token = '<unk>'
                tokens.append(self.word2i[token])

        return tokens

    def decode(self, tokens):
        return ' '.join([self.i2word[tok] for tok in tokens])











#
#  VQA
#

if __name__ == '__main__':
    import io
    import json
    import numpy
    import collections
    from guesswhat.data_provider.vqa_dataset import VQADataset

    min_nr_of_occurrences = 2
    keep_top_answers = 2000

    data_dir = '/home/sequel/fstrub/vqa_data'
    #data_dir = '/data/lisa/data/vqa'
    year = '2014'

    train_dataset = VQADataset(data_dir, year, "train")
    valid_dataset = VQADataset(data_dir, year, "val")

    answer_counters = train_dataset.answer_counter.most_common() + valid_dataset.answer_counter.most_common()
    games = train_dataset.games + valid_dataset.games

    word2i = {'<unk>': 0,
              '<start>': 1,
              '<stop>': 2,
              '<padding>': 3
              }

    answer2i = {'<unk>': 0}

    word2occ = collections.defaultdict(int)
    answer2occ = collections.Counter()


    for k, v in answer_counters:
        answer2occ[k] += v

    selected = sum([v[1] for v in answer2occ.most_common(keep_top_answers)])
    total = sum([v[1] for v in answer2occ.most_common()])
    print(float(selected)/total)

    # Input words
    tknzr = TweetTokenizer(preserve_case=False)

    for game in games:
        input_tokens = tknzr.tokenize(game.question)
        for tok in input_tokens:
            word2occ[tok] += 1


    included_cnt = 0
    excluded_cnt = 0
    for word, occ in word2occ.items():
        if occ >= min_nr_of_occurrences and word.count('.') <= 1:
            included_cnt += occ
            word2i[word] = len(word2i)
        else:
            excluded_cnt += occ


    for i, answer in enumerate(answer2occ.most_common(keep_top_answers)):
        answer2i[answer[0]] = len(answer2i)

    print(len(word2i))
    print(len(answer2i))


    save_file = data_dir + "/dict_vqa_"+ str(year) +"_" + str(keep_top_answers) + "answers_preprocessed.json"
    with io.open(save_file, 'w', encoding='utf8') as f_out:
       data = json.dumps({'word2i': word2i, 'answer2i': answer2i})
       f_out.write(data)
       #f_out.write(unicode(data))

    VQATokenizer(save_file)


#### CLEVR

# if __name__ == '__main__':
#     import io
#     import json
#     import numpy
#     import collections
#     from guesswhat.data_provider.clevr_dataset import CLEVRDataset
#
#     data_dir = '/home/sequel/fstrub/clevr_data'
#
#
#
#
#     dataset = CLEVRDataset(data_dir, which_set="train")
#
#     games = dataset.games
#
#     word2i = {'<padding>': 0,
#               '<unk>': 1
#               }
#
#     answer2i = {'<padding>': 0,
#                 '<unk>': 1
#                 }
#
#     answer2occ = dataset.answer_counter
#     word2occ = collections.defaultdict(int)
#
#
#     # Input words
#     tknzr = TweetTokenizer(preserve_case=False)
#
#     for game in games:
#         input_tokens = tknzr.tokenize(game.question)
#         for tok in input_tokens:
#             word2occ[tok] += 1
#
#
#
#     for word in word2occ.keys():
#         if word not in word2i:
#             word2i[word] = len(word2i)
#
#     for answer in answer2occ.keys():
#         if answer not in answer2i:
#             answer2i[answer] = len(answer2i)
#
#     print(len(word2i))
#     print(len(answer2i))
#
#     save_file = data_dir + "/dict_clevr.json"
#     with io.open(save_file, 'w', encoding='utf8') as f_out:
#        data = json.dumps({'word2i': word2i, 'answer2i': answer2i})
#        f_out.write(data)
#        #f_out.write(unicode(data))
#
#     VQATokenizer(save_file)