"""Create a dictionary file from specified GuessWhat dataset

example
-------
python src/guesswhat/preprocess_data/create_dictionary.py -data_dir=/path/to/guesswhat
"""
import argparse
import collections
import gzip
import io
import json
import os

from nltk.tokenize import TweetTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Creating dictionary..')

    parser.add_argument("-data_dir", type=str, help="Path where are the Guesswhat dataset")
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Name of the dictionary file")
    parser.add_argument("-min_occ", type=int, default=3,
                        help='Minimum number of occurences to add word to dictionary')

    args = parser.parse_args()

    # Set default values
    word2i = {'<padding>': 0,
              '<start>': 1,
              '<stop>': 2,
              '<stop_dialogue>': 3,
              '<unk>': 4,
              '<yes>' : 5,
              '<no>': 6,
              '<n/a>': 7,
              }

    word2occ = collections.defaultdict(int)

    tknzr = TweetTokenizer(preserve_case=False)


    print("Processing train dataset...")
    path = os.path.join(args.data_dir, "guesswhat.train.jsonl.gz")
    with gzip.open(path) as f:
        for k , line in enumerate(f):
            dialogue = json.loads(line)

            for qa in dialogue['qas']:
                tokens = tknzr.tokenize(qa['question'])
                for tok in tokens:
                    word2occ[tok] += 1

    print("filter words...")
    for word, occ in word2occ.items():
        if occ >= args.min_occ and word.count('.') <= 1:
            word2i[word] = len(word2i)

    print("Number of words (occ >= 1): {}".format(len(word2occ)))
    print("Number of words (occ >= {}): {}".format(args.min_occ, len(word2i)))

    dict_path = os.path.join(args.data_dir, 'dict.json')
    with io.open(dict_path, 'wb') as f_out:
        data = json.dumps({'word2i': word2i}, ensure_ascii=False)
        f_out.write(data.encode('utf8', 'replace'))

    print("Done!")
