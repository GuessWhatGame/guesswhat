from nltk.tokenize import TweetTokenizer

if __name__ == '__main__':
    import gzip
    import io
    import json
    import argparse
    import collections

    parser = argparse.ArgumentParser('Feature extractor! ')

    parser.add_argument("-dataset_path", type=str, help="Input dataset file ()")
    parser.add_argument("-dico_path", type=str, help="Output dico file")
    parser.add_argument("-min_occ", type=int, default=3, help='Minimum Number of occurence of a word to be put in the vocabulary ')

    args = parser.parse_args()

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

    print("Load dataset...")
    with gzip.open(args.dataset_path) as f:
        for k , line in enumerate(f):
            dialogue = json.loads(line)

            for qa in dialogue['qas']:
                tokens = tknzr.tokenize(qa['q'])
                for tok in tokens:
                    word2occ[tok] += 1

    print("filter words...")
    for word, occ in word2occ.items():
        if occ >= args.min_occ and word.count('.') <= 1:
            word2i[word] = len(word2i)

    print("Number of words (occ: 1): {}".format(len(word2occ)))
    print("Number of words (occ: {}): {}".format(args.min_occ, len(word2i)))


    with io.open(args.dico_path, 'w', encoding='utf8') as f_out:
        data = json.dumps({'word2i': word2i}, ensure_ascii=False)
        f_out.write(data.encode('utf8', 'replace'))

    print("Done!")
