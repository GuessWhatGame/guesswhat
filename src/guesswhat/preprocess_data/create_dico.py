from nltk.tokenize import TweetTokenizer

if __name__ == '__main__':
    import gzip
    import io
    import json
    import numpy
    import collections


    min_nr_of_occurrences = 3
    out_file = '/data/lisa/data/guesswhat/dict.json'
    dialogues_file = '/data/lisa/data/guesswhat/guesswhat.train.jsonl.gz'

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

    k = 0
    with gzip.open(dialogues_file) as f:
        for k , line in enumerate(f):
            dialogue = json.loads(line)

            for qa in dialogue['qas']:
                tokens = tknzr.tokenize(qa['q'])
                for tok in tokens:
                    word2occ[tok] += 1

    print(k)
    len(word2occ)

    included_cnt = 0
    excluded_cnt = 0
    for word, occ in word2occ.items():

        if occ >= min_nr_of_occurrences and word.count('.') <= 1:
            included_cnt += occ
            word2i[word] = len(word2i)
        else:
            excluded_cnt += occ

    print(included_cnt)
    print(excluded_cnt)
    print(len(word2i))

    with io.open(out_file, 'w', encoding='utf8') as f_out:
        data = json.dumps({'word2i': word2i}, ensure_ascii=False)
        f_out.write(unicode(data))