#!/usr/bin/env python

import pickle
import logging
import os
import sys 

from collections import Counter


def safe_pickle(obj, filename):
    if os.path.isfile(filename):
        logger.info("Overwriting %s." % filename)
    else:
        logger.info("Saving to %s." % filename)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def create_dictionary(input_file, dictionary_file, vocab_size):
    input_filename = os.path.basename(input_file.name)
    logger.info("Counting words in %s" % input_filename)
    counter = Counter()
    sentence_count = 0
    for line in input_file:
        words = list(line.strip())
        counter.update(words)
        sentence_count += 1
    logger.info("%d unique words in %d sentences with a total of %d words."
                % (len(counter), sentence_count, sum(counter.values())))

    if vocab_size is not None:
        if vocab_size <= 3:
            logger.info('Building a dictionary with all unique words')
            vocab_size = len(counter) + 3
        vocab_count = counter.most_common(vocab_size - 3)
        logger.info("Creating dictionary of %s most common words, covering "
                    "%2.1f%% of the text."
                    % (vocab_size,
                       100.0 * sum([count for word, count in vocab_count]) /
                       sum(counter.values())))
    else:
        logger.info("Creating dictionary of all words")
        vocab_count = counter.most_common()

    vocab = {'UNK': 1, '<S>': 0, '</S>': vocab_size - 1}
    for i, (word, count) in enumerate(vocab_count):
        vocab[word] = i + 2

    print(counter, vocab_count)
    safe_pickle(vocab, dictionary_file)


def create_vocabularies(src_file, trg_file, config):
    src_vocab_name = 'vocab.{}-{}.{}.pkl'.format(
        config['source'], config['target'], config['source'])
    trg_vocab_name = 'vocab.{}-{}.{}.pkl'.format(
        config['source'], config['target'], config['target'])

    logger.info("Creating source vocabulary [{}]".format(src_vocab_name))
    if not os.path.exists(src_vocab_name):
        create_dictionary(open(src_file, 'r', encoding='utf-8'), src_vocab_name, config['src_vocab_size'])
    else:
        logger.info("...file exists [{}]".format(src_vocab_name))

    logger.info("Creating target vocabulary [{}]".format(trg_vocab_name))
    if not os.path.exists(trg_vocab_name):
        create_dictionary(open(trg_file, 'r', encoding='utf-8'), trg_vocab_name, config['trg_vocab_size'])
    else:
        logger.info("...file exists [{}]".format(trg_vocab_name))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('create_vocab')
    if len(sys.argv) != 7:
        sys.exit(-1)

    configs = {'source': sys.argv[1], 'target': sys.argv[2], 'src_vocab_size': int(sys.argv[3]), 'trg_vocab_size': int(sys.argv[4])}
    src_file_name = sys.argv[5]
    trg_file_name = sys.argv[6]
    create_vocabularies(src_file_name, trg_file_name, configs)
