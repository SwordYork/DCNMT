#!/usr/bin/python

import logging
import os
import subprocess
import uuid
import sys

from picklable_itertools.extras import equizip


def merge_parallel(src_filename, trg_filename, merged_filename):
    total = 0
    with open(src_filename, 'r', encoding='utf-8') as left:
        with open(trg_filename, 'r', encoding='utf-8') as right:
            with open(merged_filename, 'w') as final:
                for lline, rline in equizip(left, right):
                    if (lline != '\n') and (rline != '\n'):
                        total += 1
                        final.write(lline[:-1] + ' ||| ' + rline)


def split_parallel(merged_filename, src_filename, trg_filename):
    total = 0
    with open(merged_filename) as combined:
        with open(src_filename, 'w') as left:
            with open(trg_filename, 'w') as right:
                for line in combined:
                    total += 1
                    line = line.split('|||')
                    left.write(line[0].strip() + '\n')
                    right.write(line[1].strip() + '\n')


def shuffle_parallel(src_filename, trg_filename):
    logger.info("Shuffling jointly [{}] and [{}]".format(src_filename,
                                                         trg_filename))
    out_src = src_filename + '.shuf'
    out_trg = trg_filename + '.shuf'
    merged_filename = str(uuid.uuid4())
    shuffled_filename = str(uuid.uuid4())
    if not os.path.exists(out_src) or not os.path.exists(out_trg):
        try:
            merge_parallel(src_filename, trg_filename, merged_filename)
            subprocess.check_call(
                " shuf {} > {} ".format(merged_filename, shuffled_filename),
                shell=True)
            split_parallel(shuffled_filename, out_src, out_trg)
            logger.info(
                "...files shuffled [{}] and [{}]".format(out_src, out_trg))
        except Exception as e:
            logger.error("{}".format(str(e)))
    else:
        logger.info("...files exist [{}] and [{}]".format(out_src, out_trg))
    if os.path.exists(merged_filename):
        os.remove(merged_filename)
    if os.path.exists(shuffled_filename):
        os.remove(shuffled_filename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('shuffle_data')

    if len(sys.argv) != 3:
        sys.exit(-1)
    # Shuffle datasets
    src_file_name = sys.argv[1]
    trg_file_name = sys.argv[2]
    shuffle_parallel(src_file_name, trg_file_name)
