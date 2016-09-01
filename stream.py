import numpy

from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)

import pickle
import configurations


def _ensure_special_tokens(vocab, bos_idx=0, eos_idx=0, unk_idx=1):
    """Ensures special tokens exist in the dictionary."""

    # remove tokens if they exist in some other index
    tokens_to_remove = [k for k, v in vocab.items()
                        if v in [bos_idx, eos_idx, unk_idx]]
    for token in tokens_to_remove:
        vocab.pop(token)
    # put corresponding item
    vocab['<S>'] = bos_idx
    vocab['</S>'] = eos_idx
    vocab['<UNK>'] = unk_idx
    return vocab


def _length(sentence_pair):
    """Assumes target is the last element in the tuple."""
    return len(sentence_pair[-1])


class TextFileWithSEOSS(TextFile):
    """ Add space eos space to end of source """

    def __init__(self, files, dictionary, bos_token='<S>', eos_token='</S>',
                 unk_token='<UNK>', level='word', preprocess=None,
                 encoding=None):
        super(TextFileWithSEOSS, self).__init__(files, dictionary, bos_token, eos_token,
                                                unk_token, level, preprocess, encoding)

    def get_data(self, state=None, request=None):
        if request is not None:
            raise ValueError
        sentence = next(state)
        if self.preprocess is not None:
            sentence = self.preprocess(sentence)
        data = [self.dictionary[self.bos_token]] if self.bos_token else []
        if self.level == 'word':
            data.extend(self._get_from_dictionary(word)
                        for word in sentence.split())
        else:
            data.extend(self._get_from_dictionary(char)
                        for char in sentence.strip())
        if self.eos_token:
            data.append(self.dictionary[' '])
            data.append(self.dictionary[self.eos_token])
            data.append(self.dictionary[' '])
        return (data,)


class PaddingWithEOS(Padding):
    """Padds a stream with given end of sequence idx."""

    def __init__(self, data_stream, space_idx, trg_bos, **kwargs):
        kwargs['data_stream'] = data_stream
        self.space_idx = space_idx
        self.trg_bos = trg_bos
        super(PaddingWithEOS, self).__init__(**kwargs)

    @property
    def sources(self):
        sources = ['source_char_seq', 'source_sample_matrix', 'source_char_aux', 'source_word_mask',
                   'target_char_seq', 'target_sample_matrix', 'target_char_aux', 'target_word_mask',
                   'target_char_mask', 'target_resample_matrix', 'target_prev_char_seq', 'target_prev_char_aux']
        return tuple(sources)

    def transform_batch(self, batch):
        batch_with_masks = []
        for k, (source, source_batch) in enumerate(zip(self.data_stream.sources, batch)):
            if source not in self.mask_sources:
                batch_with_masks.append(source_batch)
                continue
            word_shapes = [0] * len(source_batch)
            shapes = [0] * len(source_batch)
            for i, sample in enumerate(source_batch):
                np_sample = numpy.asarray(sample)
                word_shapes[i] = numpy.count_nonzero(np_sample == self.space_idx[source])
                shapes[i] = np_sample.shape

            lengths = [shape[0] for shape in shapes]
            max_word_len = max(word_shapes)
            add_space_length = []
            for i in range(len(lengths)):
                add_space_length.append(lengths[i] + max_word_len - word_shapes[i])
            max_char_seq_length = max(add_space_length)
            rest_shape = shapes[0][1:]
            if not all([shape[1:] == rest_shape for shape in shapes]):
                raise ValueError("All dimensions except length must be equal")
            dtype = numpy.asarray(source_batch[0]).dtype

            char_seq = numpy.zeros(
                (len(source_batch), max_char_seq_length) + rest_shape, dtype=dtype)

            for i, sample in enumerate(source_batch):
                char_seq[i, :len(sample)] = sample
                char_seq[i, len(sample):add_space_length[i]] = self.space_idx[source]

            sample_matrix = numpy.zeros((len(source_batch), max_word_len, max_char_seq_length),
                                        dtype=self.mask_dtype)
            char_seq_space_index = char_seq == self.space_idx[source]

            for i in range(len(source_batch)):
                sample_matrix[i, range(max_word_len),
                              numpy.where(char_seq_space_index[i])[0] - 1] = 1

            char_aux = numpy.ones((len(source_batch), max_char_seq_length), self.mask_dtype)
            char_aux[char_seq_space_index] = 0

            word_mask = numpy.zeros((len(source_batch), max_word_len), self.mask_dtype)
            for i, ws in enumerate(word_shapes):
                word_mask[i, :ws] = 1

            batch_with_masks.append(char_seq)
            batch_with_masks.append(sample_matrix)
            batch_with_masks.append(char_aux)
            batch_with_masks.append(word_mask)

            # target sequence
            if source == 'target':
                target_char_mask = numpy.zeros((len(source_batch), max_char_seq_length), self.mask_dtype)
                for i, sequence_length in enumerate(lengths):
                    target_char_mask[i, :sequence_length] = 1
                target_prev_char_seq = numpy.roll(char_seq, 1)
                target_prev_char_seq[:, 0] = self.trg_bos
                target_prev_char_aux = numpy.roll(char_aux, 1)
                # start of sequence, must be 0
                target_prev_char_aux[:, 0] = 0
                target_resample_matrix = numpy.zeros((len(source_batch), max_char_seq_length, max_word_len),
                                                     dtype=self.mask_dtype)

                curr_space_idx = numpy.where(char_seq_space_index)
                for i in range(len(source_batch)):
                    pj = 0
                    for cj in range(max_word_len):
                        target_resample_matrix[i, pj:curr_space_idx[1][i * max_word_len + cj] + 1, cj] = 1
                        pj = curr_space_idx[1][i * max_word_len + cj] + 1

                batch_with_masks.append(target_char_mask)
                batch_with_masks.append(target_resample_matrix)
                batch_with_masks.append(target_prev_char_seq)
                batch_with_masks.append(target_prev_char_aux)
        return tuple(batch_with_masks)


class _oov_to_unk(object):
    """Maps out of vocabulary token index to unk token index."""

    def __init__(self, src_vocab_size=120, trg_vocab_size=120,
                 unk_id=1):
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.unk_id = unk_id

    def __call__(self, sentence_pair):
        return ([x if x < self.src_vocab_size else self.unk_id
                 for x in sentence_pair[0]],
                [x if x < self.trg_vocab_size else self.unk_id
                 for x in sentence_pair[1]])


class _too_long(object):
    """Filters sequences longer than given sequence length."""

    def __init__(self, unk_id, space_idx, max_src_seq_char_len, max_src_seq_word_len,
                 max_trg_seq_char_len, max_trg_seq_word_len):
        self.unk_id = unk_id
        self.max_src_seq_char_len = max_src_seq_char_len
        self.max_src_seq_word_len = max_src_seq_word_len
        self.max_trg_seq_char_len = max_trg_seq_char_len
        self.max_trg_seq_word_len = max_trg_seq_word_len
        self.space_idx = space_idx

    def __call__(self, sentence_pair):
        max_unk = 5
        return all(
            [len(sentence_pair[0]) <= self.max_src_seq_char_len and sentence_pair[0].count(self.unk_id) < max_unk and
             sentence_pair[0].count(self.space_idx[0]) < self.max_src_seq_word_len,
             len(sentence_pair[1]) <= self.max_trg_seq_char_len and sentence_pair[1].count(self.unk_id) < max_unk and
             sentence_pair[1].count(self.space_idx[1]) < self.max_trg_seq_word_len])


def get_tr_stream(src_vocab, trg_vocab, src_data, trg_data,
                  src_vocab_size=120, trg_vocab_size=120, unk_id=1, bos_token='<S>', max_src_seq_char_len=300,
                  max_src_seq_word_len=50, max_trg_seq_char_len=300, max_trg_seq_word_len=50,
                  batch_size=80, sort_k_batches=12, **kwargs):
    """Prepares the training data stream."""

    # Load dictionaries and ensure special tokens exist
    src_vocab = _ensure_special_tokens(
        src_vocab if isinstance(src_vocab, dict)
        else pickle.load(open(src_vocab, 'rb')),
        bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)

    trg_vocab = _ensure_special_tokens(
        trg_vocab if isinstance(trg_vocab, dict) else
        pickle.load(open(trg_vocab, 'rb')),
        bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)

    # Get text files from both source and target
    src_dataset = TextFileWithSEOSS([src_data], src_vocab, None, level='character')
    trg_dataset = TextFileWithSEOSS([trg_data], trg_vocab, None, level='character')

    # Merge them to get a source, target pair
    stream = Merge([src_dataset.get_example_stream(),
                    trg_dataset.get_example_stream()],
                   ('source', 'target'))

    # Filter sequences that are too long
    stream = Filter(stream, predicate=_too_long(unk_id, [src_vocab[' '], trg_vocab[' ']],
                                                max_src_seq_char_len, max_src_seq_word_len,
                                                max_trg_seq_char_len, max_trg_seq_word_len))

    # Replace out of vocabulary tokens with unk token
    stream = Mapping(stream,
                     _oov_to_unk(src_vocab_size=src_vocab_size,
                                 trg_vocab_size=trg_vocab_size,
                                 unk_id=unk_id))

    # Build a batched version of stream to read k batches ahead
    stream = Batch(stream,
                   iteration_scheme=ConstantScheme(
                       batch_size * sort_k_batches))

    # Sort all samples in the read-ahead batch
    stream = Mapping(stream, SortMapping(_length))

    # Convert it into a stream again
    stream = Unpack(stream)
    # Construct batches from the stream with specified batch size
    stream = Batch(
        stream, iteration_scheme=ConstantScheme(batch_size))

    # Pad sequences that are short
    masked_stream = PaddingWithEOS(stream, {'source': src_vocab[' '], 'target': trg_vocab[' ']}, trg_vocab[bos_token],
                                   mask_dtype='int8')

    return masked_stream


def get_dev_stream(val_set=None, src_vocab=None, src_vocab_size=120,
                   unk_id=1, **kwargs):
    """Setup development set stream if necessary."""
    dev_stream = None
    if val_set is not None and src_vocab is not None:
        src_vocab = _ensure_special_tokens(
            src_vocab if isinstance(src_vocab, dict) else
            pickle.load(open(src_vocab, 'rb')),
            bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
        dev_src_dataset = TextFileWithSEOSS([val_set], src_vocab, None, level='character')
        dev_stream = DataStream(dev_src_dataset)
    return dev_stream


def get_test_stream(test_set=None, src_vocab=None, trg_vocab=None, src_vocab_size=120, trg_vocab_size=120, unk_id=1,
                    bos_token='<S>', **kwargs):
    """Setup development set stream if necessary."""
    test_stream = None
    if test_set is not None and src_vocab is not None and trg_vocab is not None:
        src_vocab = _ensure_special_tokens(
            src_vocab if isinstance(src_vocab, dict) else
            pickle.load(open(src_vocab, 'rb')),
            bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)

        trg_vocab = _ensure_special_tokens(
            trg_vocab if isinstance(trg_vocab, dict) else
            pickle.load(open(trg_vocab, 'rb')),
            bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)

        test_src_dataset = TextFileWithSEOSS([test_set], src_vocab, None, level='character')
        test_stream = DataStream(test_src_dataset)
        test_stream.space_idx = {'source': src_vocab[' '], 'target': trg_vocab[' ']}
        test_stream.trg_bos = trg_vocab[bos_token]
        test_stream.trg_vocab = trg_vocab
        test_stream.eos_token = '</S>'

    return test_stream


if __name__ == '__main__':
    # test stream
    configuration = configurations.get_config()
    tr = get_tr_stream(**configuration)
    total = 0
    # test
    for s in tr.get_epoch_iterator():
        total += 1
        if total % 10000 == 0:
            print(total)
    print(total)
