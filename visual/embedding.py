import pickle
from theano import tensor
from blocks.main_loop import MainLoop
from blocks.model import Model

from checkpoint import LoadNMT
from model import BidirectionalEncoder, Decoder
from stream import get_tr_stream
import numpy

import argparse
import logging
import pprint

import configurations
import matplotlib.pyplot as plt
from sklearn import manifold

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_input_dict(input_, src_vocab):
    input_length = len(input_)
    input_ = numpy.array([src_vocab[i] for i in input_])

    total_word = list(input_).count(src_vocab[' '])

    source_sample_matrix = numpy.zeros((total_word, input_length), dtype='int8')
    source_sample_matrix[range(total_word), numpy.nonzero(input_ == src_vocab[' '])[0] - 1] = 1

    source_char_aux = numpy.ones(input_length, dtype='int8')
    source_char_aux[input_ == src_vocab[' ']] = 0

    input_dict = {'source_sample_matrix': source_sample_matrix[None, :],
                  'source_char_aux': source_char_aux[None, :],
                  'source_char_seq': input_[None, :]}
    return input_length, input_dict


# Scale and visualize the embedding vectors
def plot_embedding(X, Y, title=None):
    x_min, x_max = numpy.min(X, 0), numpy.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    for i in range(X.shape[0]):
        if str(Y[i]) in ['Sunday ', 'March ', 'June ', 'January ', 'February ',
                         'April ', 'May ', 'July ', 'August ', 'September ',
                         'October ', 'November ', 'December ']:
            plt.text(X[i, 0], X[i, 1], str(Y[i]),
                     fontdict={'weight': 'bold', 'size': 18, 'color': 'blue'})
        elif str(Y[i]) != 'exercise ' and str(Y[i]) != 'exrecise ':
            plt.text(X[i, 0], X[i, 1], str(Y[i]),
                     fontdict={'size': 18})
        else:
            plt.text(X[i, 0], X[i, 1], str(Y[i]),
                     fontdict={'weight': 'bold', 'size': 18, 'color': 'red'})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def embedding(embedding_model, src_vocab):
    sampling_fn = embedding_model.get_theano_function()
    try:
        f = open('wordlist')
    except FileNotFoundError:
        print('Please create a file named wordlist, and one word per line in this file')
        exit(0)
    s = f.read()
    core_list = s.strip().split('\n')
    f.close()

    X = []
    Y = []
    for word in core_list:
        word += ' '
        _, input_dict = build_input_dict(word, src_vocab)
        w_v = sampling_fn(**input_dict)[0][0][0]
        X.append(w_v)
        Y.append(word)
    X = numpy.array(X)
    print(X.shape)

    # t-SNE embedding of the digits dataset
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)

    plot_embedding(X_tsne, Y, "t-SNE embedding of the words")
    plt.show()


def main(config, tr_stream):
    # Create Theano variables
    logger.info('Creating theano variables')
    source_char_seq = tensor.lmatrix('source_char_seq')
    source_sample_matrix = tensor.btensor3('source_sample_matrix')
    source_char_aux = tensor.bmatrix('source_char_aux')
    source_word_mask = tensor.bmatrix('source_word_mask')
    target_char_seq = tensor.lmatrix('target_char_seq')
    target_char_aux = tensor.bmatrix('target_char_aux')
    target_char_mask = tensor.bmatrix('target_char_mask')
    target_sample_matrix = tensor.btensor3('target_sample_matrix')
    target_word_mask = tensor.bmatrix('target_word_mask')
    target_resample_matrix = tensor.btensor3('target_resample_matrix')
    target_prev_char_seq = tensor.lmatrix('target_prev_char_seq')
    target_prev_char_aux = tensor.bmatrix('target_prev_char_aux')
    target_bos_idx = tr_stream.trg_bos
    target_space_idx = tr_stream.space_idx['target']
    src_vocab = pickle.load(open(config['src_vocab'], 'rb'))

    logger.info('Building RNN encoder-decoder')
    encoder = BidirectionalEncoder(config['src_vocab_size'], config['enc_embed'], config['src_dgru_nhids'],
                                   config['enc_nhids'], config['src_dgru_depth'], config['bidir_encoder_depth'])

    decoder = Decoder(config['trg_vocab_size'], config['dec_embed'], config['trg_dgru_nhids'], config['trg_igru_nhids'],
                      config['dec_nhids'], config['enc_nhids'] * 2, config['transition_depth'], config['trg_igru_depth'],
                      config['trg_dgru_depth'], target_space_idx, target_bos_idx)

    representation = encoder.apply(source_char_seq, source_sample_matrix, source_char_aux,
                                   source_word_mask)
    cost = decoder.cost(representation, source_word_mask, target_char_seq, target_sample_matrix,
                        target_resample_matrix, target_char_aux, target_char_mask,
                        target_word_mask, target_prev_char_seq, target_prev_char_aux)

    # Set up model
    logger.info("Building model")
    training_model = Model(cost)

    # Set extensions
    logger.info("Initializing extensions")
    # Reload model if necessary
    extensions = [LoadNMT(config['saveto'])]

    # Initialize main loop
    logger.info("Initializing main loop")
    main_loop = MainLoop(
        model=training_model,
        algorithm=None,
        data_stream=None,
        extensions=extensions
    )

    for extension in main_loop.extensions:
        extension.main_loop = main_loop
    main_loop._run_extensions('before_training')

    char_embedding = encoder.decimator.apply(source_char_seq.T, source_sample_matrix, source_char_aux.T)
    embedding(Model(char_embedding), src_vocab)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Get the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--proto", default="get_config_en2fr",
                    help="Prototype config to use for config")
args = parser.parse_args()

if __name__ == '__main__':
    # Get configurations for model
    configuration = getattr(configurations, args.proto)()
    logger.info("Model options:\n{}".format(pprint.pformat(configuration)))
    # Get data streams and call main
    main(configuration, get_tr_stream(**configuration))
