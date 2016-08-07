from theano import tensor

from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model

from checkpoint import LoadNMT
from model import BidirectionalEncoder, Decoder
from sampling import BleuTester
import argparse
import logging
import pprint

import configurations
from stream import get_test_stream

logger = logging.getLogger(__name__)


def main(config, test_stream):
    # Create Theano variables
    logger.info('Creating theano variables')
    source_char_seq = tensor.lmatrix('source_char_seq')
    source_sample_matrix = tensor.tensor3('source_sample_matrix')
    source_char_aux = tensor.matrix('source_char_aux')
    source_word_mask = tensor.matrix('source_word_mask')
    target_char_seq = tensor.lmatrix('target_char_seq')
    target_char_aux = tensor.matrix('target_char_aux')
    target_char_mask = tensor.matrix('target_char_mask')
    target_sample_matrix = tensor.tensor3('target_sample_matrix')
    target_word_mask = tensor.matrix('target_word_mask')
    target_resample_matrix = tensor.tensor3('target_resample_matrix')
    target_prev_char_seq = tensor.lmatrix('target_prev_char_seq')
    target_prev_char_aux = tensor.matrix('target_prev_char_aux')

    target_bos_idx = test_stream.trg_bos
    target_space_idx = test_stream.space_idx['target']

    # Construct model
    logger.info('Building RNN encoder-decoder')

    encoder = BidirectionalEncoder(config['src_vocab_size'], config['enc_embed'], config['char_enc_nhids'],
                                   config['enc_nhids'], config['encoder_layers'])

    decoder = Decoder(config['trg_vocab_size'], config['dec_embed'], config['char_dec_nhids'], config['dec_nhids'],
                      config['enc_nhids'] * 2, config['transition_layers'], target_space_idx, target_bos_idx)

    representation = encoder.apply(source_char_seq, source_sample_matrix, source_char_aux,
                                   source_word_mask)
    cost = decoder.cost(representation, source_word_mask, target_char_seq, target_sample_matrix,
                        target_resample_matrix, target_char_aux, target_char_mask,
                        target_word_mask, target_prev_char_seq, target_prev_char_aux)

    # Set up training model
    logger.info("Building model")
    training_model = Model(cost)

    # Set extensions
    logger.info("Initializing extensions")
    # Extensions
    extensions = []
    # Reload model if necessary
    if config['reload']:
        extensions.append(LoadNMT(config['saveto']))

    # Set up beam search and sampling computation graphs if necessary
    if config['bleu_script'] is not None:
        logger.info("Building sampling model")
        generated = decoder.generate(representation, source_word_mask)
        search_model = Model(generated)
        _, samples = VariableFilter(
            bricks=[decoder.sequence_generator], name="outputs")(
            ComputationGraph(generated[config['transition_layers']]))  # generated[1] is next_outputs

        logger.info("Building bleu tester")
        extensions.append(
            BleuTester(source_char_seq, source_sample_matrix, source_char_aux,
                       source_word_mask, samples=samples, config=config,
                       model=search_model, data_stream=test_stream,
                       normalize=config['normalized_bleu']))

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
    main(configuration, get_test_stream(**configuration))


