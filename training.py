import logging
import sys

from collections import Counter

from theano import tensor
from toolz import merge

from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta, CompositeRule)
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.select import Selector

from blocks.monitoring import aggregation
from checkpoint import CheckpointNMT, LoadNMT
from model import BidirectionalEncoder, Decoder
from sampling import BleuValidator, Sampler, CostCurve

import pprint

import configurations
from stream import get_tr_stream, get_dev_stream

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(config, tr_stream, dev_stream):
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

    # Construct model
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

    logger.info('Creating computational graph')
    cg = ComputationGraph(cost)

    # Initialize model
    logger.info('Initializing model')
    encoder.weights_init = decoder.weights_init = IsotropicGaussian(
        config['weight_scale'])
    encoder.biases_init = decoder.biases_init = Constant(0)
    encoder.push_initialization_config()
    decoder.push_initialization_config()
    for layer_n in range(config['src_dgru_depth']):
        encoder.decimator.dgru.transitions[layer_n].weights_init = Orthogonal()
    for layer_n in range(config['bidir_encoder_depth']):
        encoder.children[1 + layer_n].prototype.recurrent.weights_init = Orthogonal()
    if config['trg_igru_depth'] == 1:
        decoder.interpolator.igru.weights_init = Orthogonal()
    else:
        for layer_n in range(config['trg_igru_depth']):
            decoder.interpolator.igru.transitions[layer_n].weights_init = Orthogonal()
    for layer_n in range(config['trg_dgru_depth']):
        decoder.interpolator.feedback_brick.dgru.transitions[layer_n].weights_init = Orthogonal()
    for layer_n in range(config['transition_depth']):
        decoder.transition.transitions[layer_n].weights_init = Orthogonal()
    encoder.initialize()
    decoder.initialize()


    # Print shapes
    shapes = [param.get_value().shape for param in cg.parameters]
    logger.info("Parameter shapes: ")
    for shape, count in Counter(shapes).most_common():
        logger.info('    {:15}: {}'.format(str(shape), count))
    logger.info("Total number of parameters: {}".format(len(shapes)))

    # Print parameter names
    enc_dec_param_dict = merge(Selector(encoder).get_parameters(),
                               Selector(decoder).get_parameters())
    logger.info("Parameter names: ")
    for name, value in enc_dec_param_dict.items():
        logger.info('    {:15}: {}'.format(str(value.get_value().shape), name))
    logger.info("Total number of parameters: {}"
                .format(len(enc_dec_param_dict)))

    # Set up training model
    logger.info("Building model")
    training_model = Model(cost)
    # Set up training algorithm
    logger.info("Initializing training algorithm")
    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                 eval(config['step_rule'])()])
    )

    # Set extensions
    logger.info("Initializing extensions")
    # Extensions
    gradient_norm = aggregation.mean(algorithm.total_gradient_norm)
    step_norm = aggregation.mean(algorithm.total_step_norm)
    train_monitor = CostCurve([cost, gradient_norm, step_norm], config=config, after_batch=True,
                              before_first_epoch=True, prefix='tra')
    extensions = [
        train_monitor, Timing(),
        Printing(every_n_batches=config['print_freq']),
        FinishAfter(after_n_batches=config['finish_after']),
        CheckpointNMT(config['saveto'], every_n_batches=config['save_freq'])]

    # Set up beam search and sampling computation graphs if necessary
    if config['hook_samples'] >= 1 or config['bleu_script'] is not None:
        logger.info("Building sampling model")
        generated = decoder.generate(representation, source_word_mask)
        search_model = Model(generated)
        _, samples = VariableFilter(
            bricks=[decoder.sequence_generator], name="outputs")(
            ComputationGraph(generated[config['transition_depth']]))  # generated[transition_depth] is next_outputs

    # Add sampling
    if config['hook_samples'] >= 1:
        logger.info("Building sampler")
        extensions.append(
            Sampler(model=search_model, data_stream=tr_stream,
                    hook_samples=config['hook_samples'], transition_depth=config['transition_depth'],
                    every_n_batches=config['sampling_freq'], src_vocab_size=config['src_vocab_size']))

    # Add early stopping based on bleu
    if config['bleu_script'] is not None:
        logger.info("Building bleu validator")
        extensions.append(
            BleuValidator(source_char_seq, source_sample_matrix, source_char_aux,
                          source_word_mask, samples=samples, config=config,
                          model=search_model, data_stream=dev_stream,
                          normalize=config['normalized_bleu'],
                          every_n_batches=config['bleu_val_freq']))

    # Reload model if necessary
    if config['reload']:
        extensions.append(LoadNMT(config['saveto']))

    # Initialize main loop
    logger.info("Initializing main loop")
    main_loop = MainLoop(
        model=training_model,
        algorithm=algorithm,
        data_stream=tr_stream,
        extensions=extensions
    )

    # Train!
    main_loop.run()


if __name__ == "__main__":
    assert sys.version_info >= (3,4)
    # Get configurations for model
    configuration = configurations.get_config()
    logger.info("Model options:\n{}".format(pprint.pformat(configuration)))
    # Get data streams and call main
    main(configuration, get_tr_stream(**configuration),
         get_dev_stream(**configuration))
