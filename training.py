import logging

from collections import Counter

from theano import tensor
from toolz import merge

from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta, CompositeRule)
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.select import Selector

from blocks.monitoring import aggregation
from checkpoint import CheckpointNMT, LoadNMT
from model import BidirectionalEncoder, Decoder
from sampling import BleuValidator, Sampler, CostCurve

logger = logging.getLogger(__name__)


def main(config, tr_stream, dev_stream):
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
    target_bos_idx = tr_stream.trg_bos
    target_space_idx = tr_stream.space_idx['target']

    # Construct model
    logger.info('Building RNN encoder-decoder')

    encoder = BidirectionalEncoder(config['src_vocab_size'], config['enc_embed'],
                                   config['char_enc_nhids'], config['enc_nhids'])

    decoder = Decoder(
        config['trg_vocab_size'], config['dec_embed'], config['char_dec_nhids'], config['dec_nhids'],
        config['enc_nhids'] * 2, target_space_idx, target_bos_idx)

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
    encoder.decimator.dgru.weights_init = Orthogonal()
    encoder.bidir.prototype.weights_init = Orthogonal()
    decoder.interpolator.igru.weights_init = Orthogonal()
    decoder.interpolator.feedback_brick.dgru.weights_init = Orthogonal()
    decoder.transition.weights_init = Orthogonal()
    encoder.initialize()
    decoder.initialize()

    # apply dropout for regularization
    if config['dropout'] < 1.0:
        # dropout is applied to the output of maxout in ghog
        logger.info('Applying dropout')
        dropout_inputs = [x for x in cg.intermediary_variables
                          if x.name == 'maxout_apply_output']
        cg = apply_dropout(cg, dropout_inputs, config['dropout'])

    # Apply weight noise for regularization
    if config['weight_noise_ff'] > 0.0:
        logger.info('Applying weight noise to ff layers')
        enc_params = Selector(encoder.lookup).get_params().values()
        enc_params += Selector(encoder.fwd_fork).get_params().values()
        enc_params += Selector(encoder.back_fork).get_params().values()
        dec_params = Selector(
            decoder.sequence_generator.readout).get_params().values()
        dec_params += Selector(
            decoder.sequence_generator.fork).get_params().values()
        dec_params += Selector(decoder.state_init).get_params().values()
        cg = apply_noise(cg, enc_params + dec_params, config['weight_noise_ff'])

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
        Printing(after_batch=True),
        FinishAfter(after_n_batches=config['finish_after']),
        CheckpointNMT(config['saveto'], every_n_batches=config['save_freq'])]

    # Set up beam search and sampling computation graphs if necessary
    if config['hook_samples'] >= 1 or config['bleu_script'] is not None:
        logger.info("Building sampling model")
        generated = decoder.generate(representation, source_word_mask)
        search_model = Model(generated)
        _, samples = VariableFilter(
            bricks=[decoder.sequence_generator], name="outputs")(
            ComputationGraph(generated[1]))  # generated[1] is next_outputs

    # Add sampling
    if config['hook_samples'] >= 1:
        logger.info("Building sampler")
        extensions.append(
            Sampler(model=search_model, data_stream=tr_stream,
                    hook_samples=config['hook_samples'],
                    every_n_batches=config['sampling_freq'],
                    src_vocab_size=config['src_vocab_size']))

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
