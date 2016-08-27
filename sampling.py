from __future__ import print_function

import logging
import operator
import os
import time
import re
import signal

import numpy
from blocks.extensions import SimpleExtension, TrainingExtension
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.search import BeamSearch
from subprocess import Popen, PIPE
from checkpoint import SaveLoadUtils

logger = logging.getLogger(__name__)


class SamplingBase(object):
    """Utility class for Validator and Sampler."""

    def _get_attr_rec(self, obj, attr):
        return self._get_attr_rec(getattr(obj, attr), attr) \
            if hasattr(obj, attr) else obj

    def _get_true_length(self, seq, vocab):
        try:
            return seq.tolist().index(vocab['</S>']) + 1
        except ValueError:
            return len(seq)

    def _oov_to_unk(self, seq, vocab_size, unk_idx):
        return [x if x < vocab_size else unk_idx for x in seq]

    def _idx_to_word(self, seq, ivocab):
        return "".join([ivocab.get(idx, "<UNK>") for idx in seq])

    def build_input_dict(self, input_, src_vocab):
        input_length = self._get_true_length(input_, src_vocab) + 1
        input_ = input_[:input_length]
        total_word = list(input_).count(src_vocab[' '])

        source_sample_matrix = numpy.zeros((total_word, input_length), dtype='int8')
        source_sample_matrix[range(total_word), numpy.nonzero(input_ == src_vocab[' '])[0] - 1] = 1

        source_word_mask = numpy.ones(total_word, dtype='int8')

        source_char_aux = numpy.ones(input_length, dtype='int8')
        source_char_aux[input_ == src_vocab[' ']] = 0

        input_dict = {'source_sample_matrix': source_sample_matrix[None, :],
                      'source_char_aux': source_char_aux[None, :],
                      'source_char_seq': input_[None, :],
                      'source_word_mask': source_word_mask[None, :]}
        return input_length, input_dict

    def build_input_dict_tile(self, input_, src_vocab, beam_size):
        input_length = self._get_true_length(input_, src_vocab) + 1
        input_ = input_[:input_length]
        total_word = list(input_).count(src_vocab[' '])

        source_sample_matrix = numpy.zeros((total_word, input_length), dtype='int8')
        source_sample_matrix[range(total_word), numpy.nonzero(input_ == src_vocab[' '])[0] - 1] = 1

        source_word_mask = numpy.ones(total_word, dtype='int8')

        source_char_aux = numpy.ones(input_length, dtype='int8')
        source_char_aux[input_ == src_vocab[' ']] = 0

        input_dict = {'source_sample_matrix': numpy.tile(source_sample_matrix, (beam_size, 1, 1)),
                      'source_word_mask': numpy.tile(source_word_mask, (beam_size, 1)),
                      'source_char_aux': numpy.tile(source_char_aux, (beam_size, 1)),
                      'source_char_seq': numpy.tile(input_, (beam_size, 1))}

        return input_dict


class CostCurve(TrainingDataMonitoring):
    """ Record training curve """

    def __init__(self, variables, config, **kwargs):
        super(CostCurve, self).__init__(variables, **kwargs)
        self.cost_curve = []
        self.config = config
        # Create saving directory if it does not exist
        if not os.path.exists(self.config['saveto']):
            os.makedirs(self.config['saveto'])

        if self.config['reload']:
            try:
                self.cost_curve = numpy.load(os.path.join(self.config['saveto'],
                                                          'cost_curve.npz'))
                self.cost_curve = self.cost_curve['cost_curves'].tolist()
                logger.info("Cost Curve Reloaded")
            except:
                logger.info("Cost Curve not Found")

    def do(self, callback_name, *args):
        """Initializes the buffer or commits the values to the log.
        What this method does depends on from what callback it is called
        and with which arguments.  When called within `before_training`, it
        initializes the aggregation buffer and instructs the training
        algorithm what additional computations should be carried at each
        step by adding corresponding updates to it. In most_other cases it
        writes aggregated values of the monitored variables to the log. An
        exception is when an argument `just_aggregate` is given: in this
        cases it updates the values of monitored non-Theano quantities, but
        does not write anything to the log.
        """
        data, args = self.parse_args(callback_name, args)
        if callback_name == 'before_training':
            self.main_loop.algorithm.add_updates(
                self._variables.accumulation_updates)
            self.main_loop.algorithm.add_updates(
                self._required_for_non_variables.accumulation_updates)
            self._variables.initialize_aggregators()
            self._required_for_non_variables.initialize_aggregators()
            self._non_variables.initialize_quantities()
        else:
            # When called first time at any iterations, update
            # monitored non-Theano quantities
            if (self.main_loop.status['iterations_done'] >
                    self._last_time_called):
                self._non_variables.aggregate_quantities(
                    list(self._required_for_non_variables
                         .get_aggregated_values().values()))
                self._required_for_non_variables.initialize_aggregators()
                self._last_time_called = (
                    self.main_loop.status['iterations_done'])
            # If only called to update non-Theano quantities,
            # do just that
            if args == ('just_aggregate',):
                return
            # Otherwise, also save current values of from the accumulators
            curr_iter = self.main_loop.status['iterations_done']
            if curr_iter == 0:
                return

            curr_cost = self._variables.get_aggregated_values()
            curr_cost = curr_cost['decoder_cost_cost'].tolist()
            self.cost_curve.append({curr_iter: curr_cost})

            if curr_iter % 100 == 0:
                numpy.savez(os.path.join(self.config['saveto'], 'cost_curve.npz'),
                            cost_curves=self.cost_curve)

            self.add_records(
                self.main_loop.log,
                self._variables.get_aggregated_values().items())
            self._variables.initialize_aggregators()
            self.add_records(
                self.main_loop.log,
                self._non_variables.get_aggregated_values().items())
            self._non_variables.initialize_quantities()


class Sampler(SimpleExtension, SamplingBase):
    """Random Sampling from model."""

    def __init__(self, model, data_stream, hook_samples=1, transition_depth=1,
                 src_vocab=None, trg_vocab=None, src_ivocab=None,
                 trg_ivocab=None, src_vocab_size=None, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.model = model
        self.hook_samples = hook_samples
        self.data_stream = data_stream
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_ivocab = src_ivocab
        self.transition_depth = transition_depth
        self.trg_ivocab = trg_ivocab
        self.src_vocab_size = src_vocab_size
        self.is_synced = False
        self.sampling_fn = model.get_theano_function()

    def do(self, which_callback, *args):

        # Get dictionaries, this may not be the practical way
        sources = self._get_attr_rec(self.main_loop, 'data_stream')

        # Load vocabularies and invert if necessary
        # WARNING: Source and target indices from data stream
        #  can be different
        if not self.src_vocab:
            self.src_vocab = sources.data_streams[0].dataset.dictionary
        if not self.trg_vocab:
            self.trg_vocab = sources.data_streams[1].dataset.dictionary
        if not self.src_ivocab:
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        if not self.trg_ivocab:
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
        if not self.src_vocab_size:
            self.src_vocab_size = len(self.src_vocab)

        # Randomly select source samples from the current batch
        # WARNING: Source and target indices from data stream
        #  can be different
        batch = args[0]
        batch_size = batch['source_char_seq'].shape[0]
        hook_samples = min(batch_size, self.hook_samples)

        # TODO: this is problematic for boundary conditions, eg. last batch
        sample_idx = numpy.random.choice(
            batch_size, hook_samples, replace=False)
        src_batch = batch['source_char_seq']
        trg_batch = batch['target_char_seq']
        input_ = src_batch[sample_idx, :]
        target_ = trg_batch[sample_idx, :]

        # Sample
        print()
        for i in range(hook_samples):
            input_length, input_dict = self.build_input_dict(input_[i], self.src_vocab)
            target_length = self._get_true_length(target_[i], self.trg_vocab) + 1
            sfn = self.sampling_fn(**input_dict)
            outputs = sfn[self.transition_depth]
            costs = sfn[-1]
            outputs = outputs.flatten()
            costs = costs.flatten()

            sample_length = self._get_true_length(outputs, self.trg_vocab)
            print("Input : ", self._idx_to_word(input_[i][:input_length],
                                                self.src_ivocab))
            print("Target: ", self._idx_to_word(target_[i][:target_length],
                                                self.trg_ivocab))
            print("Sample: ", self._idx_to_word(outputs[:sample_length],
                                                self.trg_ivocab))
            print("Sample cost: ", costs[:sample_length].mean())
            print()


class BleuValidator(SimpleExtension, SamplingBase, SaveLoadUtils):
    # TODO: a lot has been changed in NMT, sync respectively
    """Implements early stopping based on BLEU score."""

    def __init__(self, source_char_seq, source_sample_matrix, source_char_aux,
                 source_word_mask, samples, model, data_stream,
                 config, n_best=1, track_n_models=1,
                 normalize=True, **kwargs):
        # TODO: change config structure
        super(BleuValidator, self).__init__(**kwargs)
        self.source_char_seq = source_char_seq
        self.source_sample_matrix = source_sample_matrix
        self.source_char_aux = source_char_aux
        self.source_word_mask = source_word_mask
        self.samples = samples
        self.model = model
        self.data_stream = data_stream
        self.config = config
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.normalize = normalize
        self.verbose = config.get('val_set_out', None)

        # Helpers
        self.vocab = data_stream.dataset.dictionary
        self.src_ivocab = {v: k for k, v in self.vocab.items()}
        self.unk_sym = data_stream.dataset.unk_token
        self.eos_sym = data_stream.dataset.eos_token
        self.unk_idx = self.vocab[self.unk_sym]
        self.eos_idx = self.vocab[self.eos_sym]
        self.best_models = []
        self.val_bleu_curve = []
        self.beam_search = BeamSearch(samples=samples)
        self.multibleu_cmd = ['perl', self.config['bleu_script'],
                              self.config['val_set_grndtruth'], '<']

        # Create saving directory if it does not exist
        if not os.path.exists(self.config['saveto']):
            os.makedirs(self.config['saveto'])

        if self.config['reload']:
            try:
                bleu_score = numpy.load(os.path.join(self.config['saveto'],
                                                     'val_bleu_scores.npz'))
                self.val_bleu_curve = bleu_score['bleu_scores'].tolist()
                # Track n best previous bleu scores
                for i, bleu in enumerate(
                        sorted([list(v.values())[0] for v in self.val_bleu_curve], reverse=True)):
                    if i < self.track_n_models:
                        self.best_models.append(ModelInfo(bleu, self.config['saveto']))
                logger.info("BleuScores Reloaded")
            except:
                logger.info("BleuScores not Found")

    def do(self, which_callback, *args):

        # Track validation burn in
        if self.main_loop.status['iterations_done'] < \
                self.config['val_burn_in']:
            return

        # Evaluate and save if necessary
        self._save_model(self._evaluate_model(self.main_loop))

    def _evaluate_model(self, main_loop):
        curr_iter = main_loop.status['iterations_done']
        logger.info("Started Validation: ")
        val_start_time = time.time()
        mb_subprocess = Popen(self.multibleu_cmd, stdin=PIPE, stdout=PIPE, universal_newlines=True)
        total_cost = 0.0

        # Get target vocabulary
        sources = self._get_attr_rec(self.main_loop, 'data_stream')
        trg_vocab = sources.data_streams[1].dataset.dictionary
        self.trg_vocab = trg_vocab
        self.trg_ivocab = {v: k for k, v in trg_vocab.items()}
        trg_eos_sym = sources.data_streams[1].dataset.eos_token
        self.trg_eos_idx = trg_vocab[trg_eos_sym]

        if self.verbose:
            ftrans = open(self.config['val_set_out'] + str(curr_iter), 'w')

        for i, line in enumerate(self.data_stream.get_epoch_iterator()):
            """
            Load the sentence, retrieve the sample, write to file
            """

            seq = self._oov_to_unk(
                line[0], self.config['src_vocab_size'], self.unk_idx)
            input_dict = self.build_input_dict_tile(numpy.asarray(seq), self.vocab, self.config['beam_size'])

            # draw sample, checking to ensure we don't get an empty string back
            trans, costs = \
                self.beam_search.search(
                    input_values={self.source_char_seq: input_dict['source_char_seq'],
                                  self.source_sample_matrix: input_dict['source_sample_matrix'],
                                  self.source_word_mask: input_dict['source_word_mask'],
                                  self.source_char_aux: input_dict['source_char_aux']},
                    max_length=3 * len(seq), eol_symbol=self.trg_eos_idx,
                    ignore_first_eol=False)

            # normalize costs according to the sequence lengths
            if self.normalize:
                lengths = numpy.array([len(s) for s in trans])
                costs = costs / lengths

            nbest_idx = numpy.argsort(costs)[:self.n_best]
            for j, best in enumerate(nbest_idx):
                try:
                    total_cost += costs[best]
                    trans_out = trans[best]

                    # convert idx to words
                    try:
                        sample_length = trans_out.index(self.trg_vocab['</S>'])
                    except ValueError:
                        sample_length = len(seq)
                    trans_out = trans_out[:sample_length]
                    trans_out = self._idx_to_word(trans_out, self.trg_ivocab)

                except ValueError:
                    logger.info(
                        "Can NOT find a translation for line: {}".format(i + 1))
                    trans_out = '<UNK>'

                if j == 0:
                    # Write to subprocess and file if it exists
                    print("Line:", i)
                    print("Input : ", self._idx_to_word(line[0], self.src_ivocab))
                    print("Sample: ", trans_out)
                    print("Error:", costs[best])
                    print()

                    print(trans_out, file=mb_subprocess.stdin)
                    if self.verbose:
                        print(trans_out, file=ftrans)

            if i != 0 and i % 100 == 0:
                logger.info(
                    "Translated {} lines of validation set...".format(i))

            mb_subprocess.stdin.flush()

        logger.info("Total cost of the validation: {}".format(total_cost))
        self.data_stream.reset()
        if self.verbose:
            ftrans.close()

        # send end of file, read output.
        mb_subprocess.stdin.close()
        stdout = mb_subprocess.stdout.readline()
        logger.info(stdout)
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        logger.info("Validation Took: {} minutes".format(
            float(time.time() - val_start_time) / 60.))
        assert out_parse is not None

        # extract the score
        bleu_score = float(out_parse.group()[6:])
        self.val_bleu_curve.append({curr_iter: bleu_score})
        logger.info(bleu_score)
        mb_subprocess.terminate()

        return bleu_score

    def _is_valid_to_save(self, bleu_score):
        if not self.best_models or min(self.best_models,
                                       key=operator.attrgetter('bleu_score')).bleu_score < bleu_score:
            return True
        return False

    def _save_model(self, bleu_score):
        numpy.savez(
            os.path.join(self.config['saveto'], 'val_bleu_scores.npz'),
            bleu_scores=self.val_bleu_curve)
        if self._is_valid_to_save(bleu_score):
            model = ModelInfo(bleu_score, self.config['saveto'])

            # Manage n-best model list first
            if len(self.best_models) >= self.track_n_models:
                old_model = self.best_models[0]
                if old_model.path and os.path.isfile(old_model.path):
                    logger.info("Deleting old model %s" % old_model.path)
                    os.remove(old_model.path)
                self.best_models.remove(old_model)

            self.best_models.append(model)
            self.best_models.sort(key=operator.attrgetter('bleu_score'))

            # Save the model here
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            logger.info("Saving new model {}".format(model.path))
            params_to_save = self.main_loop.model.get_parameter_values()
            self.save_parameter_values(params_to_save, model.path)
            signal.signal(signal.SIGINT, s)


class BleuTester(TrainingExtension, SamplingBase):
    # TODO: a lot has been changed in NMT, sync respectively
    """Implements Testing BLEU score."""

    def __init__(self, source_char_seq, source_sample_matrix, source_char_aux,
                 source_word_mask, samples, model, data_stream,
                 config, n_best=1, track_n_models=1,
                 normalize=True, **kwargs):
        # TODO: change config structure
        super(BleuTester, self).__init__(**kwargs)
        self.source_char_seq = source_char_seq
        self.source_sample_matrix = source_sample_matrix
        self.source_char_aux = source_char_aux
        self.source_word_mask = source_word_mask
        self.samples = samples
        self.model = model
        self.data_stream = data_stream
        self.config = config
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.normalize = normalize
        self.verbose = True

        # Helpers
        self.vocab = data_stream.dataset.dictionary
        self.src_ivocab = {v: k for k, v in self.vocab.items()}
        self.unk_sym = data_stream.dataset.unk_token
        self.eos_sym = data_stream.dataset.eos_token
        self.unk_idx = self.vocab[self.unk_sym]
        self.eos_idx = self.vocab[self.eos_sym]
        self.beam_search = BeamSearch(samples=samples)
        self.multibleu_cmd = ['perl', self.config['bleu_script'],
                              self.config['test_set_grndtruth'], '<']

        # Create saving directory if it does not exist
        if not os.path.exists(self.config['saveto']):
            os.makedirs(self.config['saveto'])

    def before_training(self):
        self._evaluate_model()

    def _evaluate_model(self):

        logger.info("Started Test: ")
        test_start_time = time.time()
        mb_subprocess = Popen(self.multibleu_cmd, stdin=PIPE, stdout=PIPE, universal_newlines=True)
        total_cost = 0.0

        # Get target vocabulary
        trg_vocab = self.data_stream.trg_vocab
        self.trg_vocab = trg_vocab
        self.trg_ivocab = {v: k for k, v in trg_vocab.items()}
        trg_eos_sym = self.data_stream.eos_token
        self.trg_eos_idx = trg_vocab[trg_eos_sym]

        if self.verbose:
            ftrans = open(self.config['test_set_out'], 'w')

        for i, line in enumerate(self.data_stream.get_epoch_iterator()):
            """
            Load the sentence, retrieve the sample, write to file
            """

            seq = self._oov_to_unk(
                line[0], self.config['src_vocab_size'], self.unk_idx)
            input_dict = self.build_input_dict_tile(numpy.asarray(seq), self.vocab, self.config['beam_size'])

            # draw sample, checking to ensure we don't get an empty string back
            trans, costs = \
                self.beam_search.search(
                    input_values={self.source_char_seq: input_dict['source_char_seq'],
                                  self.source_sample_matrix: input_dict['source_sample_matrix'],
                                  self.source_word_mask: input_dict['source_word_mask'],
                                  self.source_char_aux: input_dict['source_char_aux']},
                    max_length=3 * len(seq), eol_symbol=self.trg_eos_idx,
                    ignore_first_eol=False)

            # normalize costs according to the sequence lengths
            if self.normalize:
                lengths = numpy.array([len(s) for s in trans])
                costs = costs / lengths

            nbest_idx = numpy.argsort(costs)[:self.n_best]
            for j, best in enumerate(nbest_idx):
                try:
                    total_cost += costs[best]
                    trans_out = trans[best]

                    # convert idx to words
                    try:
                        sample_length = trans_out.index(self.trg_vocab['</S>'])
                    except ValueError:
                        sample_length = len(seq)
                    trans_out = trans_out[:sample_length]
                    trans_out = self._idx_to_word(trans_out, self.trg_ivocab)

                except ValueError:
                    logger.info(
                        "Can NOT find a translation for line: {}".format(i + 1))
                    trans_out = '<UNK>'

                if j == 0:
                    # Write to subprocess and file if it exists
                    print("Line:", i)
                    print("Input : ", self._idx_to_word(line[0], self.src_ivocab))
                    print("Sample: ", trans_out)
                    print("Error:", costs[best])
                    print()

                    print(trans_out, file=mb_subprocess.stdin)
                    if self.verbose:
                        print(trans_out, file=ftrans)

            if i != 0 and i % 100 == 0:
                logger.info(
                    "Translated {} lines of test set...".format(i))

            mb_subprocess.stdin.flush()

        logger.info("Total cost of the test: {}".format(total_cost))
        self.data_stream.reset()
        if self.verbose:
            ftrans.close()

        # send end of file, read output.
        mb_subprocess.stdin.close()
        stdout = mb_subprocess.stdout.readline()
        logger.info(stdout)
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        logger.info("Test Took: {} minutes".format(
            float(time.time() - test_start_time) / 60.))
        assert out_parse is not None

        # extract the score
        bleu_score = float(out_parse.group()[6:])
        logger.info(bleu_score)
        mb_subprocess.terminate()

        return bleu_score



class ModelInfo:
    """Utility class to keep track of evaluated models."""

    def __init__(self, bleu_score, path=''):
        self.bleu_score = bleu_score

        self.path = self._generate_path(path)

    def _generate_path(self, path):
        gen_path = os.path.join(
            path, 'best_bleu_params_BLEU%.2f.npz' %
                  (self.bleu_score) if path else None)
        return gen_path
