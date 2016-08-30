import copy

from blocks.bricks import (Tanh, Linear, FeedforwardSequence, Identity,
                           Initializable, MLP)
from blocks.bricks.attention import SequenceContentAttention, AttentionRecurrent
from blocks.bricks.base import application, lazy
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional, recurrent, RecurrentStack
from blocks.bricks.sequence_generators import (
    Readout, SoftmaxEmitter, BaseSequenceGenerator)
from blocks.roles import add_role, WEIGHT
from blocks.utils import shared_floatx_nans, dict_subset, dict_union
from theano import tensor
from toolz import merge
from blocks.bricks.recurrent.misc import RECURRENTSTACK_SEPARATOR

# Helper class
class InitializableFeedforwardSequence(FeedforwardSequence, Initializable):
    pass


class DGRU(GatedRecurrent):
    """DGRU in Decimator"""

    def __init__(self, dim, activation=None, gate_activation=None,
                 **kwargs):
        super(DGRU, self).__init__(dim, activation, gate_activation, **kwargs)

    @recurrent(sequences=['mask', 'inputs', 'gate_inputs'],
               states=['states'], outputs=['states'], contexts=[])
    def apply(self, inputs, gate_inputs, states, mask=None):
        """Apply the gated recurrent transition.
        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, dim). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            dim)
        gate_inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs to the gates in the
            shape (batch_size, 2 * dim).
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            the charater available, 0 if there is the delimiter.
        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            Next states of the network.
        """
        gate_values = self.gate_activation.apply(
            states.dot(self.state_to_gates) + gate_inputs)
        update_values = gate_values[:, :self.dim]
        reset_values = gate_values[:, self.dim:]
        states_reset = states * reset_values
        next_states = self.activation.apply(
            states_reset.dot(self.state_to_state) + inputs)
        next_states = (next_states * update_values +
                       states * (1 - update_values))
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * self.initial_states(mask.shape[0]))
        return next_states


class Decimator(Initializable):
    """Char encoder, mapping a charater-level word to a vector"""

    def __init__(self, vocab_size, embedding_dim, dgru_state_dim, dgru_depth, **kwargs):
        super(Decimator, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dgru_state_dim = dgru_state_dim
        self.embedding_dim = embedding_dim
        self.lookup = LookupTable(name='embeddings')
        self.dgru_depth = dgru_depth
        self.dgru = RecurrentStack([DGRU(activation=Tanh(), dim=self.dgru_state_dim) for _ in range(dgru_depth)],
                                   skip_connections=True)

        self.gru_fork = Fork([name for name in self.dgru.apply.sequences
                              if name != 'mask'], prototype=Linear(), name='gru_fork')

        self.children = [self.lookup, self.dgru, self.gru_fork]

    def _push_allocation_config(self):
        self.lookup.length = self.vocab_size
        self.lookup.dim = self.embedding_dim

        self.gru_fork.input_dim = self.embedding_dim
        self.gru_fork.output_dims = [self.dgru.get_dim(name)
                                     for name in self.gru_fork.output_names]

    @application(inputs=['char_seq', 'sample_matrix', 'char_aux'],
                 outputs=['representation'])
    def apply(self, char_seq, sample_matrix, char_aux):
        # Time as first dimension
        embeddings = self.lookup.apply(char_seq)
        gru_out = self.dgru.apply(
            **merge(self.gru_fork.apply(embeddings, as_dict=True),
                    {'mask': char_aux}))
        if self.dgru_depth > 1:
            gru_out = gru_out[-1]
        sampled_representation = tensor.batched_dot(sample_matrix, gru_out.dimshuffle([1, 0, 2]))
        return sampled_representation.dimshuffle([1, 0, 2])

    @application(inputs=['target_single_char'])
    def single_emit(self, target_single_char, batch_size, mask, states=None):
        # Time as first dimension
        # only one batch
        embeddings = self.lookup.apply(target_single_char)
        if states is None:
            states = self.dgru.initial_states(batch_size)
        states_dict = {'states':states[0]}
        for i in range(1,self.dgru_depth):
            states_dict['states'+RECURRENTSTACK_SEPARATOR+str(i)] = states[i]
        gru_out = self.dgru.apply(**merge(self.gru_fork.apply(embeddings, as_dict=True), states_dict,
                                          {'mask': mask, 'iterate': False}))
        return gru_out

    @single_emit.property('outputs')
    def single_emit_outputs(self):
        return ['gru_out' + RECURRENTSTACK_SEPARATOR + str(i) for i in range(self.dgru_depth)]

    def get_dim(self, name):
        if name in ['output', 'feedback']:
            return self.dgru_state_dim
        super(Decimator, self).get_dim(name)


class RecurrentWithFork(Initializable):
    @lazy(allocation=['input_dim'])
    def __init__(self, proto, input_dim, **kwargs):
        super(RecurrentWithFork, self).__init__(**kwargs)
        self.recurrent = proto
        self.input_dim = input_dim
        self.fork = Fork(
            [name for name in self.recurrent.apply.sequences
             if name != 'mask'],
            prototype=Linear())
        self.children = [self.recurrent, self.fork]

    def _push_allocation_config(self):
        self.fork.input_dim = self.input_dim
        self.fork.output_dims = [self.recurrent.get_dim(name)
                                 for name in self.fork.output_names]

    @application(inputs=['input_', 'mask'])
    def apply(self, input_, mask=None, **kwargs):
        return self.recurrent.apply(
            mask=mask, **dict_union(self.fork.apply(input_, as_dict=True),
                                    kwargs))

    @apply.property('outputs')
    def apply_outputs(self):
        return self.recurrent.states


class BidirectionalEncoder(Initializable):
    """Encoder of model."""

    def __init__(self, src_vocab_size, embedding_dim, dgru_state_dim, state_dim, src_dgru_depth,
                 bidir_encoder_depth, **kwargs):
        super(BidirectionalEncoder, self).__init__(**kwargs)
        self.state_dim = state_dim
        self.dgru_state_dim = dgru_state_dim
        self.decimator = Decimator(src_vocab_size, embedding_dim, dgru_state_dim, src_dgru_depth)
        self.bidir = Bidirectional(
            RecurrentWithFork(GatedRecurrent(activation=Tanh(), dim=state_dim), dgru_state_dim, name='with_fork'),
            name='bidir0')

        self.children = [self.decimator, self.bidir]
        for layer_n in range(1, bidir_encoder_depth):
            self.children.append(copy.deepcopy(self.bidir))
            for child in self.children[-1].children:
                child.input_dim = 2 * state_dim
            self.children[-1].name = 'bidir{}'.format(layer_n)

    @application(inputs=['source_char_seq', 'source_sample_matrix', 'source_char_aux', 'source_word_mask'],
                 outputs=['representation'])
    def apply(self, source_char_seq, source_sample_matrix, source_char_aux, source_word_mask):
        # Time as first dimension
        source_char_seq = source_char_seq.T
        source_char_aux = source_char_aux.T
        source_word_mask = source_word_mask.T
        source_word_representation = self.decimator.apply(source_char_seq, source_sample_matrix, source_char_aux)
        representation = source_word_representation
        for bidir in self.children[1:]:
            representation = bidir.apply(representation, source_word_mask)
        return representation


class IGRU(GatedRecurrent):
    """IGRU in interpolator """

    def __init__(self, dim, activation=None, gate_activation=None,
                 **kwargs):
        super(IGRU, self).__init__(dim, activation, gate_activation, **kwargs)

    @recurrent(sequences=['mask', 'inputs', 'gate_inputs', 'input_states'],
               states=['states'], outputs=['states'], contexts=[])
    def apply(self, inputs, gate_inputs, states, input_states, mask=None):
        """Apply the gated recurrent transition.
        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, dim). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            dim)
        gate_inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs to the gates in the
            shape (batch_size, 2 * dim).
        input_states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of outputs of decoder in the shape
            (batch_size, dim), which generated by decoder.
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            the charater available, 0 if there is the delimiter.
        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            Next states of the network.
        """
        # put masked states at last may be possible
        if mask:
            states = (mask[:, None] * states + (1 - mask[:, None]) * input_states)
        gate_values = self.gate_activation.apply(
            states.dot(self.state_to_gates) + gate_inputs)
        update_values = gate_values[:, :self.dim]
        reset_values = gate_values[:, self.dim:]
        #states_reset = (states + input_states) * reset_values / 2
        states_reset = states * reset_values
        next_states = self.activation.apply(
            states_reset.dot(self.state_to_state) + inputs)
        next_states = (next_states * update_values +
                       states * (1 - update_values))
        return next_states

    # using constant initial_states
    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                                                  name='state_to_state'))
        self.parameters.append(shared_floatx_nans((self.dim, 2 * self.dim),
                                                  name='state_to_gates'))
        for i in range(2):
            if self.parameters[i]:
                add_role(self.parameters[i], WEIGHT)

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return tensor.zeros((batch_size, self.dim))


class UpperIGRU(GatedRecurrent):
    """ Upper IGRU in interpolator """

    def __init__(self, dim, activation=None, gate_activation=None,
                 **kwargs):
        super(UpperIGRU, self).__init__(dim, activation, gate_activation, **kwargs)

    @recurrent(sequences=['mask', 'inputs', 'gate_inputs'],
               states=['states'], outputs=['states'], contexts=[])
    def apply(self, inputs, gate_inputs, states, mask=None):
        """Apply the gated recurrent transition.
        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, dim). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            dim)
        gate_inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs to the gates in the
            shape (batch_size, 2 * dim).
        input_states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of outputs of decoder in the shape
            (batch_size, dim), which generated by decoder.
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            the charater available, 0 if there is the delimiter.
        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            Next states of the network.
        """
        if mask:
            states = (mask[:, None] * states + (1 - mask[:, None]) * self.initial_states(mask.shape[0]))
        gate_values = self.gate_activation.apply(
            states.dot(self.state_to_gates) + gate_inputs)
        update_values = gate_values[:, :self.dim]
        reset_values = gate_values[:, self.dim:]
        states_reset = states * reset_values
        next_states = self.activation.apply(
            states_reset.dot(self.state_to_state) + inputs)
        next_states = (next_states * update_values +
                       states * (1 - update_values))

        return next_states


class Interpolator(Readout):
    def __init__(self, vocab_size, embedding_dim, igru_state_dim, igru_depth, trg_dgru_depth, emitter=None, feedback_brick=None,
                 merge=None, merge_prototype=None, post_merge=None, merged_dim=None, igru=None, **kwargs):
        # for compatible
        if igru_depth == 1:
            self.igru = IGRU(dim=igru_state_dim)
        else:
            self.igru = RecurrentStack([IGRU(dim=igru_state_dim, name='igru')] +
                                       [UpperIGRU(dim=igru_state_dim, activation=Tanh(), name='upper_igru' + str(i))
                                        for i in range(1, igru_depth)],
                                        skip_connections=True)
        self.igru_depth = igru_depth
        self.trg_dgru_depth = trg_dgru_depth
        self.lookup = LookupTable(name='embeddings')
        self.vocab_size = vocab_size
        self.igru_state_dim = igru_state_dim
        self.gru_to_softmax = Linear(input_dim=igru_state_dim, output_dim=vocab_size)
        self.embedding_dim = embedding_dim
        self.gru_fork = Fork([name for name in self.igru.apply.sequences
                              if name != 'mask' and name != 'input_states'], prototype=Linear(), name='gru_fork')
        kwargs['children'] = [self.igru, self.lookup, self.gru_to_softmax, self.gru_fork]
        super(Interpolator, self).__init__(emitter=emitter, feedback_brick=feedback_brick, merge=merge,
                                           merge_prototype=merge_prototype, post_merge=post_merge,
                                           merged_dim=merged_dim, **kwargs)

    @application
    def initial_igru_outputs(self, batch_size):
        return self.igru.initial_states(batch_size)

    def _push_allocation_config(self):
        self.lookup.length = self.vocab_size
        self.lookup.dim = self.embedding_dim
        self.emitter.readout_dim = self.get_dim('readouts')
        self.merge.input_names = self.source_names
        self.merge.input_dims = self.source_dims
        self.merge.output_dim = self.merged_dim
        self.post_merge.input_dim = self.merged_dim
        self.post_merge.output_dim = self.igru_state_dim
        self.gru_fork.input_dim = self.embedding_dim
        self.gru_fork.output_dims = [self.igru.get_dim(name)
                                     for name in self.gru_fork.output_names]

    @application(outputs=['feedback'])
    def feedback_apply(self, target_char_seq, target_sample_matrix, target_char_aux):
        return self.feedback_brick.apply(target_char_seq, target_sample_matrix, target_char_aux)

    @application
    def single_feedback(self, target_single_char, batch_size, mask=None, states=None):
        return self.feedback_brick.single_emit(target_single_char, batch_size, mask, states)

    @single_feedback.property('outputs')
    def single_feedback_outputs(self):
        return ['single_feedback' + RECURRENTSTACK_SEPARATOR + str(i) for i in range(self.trg_dgru_depth)]

    @application(outputs=['gru_out', 'readout_chars'])
    def single_readout_gru(self, target_prev_char, target_prev_char_aux, input_states, states):
        embeddings = self.lookup.apply(target_prev_char)
        states_dict = {'states':states[0]}
        if self.igru_depth > 1:
            for i in range(1, self.igru_depth):
                states_dict['states' + RECURRENTSTACK_SEPARATOR +str(i)] = states[i]
        gru_out = self.igru.apply(
            **merge(self.gru_fork.apply(embeddings, as_dict=True), states_dict,
                    {'mask': target_prev_char_aux, 'input_states': input_states,
                     'iterate': False}))
        if self.igru_depth > 1:
            readout_chars = self.gru_to_softmax.apply(gru_out[-1])
        else:
            readout_chars = self.gru_to_softmax.apply(gru_out)
        return gru_out, readout_chars

    @application(outputs=['readout_chars'])
    def readout_gru(self, target_prev_char_seq, target_prev_char_aux, input_states):
        embeddings = self.lookup.apply(target_prev_char_seq)
        gru_out = self.igru.apply(
            **merge(self.gru_fork.apply(embeddings, as_dict=True),
                    {'mask': target_prev_char_aux, 'input_states': input_states}))
        if self.igru_depth > 1:
            gru_out = gru_out[-1]
        readout_chars = self.gru_to_softmax.apply(gru_out)
        return readout_chars


class SequenceGeneratorDCNMT(BaseSequenceGenerator):
    """A more user-friendly interface for :class:`BaseSequenceGenerator`. """

    def __init__(self, trg_space_idx, readout, transition, attention=None, transition_depth=1, igru_depth=1, trg_dgru_depth=1,
                 add_contexts=True, **kwargs):
        self.trg_space_idx = trg_space_idx
        self.transition_depth = transition_depth
        self.igru_depth = igru_depth
        self.trg_dgru_depth = trg_dgru_depth
        self.igru_states_name = ['igru_states'+RECURRENTSTACK_SEPARATOR+str(i) for i in range(self.igru_depth)]
        self.feedback_name = ['feedback'+RECURRENTSTACK_SEPARATOR+str(i) for i in range(self.trg_dgru_depth)]

        normal_inputs = [name for name in transition.apply.sequences
                         if 'mask' not in name]
        kwargs.setdefault('fork', Fork(normal_inputs))
        transition = AttentionRecurrent(
            transition, attention,
            add_contexts=add_contexts, name="att_trans")
        super(SequenceGeneratorDCNMT, self).__init__(
            readout, transition, **kwargs)

    @application
    def cost_matrix_nmt(self, application_call, target_char_seq, target_sample_matrix, target_resample_matrix,
                        target_word_mask, target_char_aux, target_prev_char_seq, target_prev_char_aux, **kwargs):
        """Returns generation costs for output sequences.

        See Also
        --------
        :meth:`cost` : Scalar cost.

        """
        # We assume the data has axes (time, batch, features, ...)
        batch_size = target_char_seq.shape[1]

        # Prepare input for the iterative part
        states = dict_subset(kwargs, self._state_names, must_have=False)
        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)
        feedback = self.readout.feedback_apply(target_char_seq, target_sample_matrix, target_char_aux)
        inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network
        results = self.transition.apply(
            mask=target_word_mask, return_initial_states=True, as_dict=True,
            **dict_union(inputs, states, contexts))

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        states = {name: results[name][:-1] for name in self._state_names}
        glimpses = {name: results[name][1:] for name in self._glimpse_names}

        feedback = tensor.roll(feedback, 1, 0)
        if self.trg_dgru_depth == 1:
            feedback = tensor.set_subtensor(
                feedback[0],
                self.readout.single_feedback(self.readout.initial_outputs(batch_size), batch_size))
        else:
            feedback = tensor.set_subtensor(
                feedback[0],
                self.readout.single_feedback(self.readout.initial_outputs(batch_size), batch_size)[-1])

        decoder_readout_outputs = self.readout.readout(
            feedback=feedback, **dict_union(states, glimpses, contexts))
        resampled_representation = tensor.batched_dot(target_resample_matrix,
                                                      decoder_readout_outputs.dimshuffle([1, 0, 2]))
        resampled_readouts = resampled_representation.dimshuffle([1, 0, 2])
        readouts_chars = self.readout.readout_gru(target_prev_char_seq, target_prev_char_aux, resampled_readouts)

        # Compute the cost
        costs = self.readout.cost(readouts_chars, target_char_seq)

        for name, variable in list(glimpses.items()) + list(states.items()):
            application_call.add_auxiliary_variable(
                variable.copy(), name=name)

        # This variables can be used to initialize the initial states of the
        # next batch using the last states of the current batch.
        for name in self._state_names + self._glimpse_names:
            application_call.add_auxiliary_variable(
                results[name][-1].copy(), name=name + "_final_value")

        return costs

    @recurrent
    def generate(self, outputs, **kwargs):
        """A sequence generation step.

        Parameters
        ----------
        outputs : :class:`~tensor.TensorVariable`
            The outputs from the previous step.

        Notes
        -----
        The contexts, previous states and glimpses are expected as keyword
        arguments.

        """
        states = dict_subset(kwargs, self._state_names)
        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)
        glimpses = dict_subset(kwargs, self._glimpse_names)
        feedback = dict_subset(kwargs, self.feedback_name)
        readout_feedback = dict_subset(kwargs, ['readout_feedback'])['readout_feedback']
        batch_size = outputs.shape[0]
        igru_states = dict_subset(kwargs, self.igru_states_name)

        next_glimpses = self.transition.take_glimpses(
            as_dict=True, **dict_union(states, glimpses, contexts))

        next_readouts = self.readout.readout(
            feedback=readout_feedback,
            **dict_union(states, next_glimpses, contexts))

        next_char_aux = 1 - tensor.eq(outputs, 0) - tensor.eq(outputs, self.trg_space_idx)
        next_igru_states, readout_chars = self.readout.single_readout_gru(outputs, next_char_aux, next_readouts,
                                                                          [igru_states[self.igru_states_name[i]] for i in range(self.igru_depth)])
        next_outputs = self.readout.emit(readout_chars)
        next_costs = self.readout.cost(readout_chars, next_outputs)

        update_next = tensor.eq(next_outputs, self.trg_space_idx)
        next_char_mask = 1 - update_next
        update_next = update_next[:, None]
        next_readout_feedback = (1 - update_next) * readout_feedback + update_next * feedback[self.feedback_name[-1]]

        next_feedback = self.readout.single_feedback(next_outputs, batch_size, next_char_mask,
                                                     [feedback[self.feedback_name[i]] for i in range(self.trg_dgru_depth)])

        next_inputs = (self.fork.apply(next_readout_feedback, as_dict=True)
                       if self.fork else {'feedback': next_readout_feedback})
        next_states = self.transition.compute_states(
            as_list=True,
            **dict_union(next_inputs, states, next_glimpses, contexts))

        next_states[0] = update_next * next_states[0] + (1 - update_next) * states['states']
        for i in range(1, self.transition_depth):
            next_states[i] = update_next * next_states[i] + (1 - update_next) * states['states' + RECURRENTSTACK_SEPARATOR + str(i)]

        next_glimpses['weights'] = update_next * next_glimpses['weights'] + (1 - update_next) * glimpses['weights']
        next_glimpses['weighted_averages'] = update_next * next_glimpses['weighted_averages'] + \
                                             (1 - update_next) * glimpses['weighted_averages']
        # combine all updates
        next_all = list(next_states) + [next_outputs] + list(next_glimpses.values())
        if self.trg_dgru_depth > 1:
            next_all += list(next_feedback)
        else:
            next_all += [next_feedback]
        if self.igru_depth > 1:
            next_all += list(next_igru_states)
        else:
            next_all += [next_igru_states]
        next_all += [next_readout_feedback] + [next_costs]

        return (next_all)

    @generate.delegate
    def generate_delegate(self):
        return self.transition.apply

    @generate.property('outputs')
    def generate_outputs(self):
        return self._state_names + ['outputs'] + self._glimpse_names + \
              self.feedback_name + self.igru_states_name + ['readout_feedback', 'cost']

    @generate.property('states')
    def generate_states(self):
        return self._state_names + ['outputs'] + self._glimpse_names + \
            self.feedback_name + self.igru_states_name + ['readout_feedback']

    def get_dim(self, name):
        if name in (self._state_names + self._context_names + self._glimpse_names):
            return self.transition.get_dim(name)
        elif name == 'outputs' or name in self.feedback_name or name == 'readout_feedback' or name in self.igru_states_name:
            return self.readout.get_dim('outputs')
        return super(BaseSequenceGenerator, self).get_dim(name)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        # TODO: support dict of outputs for application methods
        # to simplify this code.
        igru_initial_states = self.readout.initial_igru_outputs(batch_size)
        if self.igru_depth == 1:
            igru_initial_states_dict = {self.igru_states_name[0]:igru_initial_states}
        else:
            igru_initial_states_dict = {self.igru_states_name[i]:igru_initial_states[i]
                                        for i in range(self.igru_depth)}

        initial_outputs=self.readout.initial_outputs(batch_size)
        feedback = self.readout.single_feedback(initial_outputs, batch_size)
        if self.trg_dgru_depth == 1:
            feedback_dict = {self.feedback_name[0]:feedback, 'readout_feedback':feedback}
        else:
            feedback_dict = {'readout_feedback':feedback[-1]}
            feedback_dict.update({self.feedback_name[i]:feedback[i] for i in range(self.trg_dgru_depth)})

        state_dict = dict(
            self.transition.initial_states(
                batch_size, as_dict=True, *args, **kwargs),
            outputs=initial_outputs)
        state_dict.update(feedback_dict)
        state_dict.update(igru_initial_states_dict)
        return [state_dict[state_name]
                for state_name in self.generate.states]

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.generate.states


class GRUInitialState(GatedRecurrent):
    """Gated Recurrent with special initial state.

    Initial state of Gated Recurrent is set by an MLP that conditions on the
    first hidden state of the bidirectional encoder, applies an affine
    transformation followed by a tanh non-linearity to set initial state.

    """

    def __init__(self, attended_dim, **kwargs):
        super(GRUInitialState, self).__init__(**kwargs)
        self.attended_dim = attended_dim
        self.initial_transformer = MLP(activations=[Tanh()],
                                       dims=[attended_dim, self.dim],
                                       name='state_initializer')
        self.children.append(self.initial_transformer)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        attended = kwargs['attended']
        initial_state = self.initial_transformer.apply(
            attended[0, :, -self.attended_dim:])
        return initial_state

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                                                  name='state_to_state'))
        self.parameters.append(shared_floatx_nans((self.dim, 2 * self.dim),
                                                  name='state_to_gates'))
        for i in range(2):
            if self.parameters[i]:
                add_role(self.parameters[i], WEIGHT)


class Decoder(Initializable):
    """Decoder of dcnmt model."""

    def __init__(self, vocab_size, embedding_dim, dgru_state_dim, igru_state_dim, state_dim,
                 representation_dim, transition_depth, trg_igru_depth, trg_dgru_depth, trg_space_idx, trg_bos,
                 theano_seed=None, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dgru_state_dim = dgru_state_dim
        self.igru_state_dim = igru_state_dim
        self.state_dim = state_dim
        self.trg_space_idx = trg_space_idx
        self.representation_dim = representation_dim
        self.theano_seed = theano_seed

        # Initialize gru with special initial state
        self.transition = RecurrentStack(
            [GRUInitialState(attended_dim=state_dim, dim=state_dim, activation=Tanh(), name='decoder_gru_withinit')] +
            [GatedRecurrent(dim=state_dim, activation=Tanh(), name='decoder_gru' + str(i))
             for i in range(1, transition_depth)], skip_connections=True)

        # Initialize the attention mechanism
        self.attention = SequenceContentAttention(
            state_names=self.transition.apply.states,
            attended_dim=representation_dim,
            match_dim=state_dim, name="attention")

        # for compatible
        if self.igru_state_dim == self.state_dim:
            post_merge_layer = [Identity().apply]
        else:
            post_merge_layer = [Linear(input_dim=self.state_dim,
                                      output_dim=self.igru_state_dim).apply,
                                Tanh().apply]

        self.interpolator = Interpolator(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            igru_state_dim=igru_state_dim,
            igru_depth=trg_igru_depth,
            trg_dgru_depth=trg_dgru_depth,
            source_names=['states', 'feedback', self.attention.take_glimpses.outputs[0]],
            readout_dim=self.vocab_size,
            emitter=SoftmaxEmitter(initial_output=trg_bos, theano_seed=theano_seed),
            feedback_brick=Decimator(vocab_size, embedding_dim, self.dgru_state_dim, trg_dgru_depth),
            post_merge=InitializableFeedforwardSequence(post_merge_layer),
            merged_dim=igru_state_dim)

        # Build sequence generator accordingly
        self.sequence_generator = SequenceGeneratorDCNMT(
            trg_space_idx=self.trg_space_idx,
            readout=self.interpolator,
            transition=self.transition,
            attention=self.attention,
            transition_depth=transition_depth,
            igru_depth=trg_igru_depth,
            trg_dgru_depth=trg_dgru_depth,
            fork=Fork([name for name in self.transition.apply.sequences
                       if name != 'mask'], prototype=Linear())
        )
        self.children = [self.sequence_generator]

    @application(inputs=['representation', 'source_word_mask', 'target_char_seq', 'target_sample_matrix',
                         'target_resample_matrix', 'target_char_aux', 'target_char_mask', 'target_word_mask',
                         'target_prev_char_seq', 'target_prev_char_aux'],
                 outputs=['cost'])
    def cost(self, representation, source_word_mask, target_char_seq, target_sample_matrix,
             target_resample_matrix, target_char_aux, target_char_mask, target_word_mask,
             target_prev_char_seq, target_prev_char_aux):
        source_word_mask = source_word_mask.T
        target_char_seq = target_char_seq.T
        target_prev_char_seq = target_prev_char_seq.T
        target_char_mask = target_char_mask.T
        target_char_aux = target_char_aux.T
        target_prev_char_aux = target_prev_char_aux.T
        target_word_mask = target_word_mask.T

        # Get the cost matrix
        cost = self.sequence_generator.cost_matrix_nmt(**{
            'target_char_seq': target_char_seq,
            'target_sample_matrix': target_sample_matrix,
            'target_resample_matrix': target_resample_matrix,
            'target_word_mask': target_word_mask,
            'target_char_aux': target_char_aux,
            'target_prev_char_seq': target_prev_char_seq,
            'target_prev_char_aux': target_prev_char_aux,
            'attended': representation,
            'attended_mask': source_word_mask})

        return (cost * target_char_mask).sum() / target_char_mask.shape[1]

    @application
    def generate(self, representation, attended_mask, **kwargs):
        return self.sequence_generator.generate(
            n_steps=10 * attended_mask.shape[1],
            batch_size=attended_mask.shape[0],
            attended=representation,
            attended_mask=attended_mask.T,
            **kwargs)
