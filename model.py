import copy

from blocks.bricks import (Tanh, Linear, FeedforwardSequence, Identity,
                           Initializable, MLP)
from blocks.bricks.attention import SequenceContentAttention, AttentionRecurrent

from blocks.bricks.base import application, lazy
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional, recurrent, BaseRecurrent, RecurrentStack
from blocks.bricks.sequence_generators import (
    Readout, SoftmaxEmitter, BaseSequenceGenerator)
from blocks.roles import add_role, WEIGHT
from blocks.utils import shared_floatx_nans, dict_subset, dict_union

from picklable_itertools.extras import equizip
from theano import tensor
from toolz import merge


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
            data available, 0 if not. Assumed to be 1-s only if not given.
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
    """Char encoder, mapping a char-level word to a vector"""

    def __init__(self, vocab_size, embedding_dim, dgru_state_dim, dgru_layers, **kwargs):
        super(Decimator, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dgru_state_dim = dgru_state_dim
        self.embedding_dim = embedding_dim
        self.lookup = LookupTable(name='embeddings')
        self.dgru_layers = dgru_layers
        if dgru_layers == 1:
            self.dgru = DGRU(activation=Tanh(), dim=self.dgru_state_dim)
        else:
            self.dgru = RecurrentStack([DGRU(activation=Tanh(), dim=self.dgru_state_dim) for _ in range(dgru_layers)])

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
        if self.dgru_layers > 1:
            gru_out = gru_out[-1]
        sampled_representation = tensor.batched_dot(sample_matrix, gru_out.dimshuffle([1, 0, 2]))
        return sampled_representation.dimshuffle([1, 0, 2])

    @application(inputs=['target_single_char'],
                 outputs=['gru_out'])
    def single_emit(self, target_single_char, batch_size, mask, states=None):
        # Time as first dimension
        # only one batch
        embeddings = self.lookup.apply(target_single_char)
        if states is None:
            states = self.dgru.initial_states(batch_size)
        gru_out = self.dgru.apply(**merge(self.gru_fork.apply(embeddings, as_dict=True),
                                          {'states': states, 'mask': mask, 'iterate': False}))
        return gru_out

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
            [name for name in self.recurrent.sequences
             if name != 'mask'],
            prototype=Linear())
        self.children = [self.recurrent.brick, self.fork]

    def _push_allocation_config(self):
        self.fork.input_dim = self.input_dim
        self.fork.output_dims = [self.recurrent.brick.get_dim(name)
                                 for name in self.fork.output_names]

    @application(inputs=['input_', 'mask'])
    def apply(self, input_, mask=None, **kwargs):
        return self.recurrent(
            mask=mask, **dict_union(self.fork.apply(input_, as_dict=True),
                                    kwargs))

    @apply.property('outputs')
    def apply_outputs(self):
        return self.recurrent.states


class BidirectionalEncoder(Initializable):
    """Encoder of model."""

    def __init__(self, src_vocab_size, embedding_dim, dgru_state_dim, state_dim, encoder_layers, **kwargs):
        super(BidirectionalEncoder, self).__init__(**kwargs)
        self.state_dim = state_dim
        self.dgru_state_dim = dgru_state_dim
        self.decimator = Decimator(src_vocab_size, embedding_dim, dgru_state_dim, encoder_layers)
        self.bidir = Bidirectional(
            RecurrentWithFork(GatedRecurrent(activation=Tanh(), dim=state_dim).apply, dgru_state_dim, name='with_fork'),
            name='bidir0')

        self.children = [self.decimator, self.bidir]
        for layer in range(1, encoder_layers):
            self.children.append(copy.deepcopy(self.bidir))
            for child in self.children[-1].children:
                child.input_dim = 2 * state_dim
            self.children[-1].name = 'bidir{}'.format(layer)

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
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.
        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            Next states of the network.
        """
        if mask:
            states = (mask[:, None] * states + (1 - mask[:, None]) * input_states)
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
    def __init__(self, vocab_size, embedding_dim, igru_state_dim, emitter=None, feedback_brick=None,
                 merge=None, merge_prototype=None, post_merge=None, merged_dim=None, igru=None, **kwargs):
        self.igru = igru
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

    @application(outputs=['single_feedback'])
    def single_feedback(self, target_single_char, batch_size, mask=None, states=None):
        return self.feedback_brick.single_emit(target_single_char, batch_size, mask, states)

    @application(outputs=['gru_out', 'readout_chars'])
    def single_readout_gru(self, target_prev_char, target_prev_char_aux, input_states, states):
        embeddings = self.lookup.apply(target_prev_char)
        gru_out = self.igru.apply(
            **merge(self.gru_fork.apply(embeddings, as_dict=True),
                    {'mask': target_prev_char_aux, 'input_states': input_states, 'states': states,
                     'iterate': False}))
        readout_chars = self.gru_to_softmax.apply(gru_out)
        return gru_out, readout_chars

    @application(outputs=['readout_chars'])
    def readout_gru(self, target_prev_char_seq, target_prev_char_aux, input_states):
        embeddings = self.lookup.apply(target_prev_char_seq)
        gru_out = self.igru.apply(
            **merge(self.gru_fork.apply(embeddings, as_dict=True),
                    {'mask': target_prev_char_aux, 'input_states': input_states}))
        readout_chars = self.gru_to_softmax.apply(gru_out)
        return readout_chars


class SequenceGeneratorDCNMT(BaseSequenceGenerator):
    r"""A more user-friendly interface for :class:`BaseSequenceGenerator`.

    Parameters
    ----------
    readout : instance of :class:`AbstractReadout`
        The readout component for the sequence generator.
    transition : instance of :class:`.BaseRecurrent`
        The recurrent transition to be used in the sequence generator.
        Will be combined with `attention`, if that one is given.
    attention : object, optional
        The attention mechanism to be added to ``transition``,
        an instance of
        :class:`~blocks.bricks.attention.AbstractAttention`.
    add_contexts : bool
        If ``True``, the
        :class:`.AttentionRecurrent` wrapping the
        `transition` will add additional contexts for the attended and its
        mask.
    \*\*kwargs : dict
        All keywords arguments are passed to the base class. If `fork`
        keyword argument is not provided, :class:`.Fork` is created
        that forks all transition sequential inputs without a "mask"
        substring in them.

    """

    def __init__(self, trg_space_idx, readout, transition, attention=None, transition_layers=1,
                 add_contexts=True, **kwargs):
        self.trg_space_idx = trg_space_idx
        self.transition_layers = transition_layers
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

        # Compute the cost
        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(
            feedback[0],
            self.readout.single_feedback(self.readout.initial_outputs(batch_size), batch_size))

        decoder_readout_outputs = self.readout.readout(
            feedback=feedback, **dict_union(states, glimpses, contexts))
        resampled_representation = tensor.batched_dot(target_resample_matrix,
                                                      decoder_readout_outputs.dimshuffle([1, 0, 2]))
        resampled_readouts = resampled_representation.dimshuffle([1, 0, 2])
        readouts_chars = self.readout.readout_gru(target_prev_char_seq, target_prev_char_aux, resampled_readouts)

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
        feedback = dict_subset(kwargs, ['feedback'])['feedback']
        gru_out = dict_subset(kwargs, ['gru_out'])['gru_out']
        readout_feedback = dict_subset(kwargs, ['readout_feedback'])['readout_feedback']
        batch_size = outputs.shape[0]

        next_glimpses = self.transition.take_glimpses(
            as_dict=True, **dict_union(states, glimpses, contexts))

        next_readouts = self.readout.readout(
            feedback=readout_feedback,
            **dict_union(states, next_glimpses, contexts))

        next_char_aux = 1 - tensor.eq(outputs, 0) - tensor.eq(outputs, self.trg_space_idx)
        next_gru_out, readout_chars = self.readout.single_readout_gru(outputs, next_char_aux, next_readouts, gru_out)
        next_outputs = self.readout.emit(readout_chars)
        next_costs = self.readout.cost(readout_chars, next_outputs)

        update_next = tensor.eq(next_outputs, self.trg_space_idx)
        next_char_mask = 1 - update_next
        update_next = update_next[:, None]
        next_readout_feedback = (1 - update_next) * readout_feedback + update_next * feedback

        next_feedback = self.readout.single_feedback(next_outputs, batch_size, next_char_mask, feedback)

        next_inputs = (self.fork.apply(next_readout_feedback, as_dict=True)
                       if self.fork else {'feedback': next_readout_feedback})
        next_states = self.transition.compute_states(
            as_list=True,
            **dict_union(next_inputs, states, next_glimpses, contexts))

        next_states[0] = update_next * next_states[0] + (1 - update_next) * states['states']
        for i in range(1, self.transition_layers):
            next_states[i] = update_next * next_states[i] + (1 - update_next) * states['states#' + str(i)]

        next_glimpses['weights'] = update_next * next_glimpses['weights'] + (1 - update_next) * glimpses['weights']
        next_glimpses['weighted_averages'] = update_next * next_glimpses['weighted_averages'] + \
                                             (1 - update_next) * glimpses['weighted_averages']

        return (list(next_states) + [next_outputs] +
                list(next_glimpses.values()) + [next_feedback] + [next_gru_out] +
                [next_readout_feedback] + [next_costs])

    @generate.delegate
    def generate_delegate(self):
        return self.transition.apply

    @generate.property('outputs')
    def generate_outputs(self):
        return self._state_names + ['outputs'] + self._glimpse_names + \
               ['feedback', 'gru_out', 'readout_feedback', 'costs']

    @generate.property('states')
    def generate_states(self):
        return self._state_names + ['outputs'] + self._glimpse_names + ['feedback', 'gru_out', 'readout_feedback']

    def get_dim(self, name):
        if name in (self._state_names + self._context_names + self._glimpse_names):
            return self.transition.get_dim(name)
        elif name == 'outputs' or name == 'feedback' or name == 'readout_feedback' or name == 'gru_out':
            return self.readout.get_dim('outputs')
        return super(BaseSequenceGenerator, self).get_dim(name)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        # TODO: support dict of outputs for application methods
        # to simplify this code.
        state_dict = dict(
            self.transition.initial_states(
                batch_size, as_dict=True, *args, **kwargs),
            outputs=self.readout.initial_outputs(batch_size),
            feedback=self.readout.single_feedback(self.readout.initial_outputs(batch_size), batch_size),
            gru_out=self.readout.initial_igru_outputs(batch_size),
            readout_feedback=self.readout.single_feedback(self.readout.initial_outputs(batch_size), batch_size))
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

    def __init__(self, vocab_size, embedding_dim, dgru_state_dim, state_dim,
                 representation_dim, transition_layers, trg_space_idx, trg_bos, theano_seed=None, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dgru_state_dim = dgru_state_dim
        self.state_dim = state_dim
        self.trg_space_idx = trg_space_idx
        self.representation_dim = representation_dim
        self.theano_seed = theano_seed

        # Initialize gru with special initial state
        self.transition = RecurrentStack(
            [GRUInitialState(attended_dim=state_dim, dim=state_dim, activation=Tanh(), name='decoder_init_gru')] + [
                GatedRecurrent(dim=state_dim, activation=Tanh(), name='decoder_gru' + str(i)) for i in
                range(1, transition_layers)], skip_connections=True)

        # Initialize the attention mechanism
        self.attention = SequenceContentAttention(
            state_names=self.transition.apply.states,
            attended_dim=representation_dim,
            match_dim=state_dim, name="attention")

        self.interpolator = Interpolator(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            igru_state_dim=state_dim,
            source_names=['states', 'feedback',
                          self.attention.take_glimpses.outputs[0]],
            readout_dim=self.vocab_size,
            emitter=SoftmaxEmitter(initial_output=trg_bos, theano_seed=theano_seed),
            igru=IGRU(dim=state_dim),
            feedback_brick=Decimator(vocab_size, embedding_dim, self.dgru_state_dim, 1),
            post_merge=InitializableFeedforwardSequence(
                [Identity().apply]),
            merged_dim=state_dim)

        # Build sequence generator accordingly
        self.sequence_generator = SequenceGeneratorDCNMT(
            trg_space_idx=self.trg_space_idx,
            readout=self.interpolator,
            transition=self.transition,
            attention=self.attention,
            transition_layers=transition_layers,
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
