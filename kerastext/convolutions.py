"""
Convolution Layers for N-Dimensional Text with Masking.

Calling each document a sample, the input is a batch of documents.
Assumes input shape is: (nbr_samples, ..., sentence_width, word_embedding_size).
Each document can be:
    - a single sentence or word-sequence (Input has 3 dimensions),
    - a sequence of sentences (Input has 4 dimensions),
    - or higher dimensions.

For variable-sized word-sequences, the sentence or word-sequence is assumed to be right-padded.
This means that in the accompanying mask (e.g. as output from an Embedding layer),
each sentence corresponds to a mask-sequence of: 1^{m} 0^{n}
where:
    - 1 is True (for a valid word), and 0 is False (to indicating padding)
    - m >= 0, n >= 0
    - m + n = sentence_width.

"""

# noinspection PyPep8Naming
import keras.backend as K
import keras.initializers
from keras.engine import InputSpec
from keras.utils import conv_utils

if K.backend() == 'tensorflow':
    # noinspection PyProtectedMember
    from keras.backend.tensorflow_backend import _to_tensor

from .utils import force_masked_to_zero, gather_from_last_axis


# =============================================================================
#   Layers: Convolution
# =============================================================================


# noinspection PyProtectedMember
class KTextConvolution(keras.layers.convolutional._Conv):
    """
    Convolution on Text with Masking, where each word is represented as an embedding.
    Assumes input shape is: (nbr_samples, ..., sentence_width, word_embedding_size).
    Calling each document a sample, the input is a batch of documents.
    Each document can be:
        - a single sentence or word-sequence (Input has 3 dimensions),
        - a sequence of sentences (Input has 4 dimensions),
        - or higher dimensions.

    # Masking
        For variable-sized word-sequences, the sentence or word-sequence is assumed to be right-padded.
        This means that in the accompanying mask (e.g. as conveyed from the preceding Embedding layer),
        each sentence corresponds to a mask-sequence of: 1^{m} 0^{n}
        where:
            - 1 is True (for a valid word), and 0 is False (to indicating padding)
            - m >= 0, n >= 0
            - m + n = sentence_width.
        Note that TensorFlow requires masks to be boolean.

    border_mode:
        Input is 0-padded on either side before convolving, depending on border_mode.
        same = (aka `half` in Theano) Just enough padding applied to make output seq length same as input seq length.
                In Theano, this requires `filter_length` to be odd. Size of padding is then (filter_length // 2).

        valid, causal = ('valid' aka 'Narrow'). No input padding applied.

        full = (aka 'Wide').  Padding size on either side is (filter_length - 1).
                This mode is only supported on Theano.

    weights:
        List of initial weights: [W, b].
        W.shape = (`kernel_size`, input_dim, `filters`), where input_dim is the size of the last dimension of the input.
        b.shape = (`filters`,)

    terminate_mask:
        If True Then no output mask. Default is False.

    force_masked_input_to_zero:
        If True (default) Then Force masked input values (where mask is False) to Zero.
        If preceding layer has already done this, then this extra step can be avoided by setting this arg to False.

    force_masked_output_to_zero:
        If True (default) Then Force masked output values to Zero.
        Change from default with caution!
        If `force_output_after_activation` is True (default)
            Then this step is applied AFTER the activation is applied to the convolved inputs
            Else before the activations.
        Note that the mask is not sent to the activation function.

    force_output_after_activation:
        See `force_masked_output_to_zero`. Default is True.

    If this is not the first layer in a model (which is the expected usage, e.g. preceded by NTextEmbedding),
    then no need to specify `input_dim` or `input_shape`.


    # Input shape
        N-Dimensional tensor with shape: `(samples, ..., sentence_width, input_dim)`,
        N >= 3.

    # Output shape
        N-Dimensional tensor with shape: `(samples, ..., output_width, nbr_filters)`.
        output_width depends on sentence_width, filter_length, border_mode and stride (subsample_length).
        See `conv_output_length`.


    # Documentation from Keras for keras.layers.convolutional.Conv1D
    ----------------------------------------------------------------

    1D convolution layer (e.g. temporal convolution).

    This layer creates a convolution kernel that is convolved
    with the layer input over a single spatial (or temporal) dimension
    to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide an `input_shape` argument
    (tuple of integers or `None`, e.g.
    `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
    or `(None, 128)` for variable-length sequences of 128-dimensional vectors.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).

        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.

        strides: An integer or tuple/list of a single integer,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.

        padding: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
            `"valid"` means "no padding".
            `"same"` results in padding the input such that
            the output has the same length as the original input.
            `"causal"` results in causal (dilated) convolutions, e.g. output[t]
            does not depend on input[t+1:]. Useful when modeling temporal data
            where the model should not violate the temporal order.
            See [WaveNet: A Generative Model for Raw Audio, section 2.1](https://arxiv.org/abs/1609.03499).

        dilation_rate: an integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.

        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).

        use_bias: Boolean, whether the layer uses a bias vector.

        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        3D tensor with shape: `(batch_size, steps, input_dim)`

    # Output shape
        3D tensor with shape: `(batch_size, new_steps, filters)`
        `steps` value might have changed due to padding or strides.
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 # Args not in keras.layers.convolutional.Convolution1D
                 terminate_mask=False,
                 force_masked_input_to_zero=True,
                 force_masked_output_to_zero=True,
                 force_output_after_activation=True,
                 **kwargs):
        assert padding != 'full' or K.backend() == 'theano', "padding = 'full' requires Theano backend."

        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        assert padding in {'valid', 'same', 'full', 'causal'}, "Invalid padding = " + padding
        assert self.strides[0] > 0, 'stride = {} must be > 0'.format(self.strides[0])
        if padding == 'same':
            assert kernel_size % 2 == 1, 'When padding is `same`, kernel_size must be Odd integer'

        assert self.strides[0] == 1 or self.dilation_rate[0] == 1, 'Both strides and dilation_rate cannot be > 1'

        self.supports_masking = True

        self.terminate_mask = terminate_mask
        self.force_masked_input_to_zero = force_masked_input_to_zero
        self.force_masked_output_to_zero = force_masked_output_to_zero
        self.force_output_after_activation = force_output_after_activation

        # If `input_shape` or `batch_input_shape` was specified in kwargs, then this is set in Layer.
        # `input_spec` is used in checking compatibility between connected layers.
        if hasattr(self, 'batch_input_shape'):
            self.input_spec = InputSpec(ndim=len(self.batch_input_shape))
        else:
            self.input_spec = None

        return

    def build(self, input_shape):
        # Save self.input_spec as super will modify it
        inp_spec = self.input_spec
        super().build(input_shape)

        channel_axis = -1
        input_dim = input_shape[channel_axis]
        if inp_spec is None:
            self.input_spec = InputSpec(ndim=len(input_shape), axes={channel_axis: input_dim})
        else:
            self.input_spec = InputSpec(ndim=inp_spec.ndim, axes={channel_axis: input_dim})

        # The following might not be needed, as it may be done in Layer.__call__()
        if self._initial_weights is not None:
            self.set_weights(self._initial_weights)
            # del self._initial_weights
            # self._initial_weights = None

        return

    def compute_output_shape(self, input_shape, symbolic=False):
        """
        output.ndim = input.ndim, the sizes of the last 2 dimensions get modified by the convolution.

        symbolic = True if requesting a symbolic expression for the output-shape.
        """
        input_seq_length = input_shape[-2]
        output_seq_length = conv_utils.conv_output_length(input_seq_length,
                                                          self.kernel_size[0],
                                                          padding=self.padding,
                                                          stride=self.strides[0],
                                                          dilation=self.dilation_rate[0])

        # Handle the different cases of calls to this method ...

        if K.backend() == 'theano':
            # In Theano a tensor's shape is a tuple
            output_shape = tuple(input_shape[:-2]) + (output_seq_length, self.filters)

        elif K.backend() == 'tensorflow':
            if symbolic:
                # In TensorFlow a tensor's shape is a tensor.
                out_shape_sfx = _to_tensor([output_seq_length, self.filters], 'int32')
                output_shape = K.concatenate([input_shape[:-2], out_shape_sfx], axis=0)

            else:
                if isinstance(input_shape, (list, tuple)):
                    output_shape = tuple(input_shape[:-2]) + (output_seq_length, self.filters)
                else:
                    # Not clear if this case ever happens
                    output_shape = tuple([None] * K.ndim(input_shape))

        else:
            raise NotImplementedError("Backend '{}' not supported".format(K.backend()))

        return output_shape

    def compute_mask(self, inputs, input_mask=None):
        if input_mask is None or self.terminate_mask:
            return None

        # input_mask.shape  = (nbr_samples, ..., input_seq_length), ndim = N
        # input_mask[i_1, ..., i_{N-1}] = 1* 0* = 1^{m} 0^{n} ... where m + n = N,  and  m, n >= 0
        # output_mask.shape = (nbr_samples, ..., output_seq_length), ndim = N
        # output_mask[i_1, ..., i_{N-1}] = 1* 0*
        # output_mask[i_1, ..., i_{N}] = 1 if any of the inputs to that step have mask value 1.

        kernel_width = self.kernel_size[0]

        if self.padding in ['same', 'causal']:

            output_mask = input_mask

        elif self.padding == 'valid':

            output_length = K.shape(input_mask)[-1] - kernel_width + 1
            output_mask = input_mask[..., :output_length]

        else:  # 'full'

            # ... left-pad the input_mask with `input_mask[..., 0]` till it is of required output_length.
            # This way, any convolutions at the right end that are completely over input_mask=False
            # will get output_mask=False. All other convolutions will get output_mask=True.

            # first elements from each row
            delta_elem = K.expand_dims(input_mask[..., 0])

            delta_length = kernel_width - 1
            delta_mask = K.repeat_elements(delta_elem, delta_length, axis=-1)

            output_mask = K.concatenate([delta_mask, input_mask], axis=-1)

        stride = self.strides[0]
        dilation = self.dilation_rate[0]
        if stride > 1:
            # In each row, select every stride'th element from output_mask, starting with element 0
            selector = K.arange(0, K.shape(output_mask)[-1], stride)
            output_mask = gather_from_last_axis(output_mask, selector)

        elif dilation > 1:
            # In each row, Select the first few elements of the input mask
            output_seq_length = conv_utils.conv_output_length(K.shape(inputs)[-2],
                                                              kernel_width,
                                                              padding=self.padding,
                                                              stride=stride,
                                                              dilation=dilation)
            output_mask = output_mask[..., :output_seq_length]

        return output_mask

    def _force_outputs_to_zero(self, input_ndim, inputs, input_mask, outputs):
        out_mask = self.compute_mask(inputs, input_mask)
        if out_mask is not None:
            outputs = force_masked_to_zero(outputs, out_mask, x_ndim=input_ndim)

        return outputs

    def call(self, inputs, mask=None):

        input_ndim = K.ndim(inputs)
        input_shape = K.shape(inputs)

        x = inputs
        if mask is not None and self.force_masked_input_to_zero:
            x = force_masked_to_zero(x, mask, x_ndim=input_ndim)

        # We will handle higher dimensioned inputs by converting them to 3-dimensions by expanding the batch dimension,
        # and then converting them back to output shape before returning.
        if input_ndim > 3:
            # Convert input to 3-D by expanding the batch dimension
            x = K.reshape(x, (-1, input_shape[-2], input_shape[-1]))

        outputs = K.conv1d(x,
                           self.kernel,
                           strides=self.strides[0],
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate[0])

        if self.use_bias:
            outputs = K.bias_add(outputs,
                                 self.bias,
                                 data_format=self.data_format)

        if input_ndim > 3:
            # convert back to appropriate shape
            out_shape = self.compute_output_shape(input_shape, symbolic=True)
            outputs = K.reshape(outputs, out_shape)

        # Handle Mask if pre-activation
        if self.force_masked_output_to_zero and not self.force_output_after_activation:
            outputs = self._force_outputs_to_zero(input_ndim, inputs, mask, outputs)

        if self.activation is not None:
            outputs = self.activation(outputs)  # , mask=out_mask)

        # Handle Mask if post-activation
        if self.force_masked_output_to_zero and self.force_output_after_activation:
            outputs = self._force_outputs_to_zero(input_ndim, inputs, mask, outputs)

        return outputs

    def get_config(self):
        base_config = super().get_config()
        base_config.pop('rank')
        base_config.pop('data_format')
        config = {
            'terminate_mask': self.terminate_mask,
            'force_masked_input_to_zero': self.force_masked_input_to_zero,
            'force_masked_output_to_zero': self.force_masked_output_to_zero,
            'force_output_after_activation': self.force_output_after_activation,
        }
        return dict(list(base_config.items()) + list(config.items()))
# /
