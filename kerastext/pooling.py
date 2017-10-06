"""
Pooling Layers for processing variable-length Text with appropriate Masking in Keras.

General support for representing a Document as N-dimensional text.
Each Word or Token is represented as an Integer index into a Vocabulary, with `0` used for padding.
The last dimension represents a Word-sequence.
So a Document represented as a sequence of Sentences would have 2 Dimensions:
    Doc = Array of Sentences
    Sentence = Array of Word-indices

Embeddings would translate each Doc into a 3D array.
A Batch of Docs would then be a 4D array.

Mask is a boolean array (in Theano represented as int8), where:
    1 = valid value
    0 = invalid value
So when X uses `0` as padding, mask = K.not_equal(X, 0).
Since this is designed for text, the Mask is assumed to be of the form:
    1* 0*  =  1^{m} 0^{n} ... where m + n = Sentence_width,  and  m, n >= 0
i.e. a consecutive sequence of 1's, followed by a sequence of 0's, where one of them can be empty.
The mask is also assumed to be at the Word (or Word-index) level.

"""

import numpy as np

# noinspection PyPep8Naming
import keras.backend as K
from keras.engine import Layer
from keras.utils import conv_utils

from .utils import masked_where


# =============================================================================
#   Globals
# =============================================================================

# A convenience value to use, e.g. as default for max()
LARGE_NEG = -9999.99


# A convenience value to use, e.g. as default for min()
LARGE_POS = 9999.99


# =============================================================================
#   Layers: Global Pooling
# =============================================================================


class _KTextGlobalPooling(Layer):
    """
    Abstract class for Global Pooling.

    # Arguments

        default_value:
            The value to use for masked inputs. This is also the value returned
            when all inputs in a vector are masked.

        keepdims:
            If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
            Default is False.

        terminate_mask:
            If True Then output-mask is always None. Default is False.

    # Input shape
        N-D tensor with shape: (samples, d2, ..., d[N-2], d[N-1], dN).

    # Output shape
        IF keepdims THEN:   N-D tensor shape (samples, d2, ..., d[N-2], 1, dN).
        ELSE: (default) (N-1)-D tensor shape (samples, d2, ..., d[N-2], dN).

    # Input mask shape:  (samples, d2, ..., d[N-2], d[N-1])
    # Output mask shape:
        IF keepdims THEN: (samples, d2, ..., d[N-2], 1)
        ELSE: (default)   (samples, d2, ..., d[N-2])

         out_mask[ivec] = 0 only if For all j, in_mask[ivec, j] == 0
                        = 1  o/w
    """

    def __init__(self, default_value=0.0, keepdims=False, terminate_mask=False, **kwargs):
        super().__init__(**kwargs)

        self.default_value = default_value
        self.keepdims = keepdims
        self.terminate_mask = terminate_mask

        self.supports_masking = True
        self.input_spec = None  # removed ndim restriction [InputSpec(ndim=3)]

    def compute_output_shape(self, input_shape):
        if self.keepdims:
            return tuple(input_shape[:-2]) + (1, input_shape[-1])
        else:
            return tuple(input_shape[:-2]) + (input_shape[-1],)

    def compute_mask(self, inp, input_mask=None):
        if input_mask is None or self.terminate_mask:
            return None

        # Input Mask at word level, which is the 2nd-last dimension.
        # Output Mask is at Seq level: False if all words in seq are masked, Else True

        if K.ndim(input_mask) == K.ndim(inp) - 1:
            # For TensorFlow compatibility, cast mask to 'int8' before summing
            return K.not_equal(K.sum(K.cast(input_mask, 'int8'), axis=-1, keepdims=self.keepdims), 0)
        # elif input_mask.ndim < inp.ndim - 1:
        #     return input_mask
        else:
            raise TypeError('input_mask.ndim should be {}, actual is {}'.format(inp.ndim - 1, input_mask.ndim))

    def call(self, x, mask=None):
        raise NotImplementedError

    def get_config(self):
        config = {'default_value': self.default_value,
                  'terminate_mask': self.terminate_mask,
                  }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
# /


class KTextGlobalMaxPooling(_KTextGlobalPooling):
    """
    Global max pooling along 2nd last dimension, of the features in the last dimension.
    """
    def __init__(self, default_value=LARGE_NEG, keepdims=False, terminate_mask=False, **kwargs):
        super().__init__(default_value, keepdims, terminate_mask, **kwargs)

    def call(self, x, mask=None):
        return masked_max(x, axis=-2, mask=mask, default_value=self.default_value, keepdims=self.keepdims)
# /


class KTextGlobalMinPooling(_KTextGlobalPooling):
    """
    Global min pooling along 2nd last dimension, of the features in the last dimension.
    """
    def __init__(self, default_value=LARGE_POS, keepdims=False, terminate_mask=False, **kwargs):
        super().__init__(default_value, keepdims, terminate_mask, **kwargs)

    def call(self, x, mask=None):
        return masked_min(x, axis=-2, mask=mask, default_value=self.default_value, keepdims=self.keepdims)
# /


class KTextGlobalAvgPooling(_KTextGlobalPooling):
    """
    Global average pooling along 2nd last dimension, of the features in the last dimension.
    """
    def __init__(self, keepdims=False, terminate_mask=False, **kwargs):
        super().__init__(keepdims=keepdims, terminate_mask=terminate_mask, **kwargs)

    def call(self, x, mask=None):
        return masked_avg(x, axis=-2, mask=mask, keepdims=self.keepdims)
# /


# =============================================================================
#   Layers: Pooling
# =============================================================================


class _KTextPooling1D(Layer):
    """
    Abstract class for pooling along 1 dimension:
    Pooling regions extend along the 2nd-last dimension of the input.
    The features that get pooled are in the last dimension of the input.

    # Arguments

        pool_size:
            Factor by which to downscale. In text, the number of adjacent words that are pooled.
            Integer >= 1. Default is 2.

        strides:
            The number of words shifted over to get to the next pooling region.
            Default `None` means consecutive pooling regions are adjacent and do not overlap, i.e. strides = pool_size.
            Otherwise Integer >= 1.

        padding: one of 'valid', 'same'.

            'valid' has the usual meaning: each pooling region must consist only of valid inputs (ignoring mask).
                So for example with strides = None, if input_width is not a multiple of pool_size,
                then the last (input_width % pool_size) inputs do not get pooled.
                This the most commonly used option.

            'same': output_width = (input_width + stride - 1) // stride
                Input is left- and right-padded with 0's. Padding-width is computed as:
                    pool_size - 2 if pool_size > 2 and pool_size % 2 == 1 else pool_size - 1
                Pooling regions then begin from the new left-most end.
                This produces some unusual results, so use with caution!

        default_value:
            The value to use for masked inputs. This is also the value returned
            when all inputs in a vector are masked.

        terminate_mask:
            If True Then output-mask is always None. Default is False.

    """

    def __init__(self, pool_size=2, strides=None, padding='valid',
                 default_value=0.0, terminate_mask=False,
                 **kwargs):

        assert padding in ['valid', 'same'], "Invalid value for `padding`: " + padding

        super().__init__(**kwargs)

        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 1, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 1, 'strides')
        self.padding = conv_utils.normalize_padding(padding)

        self.default_value = default_value
        self.terminate_mask = terminate_mask

        self.input_spec = None  # removed ndim restriction: InputSpec(ndim=3)
        return

    def compute_output_shape(self, input_shape):
        length = conv_utils.conv_output_length(input_shape[-2],
                                               self.pool_size[0],
                                               self.padding,
                                               self.strides[0])
        return tuple(input_shape[:-2]) + (length, input_shape[-1])

    def compute_mask(self, inputs, input_mask=None):
        if input_mask is None or self.terminate_mask:
            return None

        # input_mask.shape  = (nbr_samples, ..., input_seq_length), ndim = N
        # output_mask.shape = (nbr_samples, ..., output_seq_length), ndim = N
        # input_mask[i_1, ..., i_{N-1}] = 1* 0*
        # output_mask[i_1, ..., i_{N-1}] = 1* 0*
        # output_mask[i_1, ..., i_{N}] = 1 if any of the inputs to that step have mask value 1.

        pool_width = self.pool_size[0]

        if self.padding == 'same':

            output_mask = input_mask

        elif self.padding == 'valid':

            output_length = K.shape(input_mask)[-1] - pool_width + 1
            output_mask = input_mask[..., :output_length]

        else:  # 'full' not supported
            raise ValueError('Invalid value for `padding`:', self.padding)

        stride = self.strides[0]
        if stride > 1:
            # In each row, select every stride'th element from output_mask, starting with element 0
            selector = K.arange(0, K.shape(output_mask)[-1], stride)
            output_mask = output_mask[..., selector]

        return output_mask

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        raise NotImplementedError

    def call(self, inputs, mask=None):
        input_ndim = K.ndim(inputs)
        input_shape = K.shape(inputs)

        if mask is not None:
            # Set masked inputs to default value
            if K.ndim(mask) == K.ndim(inputs) - 1:
                mask = K.expand_dims(mask, axis=-1)
            assert K.ndim(mask) == K.ndim(inputs)
            inputs = masked_where(mask, inputs, self.default_value)

        # We will handle higher dimensioned inputs by converting them to 3-dimensions by expanding the batch dimension,
        # and then converting them back to output shape before returning.
        if input_ndim > 3:
            # Convert input to 3-D by expanding the batch dimension
            inputs = K.reshape(inputs, (-1, input_shape[-2], input_shape[-1]))

        # Begin: from keras.layers.pooling._Pooling1D
        #
        inputs = K.expand_dims(inputs, 2)   # add dummy last dimension
        outputs = self._pooling_function(inputs=inputs,
                                         pool_size=self.pool_size + (1,),
                                         strides=self.strides + (1,),
                                         padding=self.padding,
                                         data_format='channels_last')
        outputs = K.squeeze(outputs, 2)  # remove dummy last dimension
        #
        # End

        if input_ndim > 3:
            # convert back to appropriate shape
            outputs = K.reshape(outputs, self.compute_output_shape(input_shape))

        return outputs

    def get_config(self):
        config = {'strides': self.strides,
                  'pool_size': self.pool_size,
                  'padding': self.padding,
                  'default_value': self.default_value,
                  'terminate_mask': self.terminate_mask,
                  }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
# /


class KTextMaxPooling1D(_KTextPooling1D):
    """
    Max pooling for N-Dimensional Text data with word-embeddings.
    Pooling is done along the 'word dimension', which is the 2nd-last dimension.
    The features pooled are in the last dimension.
    """

    def __init__(self, pool_size=2, strides=None, padding='valid',
                 default_value=LARGE_NEG, terminate_mask=False,
                 **kwargs):
        super().__init__(pool_size, strides, padding,
                         default_value, terminate_mask,
                         **kwargs)
        return

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        output = K.pool2d(inputs, pool_size, strides,
                          padding, data_format, pool_mode='max')
        return output
# /


class KTextAveragePooling1D(_KTextPooling1D):
    """
    Warning: This probably does not handle Masking as desired.

    Average pooling for N-Dimensional Text data with word-embeddings.
    Pooling is done along the 'word dimension', which is the 2nd-last dimension.
    The features pooled are in the last dimension.
    """

    def __init__(self, pool_size=2, strides=None, padding='valid',
                 terminate_mask=False,
                 **kwargs):
        super().__init__(pool_size, strides, padding,
                         default_value=0.0, terminate_mask=terminate_mask,
                         **kwargs)
        return

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        output = K.pool2d(inputs, pool_size, strides,
                          padding, data_format, pool_mode='avg')

        # TODO: Compute the masked average.

        return output
# /


# =============================================================================
#   Functions
# =============================================================================


def masked_max(x, axis=1, mask=None, keepdims=False, default_value=-np.inf):
    if mask is None:
        return K.max(x, axis=axis, keepdims=keepdims)
    else:
        if K.ndim(mask) == K.ndim(x) - 1:
            mask = K.expand_dims(mask, axis=-1)
        assert K.ndim(mask) == K.ndim(x)

        return K.max(masked_where(mask, x, default_value), axis=axis, keepdims=keepdims)


def masked_min(x, axis=1, mask=None, keepdims=False, default_value=np.inf):
    if mask is None:
        return K.min(x, axis=axis, keepdims=keepdims)
    else:
        if K.ndim(mask) == K.ndim(x) - 1:
            mask = K.expand_dims(mask, axis=-1)
        assert K.ndim(mask) == K.ndim(x)

        return K.min(masked_where(mask, x, default_value), axis=axis, keepdims=keepdims)


def masked_avg(x, axis=1, keepdims=False, mask=None):

    if mask is None:
        return K.mean(x, axis=axis, keepdims=keepdims)
    else:
        if K.ndim(mask) == K.ndim(x) - 1:
            mask = K.expand_dims(mask, axis=-1)
        assert K.ndim(mask) == K.ndim(x)

        # mask is Boolean, so counts (sums) are Integers >= 0.
        # To avoid divide-by-zero ...
        # IF count is 0 THEN make it 1 because sum of (mask * x) will also be 0, and 0/1 = 0.
        counts = K.cast(K.maximum(1, K.sum(mask, axis=axis, keepdims=keepdims)), K.dtype(x))  # 'float32')

        return K.sum(mask * x, axis=axis, keepdims=keepdims) / counts
