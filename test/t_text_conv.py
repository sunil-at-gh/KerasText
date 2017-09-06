"""
Test for KTextConvolution
"""

import numpy as np

# noinspection PyPep8Naming
import keras.backend as K
from keras.utils import conv_utils
from keras.layers import Input, Embedding
from keras.models import Model

from kerastext import KTextConvolution, KTextGlobalMaxPooling, LARGE_NEG


VERBOSE = True

MAX_POOL = True


def pparray(name, a):
    print(name, 'shape', a.shape, '=')
    print(a)
    print()
    return


def create_wvecs(wvec_sz, vocab_sz):
    np.random.seed(123)
    wvecs = np.random.rand(vocab_sz + 1, wvec_sz).astype(np.float32)
    # Set pad values to 0
    # wvecs[0] = 0 ... not doing this tests the masking
    return wvecs


def create_input(input_shape, vocab_sz):
    x = np.random.randint(vocab_sz, size=input_shape, dtype=np.int32) + 1
    x = x.reshape((-1, input_shape[-1]))
    # Set random-sized suffixes to 0
    nz = np.random.randint(x.shape[1], size=x.shape[0])
    for i in range(x.shape[0]):
        x[i, nz[i]:] = 0
    return x.reshape(input_shape)


def get_wts(filter_width, wvec_sz, nbr_filters):
    # Shape as used in NTextConvolution.build() for `W`
    w_shape = (filter_width, wvec_sz, nbr_filters)
    # noinspection PyTypeChecker
    w = np.arange(1, np.prod(w_shape) + 1).astype(np.float32).reshape(w_shape)
    b = np.zeros((nbr_filters,), dtype=np.float32)
    return w, b


class NumpyTextEmbAndConvolution:
    def __init__(self, nbr_filters, filter_length, stride, input_shape, border_mode, wvec_sz):
        self.nbr_filters = nbr_filters
        self.filter_length = filter_length
        self.stride = stride
        self.input_shape = input_shape
        self.border_mode = border_mode

        # W.shape = (fil_len, wv_sz, n_fils)
        self.W_shape = (self.filter_length, wvec_sz, self.nbr_filters)
        self.W, self.bias = get_wts(self.filter_length, wvec_sz, self.nbr_filters)

        # Theano computes a Convolution, but TensorFlow computes a Cross-Correlation
        if K.backend() == 'theano':
            # Reverse the weights for a Convolution (compared to Cross-Correlation)
            self.W = self.W[::-1]

        return

    def compute_mask_np(self, input_mask):
        assert input_mask.ndim == len(self.input_shape)

        if self.border_mode == 'same':

            output_mask = input_mask

        elif self.border_mode == 'valid':

            output_length = input_mask.shape[-1] - self.filter_length + 1
            output_mask = input_mask[..., :output_length]

        else:  # 'full'

            # ... left-pad the input_mask with `input_mask[..., 0]` till it is of output_length.
            # This way, any convolutions at the right end that are completely over input_mask=False
            # will get output_mask=False. All other convolutions will get output_mask=True.

            delta_length = self.filter_length - 1

            delta_mask = np.repeat(input_mask[..., 0].reshape(input_mask.shape[:-1] + (1,)), delta_length, axis=-1)
            output_mask = np.concatenate([delta_mask, input_mask], axis=-1)

        if self.stride > 1:

            # In each row, select every stride'th element from output_mask, starting with element 0
            selector = np.arange(0, output_mask.shape[-1], self.stride)
            output_mask = output_mask[..., selector]

        return output_mask

    @staticmethod
    def global_max_pool(arr, mask):
        if mask is not None:
            if mask.ndim == arr.ndim - 1:
                mask = np.expand_dims(mask, -1)
            assert mask.ndim == arr.ndim

            arr = np.where(mask, arr, LARGE_NEG)

        return np.amax(arr, axis=-2)

    def __call__(self, input_arr, wvecs):
        """
        Computes result of Conv( Emb(input_arr, wvecs) )

        @param input_arr: Tensor of int32, each element is index into vocabulary, with 0 used for padding.
        @param wvecs: Word embeddings
        @return:
        """
        assert np.alltrue(input_arr.shape == self.input_shape)

        # Save n-dimensional shape of input
        input_nd_shape = input_arr.shape
        input_ndim = input_arr.ndim

        mask = np.not_equal(input_arr, 0)

        input_arr = wvecs[input_arr]

        # Force masked inputs to Zero

        input_mask = mask
        if input_mask.ndim == input_arr.ndim - 1:
            input_mask = np.expand_dims(input_mask, -1)

        input_arr = input_arr * input_mask

        # Convolution Padding

        pad_size = 0
        if self.border_mode == 'same':
            pad_size = self.filter_length // 2
        elif self.border_mode == 'full':
            pad_size = self.filter_length - 1

        if pad_size > 0:
            # noinspection PyTypeChecker
            pad_shape = input_arr.shape[:-2] + (pad_size, input_arr.shape[-1])
            input_arr = np.concatenate([np.zeros(pad_shape), input_arr, np.zeros(pad_shape)],
                                       axis=-2).astype(np.float32)

        # reshape input to 3D
        input_arr = input_arr.reshape((-1, input_arr.shape[-2], input_arr.shape[-1]))

        out_length = conv_utils.conv_output_length(input_nd_shape[-1],
                                                   self.filter_length,
                                                   self.border_mode,
                                                   self.stride)

        # output as 3D tensor
        out_shape = (input_arr.shape[0], out_length, self.nbr_filters)
        out_arr = np.zeros(out_shape, dtype=np.float32)

        for bi in range(out_shape[0]):  # batch dimension
            for j in range(out_length):
                s = j * self.stride
                e = s + self.filter_length
                #                              (fil_len, wv_sz) x (fil_len, wv_sz, n_fils)
                out_arr[bi, j] = np.tensordot(input_arr[bi, s:e], self.W, axes=([0, 1], [0, 1])) + self.bias

        # reshape output to n-dimensions
        out_arr = out_arr.reshape(input_nd_shape[:-1] + out_shape[1:])

        # Force masked output to Zero
        out_mask = self.compute_mask_np(mask)
        if out_mask.ndim == input_ndim:
            out_mask = np.expand_dims(out_mask, -1)
        assert out_mask.ndim == input_ndim + 1
        out_arr = out_arr * out_mask

        if MAX_POOL:
            out_arr = self.global_max_pool(out_arr, out_mask)

        return out_arr
# /


def build_model(wvecs, nbr_filters, filter_length, stride, input_shape, border_mode):
    vocab_sz, wvec_sz = wvecs.shape

    # Exclude batch dimension from input_shape
    input_tensor = Input(shape=input_shape[1:], dtype='int32', name='input_tensor')

    out = Embedding(vocab_sz, wvec_sz, weights=[wvecs], trainable=False, mask_zero=True)(input_tensor)

    # Assumes 3D input: (batch_sz, width, depth)
    conv_layer = KTextConvolution(filters=nbr_filters, kernel_size=filter_length, padding=border_mode,
                                  strides=stride,
                                  weights=get_wts(filter_length, wvec_sz, nbr_filters))

    out = conv_layer(out)

    if MAX_POOL:
        out = KTextGlobalMaxPooling()(out)

    mdl = Model(inputs=[input_tensor], outputs=out, name='conv_test')
    mdl.compile(optimizer='adadelta', loss='MSE')
    return mdl


def test():
    nbr_docs = 3
    nbr_sents = 4
    nbr_words = 5
    wv_sz = 2
    vocab_sz = 10
    input_shape = (nbr_docs, nbr_sents, nbr_words)

    inp_arr = create_input(input_shape, vocab_sz)
    wvecs = create_wvecs(wv_sz, vocab_sz)

    nbr_filters = 4
    filter_length = 3

    print()
    print('Test for text input of shape:', input_shape)
    print('  vocabulary size = {}, wvec_size = {}'.format(vocab_sz, wv_sz))
    print('  filter_length = {}, nbr_filters = {}'.format(filter_length, nbr_filters))
    if MAX_POOL:
        print('   Model ends in Global-Max-Pool')
    print()

    border_modes = ['valid', 'same']
    if K.backend() == 'theano':
        border_modes.append('full')

    for border_mode in border_modes:
        print('border_mode =', border_mode)
        for stride in [1, 2]:

            print('   stride =', stride)

            mdl = build_model(wvecs, nbr_filters, filter_length, stride, input_shape, border_mode)
            y_mdl = mdl.predict(inp_arr, batch_size=nbr_docs)

            np_conv = NumpyTextEmbAndConvolution(nbr_filters, filter_length, stride, input_shape, border_mode, wv_sz)
            y_np = np_conv(inp_arr, wvecs)

            mld_np_match = np.allclose(y_mdl, y_np)
            print('      y_mdl == y_np?', mld_np_match)
            if not mld_np_match and VERBOSE:
                print()
                pparray('inp_arr[0]', inp_arr[0])
                pparray('y_mdl[0]', y_mdl[0])
                pparray('y_np[0]', y_np[0])

            print()

    print()
    return


if __name__ == '__main__':

    # Run as: $> ../runk2.sh -m test.t_text_conv

    test()
