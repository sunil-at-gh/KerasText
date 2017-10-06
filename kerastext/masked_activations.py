"""
Masked versions of some activation functions.
"""

# noinspection PyPep8Naming
import keras.backend as K
import keras.activations

from .pooling import masked_max
from .utils import masked_where


def masked_softmax(x, mask=None, axis=-1):
    """
    Masked version of softmax, where the masked-out values (where Mask value is False) return 0.
    :param x: a tensor
    :param mask: a boolean mask
    :param int axis: axis along which to perform softmax
    :return: tensor of same shape as x
    """
    if mask is None:
        return keras.activations.softmax(x, axis)
    elif K.ndim(x) <= 1:
        raise ValueError('Cannot apply softmax to a tensor that is 1D or lower. Got ndim = ' + str(K.ndim(x)))

    if K.ndim(mask) == K.ndim(x) - 1:
        mask = K.expand_dims(mask, axis=-1)
    assert K.ndim(mask) == K.ndim(x)

    # For better stability, compute as:
    # e_x = exp(x - x.max(axis=1, keepdims=True))
    # out = e_x / e_x.sum(axis=1, keepdims=True)

    x_max = masked_max(x, mask=mask, axis=axis, keepdims=True)
    e_x = masked_where(mask, K.exp(x - x_max), 0)
    s = K.sum(e_x, axis=axis, keepdims=True)
    # prevent divide-by-zero
    return masked_where(mask, e_x / s, 0)
