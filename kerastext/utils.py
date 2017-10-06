"""
General Utility and other functions.
Mostly platform dependent code.
"""

# noinspection PyPep8Naming
import keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf


def force_masked_to_zero(x, mask, x_ndim=None):
    """
    Return a copy of tensor where the masked values are forced to zero.

    :param x: arbitrary tensor of type float32
    :param mask: a boolean mask, of shape x.shape[:-1] or x.shape.
    :param x_ndim: integer or expression giving number of dimensions in x
    :return:
    """
    if mask is None:
        return x

    if x_ndim is None:
        x_ndim = K.ndim(x)

    if K.ndim(mask) == x_ndim - 1:
        mask = K.expand_dims(mask, axis=-1)
    assert K.ndim(mask) == x_ndim

    if K.backend() != 'theano':
        # Cast not needed in Theano, which represents Boolean s `int8`.
        mask = K.cast(mask, 'float32')

    return mask * x


def gather_from_last_axis(x, indices):
    if K.backend() == 'theano':
        return x[..., indices]
    elif K.backend() == 'tensorflow':
        return tf.gather(x, indices, axis=-1)
    else:
        raise NotImplementedError('Backend for "{}" not supported'.format(K.backend()))


def masked_where(mask, x, default):
    """
    :param mask: Of same ndim as x. Last dimension may be 1.
    :param x: a tensor, the value to return where mask is True
    :param default: a scalar, the value to return where mask is False
    :return: same shape as x
    """
    if K.backend() == 'theano':
        return K.switch(mask, x, default)

    elif K.backend() == 'tensorflow':
        def tile_mask():
            return tf.tile(mask, tf.concat([tf.ones([tf.rank(x) - 1], tf.int32),
                                           [tf.shape(x)[-1]]],
                                           axis=0))

        def ident():
            return mask

        # tf.where() requires shapes of all args to be the same

        tiled_mask = tf.cond(tf.equal(tf.shape(mask)[-1], 1), tile_mask, ident)

        tiled_default = tf.zeros_like(x) + default

        return tf.where(tiled_mask, x, tiled_default)

    else:
        raise NotImplementedError('Backend for "{}" not supported'.format(K.backend()))
