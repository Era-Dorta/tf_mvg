import tensorflow as tf
import numpy as np


def unpooling2d_2x2_zero_filled(x):
    x = tf.convert_to_tensor(x)
    out = tf.concat([x, tf.zeros_like(x)], 3)
    out = tf.concat([out, tf.zeros_like(out)], 2)

    sh = x.get_shape().as_list()
    if None not in sh[1:]:
        out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
        return tf.reshape(out, out_size)
    else:
        shv = tf.shape(x)
        ret = tf.reshape(out, tf.stack([-1, shv[1] * 2, shv[2] * 2, sh[3]]))
    return ret


def unpooling2d_zero_filled(x, stride, unpool_mat=None, data_format='channels_last'):
    """
    Unpool the input with a fixed matrix to perform kronecker product with.
    Args:
        x (tf.Tensor): a 4D image tensor
        stride: int or (h, w) tuple
        unpool_mat: a tf.Tensor or np.ndarray 2D matrix with size=shape.
            If is None, will use a matrix with 1 at top-left corner.
    Returns:
        tf.Tensor: a 4D image tensor.
    """
    x = tf.convert_to_tensor(x)
    stride = _shape2d(stride)

    output_shape = _StaticDynamicShape(x)
    output_shape.apply(1 if data_format == 'channels_last' else 2, lambda x: x * stride[0])
    output_shape.apply(2 if data_format == 'channels_last' else 3, lambda x: x * stride[1])

    # a faster implementation for this special case
    if stride[0] == 2 and stride[1] == 2 and unpool_mat is None and data_format == 'channels_last':
        ret = unpooling2d_2x2_zero_filled(x)
    else:
        # check unpool_mat
        if unpool_mat is None:
            mat = np.zeros(stride, dtype='float32')
            mat[0][0] = 1
            unpool_mat = tf.constant(mat, name='unpool_mat')
        elif isinstance(unpool_mat, np.ndarray):
            unpool_mat = tf.constant(unpool_mat, name='unpool_mat')
        assert unpool_mat.shape.as_list() == list(stride)

        if data_format == 'channels_last':
            x = tf.transpose(x, [0, 3, 1, 2])
        # perform a tensor-matrix kronecker product
        x = tf.expand_dims(x, -1)  # bchwx1
        mat = tf.expand_dims(unpool_mat, 0)  # 1xshxsw
        ret = tf.tensordot(x, mat, axes=1)  # bxcxhxwxshxsw

        if data_format == 'channels_last':
            ret = tf.transpose(ret, [0, 2, 4, 3, 5, 1])
        else:
            ret = tf.transpose(ret, [0, 1, 2, 4, 3, 5])

        shape3_dyn = [output_shape.get_dynamic(k) for k in range(1, 4)]
        ret = tf.reshape(ret, tf.stack([-1] + shape3_dyn))

    ret.set_shape(tf.TensorShape(output_shape.get_static()))
    return ret


def _shape2d(a):
    """
    Ensure a 2D shape.
    Args:
        a: a int or tuple/list of length 2
    Returns:
        list: of length 2. if ``a`` is a int, return ``[a, a]``.
    """
    if type(a) == int:
        return [a, a]
    if isinstance(a, (list, tuple)):
        assert len(a) == 2
        return list(a)
    raise RuntimeError("Illegal shape: {}".format(a))


def _DynamicLazyAxis(shape, idx):
    return lambda: shape[idx]


def _StaticLazyAxis(dim):
    return lambda: dim


class _StaticDynamicShape(object):
    def __init__(self, tensor):
        assert isinstance(tensor, tf.Tensor), tensor
        ndims = tensor.shape.ndims
        self.static = tensor.shape.as_list()
        if tensor.shape.is_fully_defined():
            self.dynamic = self.static[:]
        else:
            dynamic = tf.shape(tensor)
            self.dynamic = [_DynamicLazyAxis(dynamic, k) for k in range(ndims)]

        for k in range(ndims):
            if self.static[k] is not None:
                self.dynamic[k] = _StaticLazyAxis(self.static[k])

    def apply(self, axis, f):
        if self.static[axis] is not None:
            try:
                st = f(self.static[axis])
                self.static[axis] = st
                self.dynamic[axis] = _StaticLazyAxis(st)
                return
            except TypeError:
                pass
        self.static[axis] = None
        dyn = self.dynamic[axis]
        self.dynamic[axis] = lambda: f(dyn())

    def get_static(self):
        return self.static

    @property
    def ndims(self):
        return len(self.static)

    def get_dynamic(self, axis=None):
        if axis is None:
            return [self.dynamic[k]() for k in range(self.ndims)]
        return self.dynamic[axis]()
