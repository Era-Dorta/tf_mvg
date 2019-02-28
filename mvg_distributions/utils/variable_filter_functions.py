import tensorflow as tf
import numpy as np
from enum import Enum


class _LinearCombinationMethod(Enum):
    SIMPLE = 0
    EINSUM = 1
    MATMUL = 2


def _conv2d_combination_filter(inputs, filters, strides, padding, dilation_rate=None, **kwargs):
    tf.assert_rank(inputs, 4)
    tf.assert_rank(filters, 5)
    if filters.shape[3].value != 1 or filters.shape[4].value != 1:
        raise NotImplementedError("Only one channel is supported {}".format(filters))
    filters = tf.squeeze(filters, axis=(3, 4))  # Remove the in/out channels as they are one

    filters = tf.transpose(filters, perm=[1, 2, 0])  # [filter width, filter height, num filters]
    filters = tf.expand_dims(filters, axis=2)  # [filter width, filter height, 1, num filters]

    # Do the convolution with each of the filters: [batch size, img width, img height, num filters]
    if dilation_rate is not None:
        assert np.alltrue(np.array(strides) == 1)
        return tf.nn.convolution(inputs, filters, padding=padding, dilation_rate=dilation_rate, **kwargs)
    else:
        return tf.nn.conv2d(inputs, filters, strides, padding, **kwargs)


def _linear_combination_simple(filtered, alpha, axis=3):
    # Simple linear combination, multiply element wise and sum over axis
    assert axis in [3, 4]
    if axis == 4:
        alpha = tf.expand_dims(alpha, axis=1)
    filtered_combined = tf.multiply(filtered, alpha)
    return tf.reduce_sum(filtered_combined, axis=axis, keep_dims=True)


def _linear_combination_einsum(filtered, alpha, axis=3):
    # Linear combination with einstein summation, same as _linear_combination_simple but it avoids
    # creating the intermediate tensor after multiply of [batch size, <num_samples>, img width, img height, num filters]
    assert axis in [3, 4]

    if axis == 4:
        filtered_combined = tf.einsum('bijf,bsijf->bsij', alpha, filtered)
    else:
        filtered_combined = tf.einsum('bijf,bijf->bij', alpha, filtered)
    return tf.expand_dims(filtered_combined, axis=axis)


def _linear_combination_matmul(filtered, alpha, axis=3):
    # Linear combination with matrix multiplication, the same as einstein summation but it avoids a few
    # transpose and slice operations
    assert axis in [3, 4]
    if axis == 3:
        filtered = tf.expand_dims(filtered, axis=1)
    alpha = tf.expand_dims(alpha, axis=3)  # [batch size, img width, img height, 1, num filters]
    filtered = tf.transpose(filtered, [0, 2, 3, 4, 1])  # [batch size, img width, img height, num filters, num samples]
    filtered_combined = tf.matmul(alpha, filtered)  # [batch size, img width, img height, 1, num samples]

    # [batch size, num samples, img width, img height, 1]
    filtered_combined = tf.transpose(filtered_combined, [0, 4, 1, 2, 3])
    if axis == 3:  # Remove sample dim if filtered didn't have it
        filtered_combined = tf.squeeze(filtered_combined, axis=1)
    return filtered_combined


def _linear_combination(filtered, alpha, op_method=_LinearCombinationMethod.SIMPLE):
    # Do the linear combination with [batch size, img width, img height, num filters]
    # or with [batch size, num_samples, img width, img height, num filters]
    # It multiplies filtered and alpha element wise and sums over axis num filters
    """
    :param filtered: filtered input [batch size, <num samples>, img width, img height, num filters]
    :param alpha: weights for the linear combination [batch size, img width, img height, num filters]
    :return: the linear combination of filtered and alpha [batch size, <num samples>, img width, img height]
    """
    if filtered.shape.ndims == 5:
        axis = 4
    else:
        axis = 3

    if op_method is _LinearCombinationMethod.SIMPLE:
        return _linear_combination_simple(filtered, alpha, axis)
    elif op_method is _LinearCombinationMethod.EINSUM:
        return _linear_combination_einsum(filtered, alpha, axis)
    elif op_method is _LinearCombinationMethod.MATMUL:
        return _linear_combination_matmul(filtered, alpha, axis)
    else:
        raise RuntimeError("Unknown method {}".format(op_method))


def conv2d_linear_combination_filters(inputs, filters, alpha, strides=(1, 1, 1, 1), padding="SAME", name=None,
                                      **kwargs):
    """
    :param inputs: [batch size, img width, img height, num channels]
    :param alpha: [batch size, img width, img height, num filters]
    :param filters: [num filters, filter width, filter height, in channels, out channels]
    :param name: name for the operation
    :return:
    """
    with tf.name_scope(name, default_name="conv2d_linear_combination_filters"):
        inputs = tf.convert_to_tensor(inputs)
        filters = tf.convert_to_tensor(filters)
        alpha = tf.convert_to_tensor(alpha)

        inputs.shape[0:3].assert_is_compatible_with(alpha.shape[0:3])
        filters.shape[0].assert_is_compatible_with(alpha.shape[3])

        filtered = _conv2d_combination_filter(inputs, filters, strides, padding, **kwargs)

        return _linear_combination(filtered, alpha)


def conv2d_samples_linear_combination_filters(inputs, filters, alpha, strides=(1, 1, 1, 1), padding="SAME", name=None,
                                              **kwargs):
    """
    :param inputs: [batch size, num_samples, img width, img height, num channels]
    :param filters: [num filters, filter width, filter height, in channels, out channels]
    :param alpha: [batch size, img width, img height, num filters]
    :param name: name for the operation
    :return:
    """
    with tf.name_scope(name, default_name="conv2d_linear_combination_filters"):
        inputs = tf.convert_to_tensor(inputs)
        filters = tf.convert_to_tensor(filters)
        alpha = tf.convert_to_tensor(alpha)
        tf.assert_rank(inputs, 5)

        # Reshape to [batch size * num_samples, img width, img height, num channels]
        in_shape = tf.shape(inputs)
        new_shape = tf.concat([[tf.multiply(in_shape[0], in_shape[1])], in_shape[2:5]], axis=0)
        inputs = tf.reshape(inputs, new_shape)

        inputs.shape[1:3].assert_is_compatible_with(alpha.shape[1:3])
        filters.shape[0].assert_is_compatible_with(alpha.shape[3])

        filtered = _conv2d_combination_filter(inputs, filters, strides, padding, **kwargs)

        # Reshape to [batch size, num_samples, img width, img height, num basis]
        filtered_shape = tf.concat([in_shape[0:4], [tf.shape(alpha)[-1]]], axis=0)
        filtered = tf.reshape(filtered, filtered_shape)
        return _linear_combination(filtered, alpha)
