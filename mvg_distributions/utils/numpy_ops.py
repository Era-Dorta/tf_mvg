import math

import numpy as np
import tensorflow as tf
from scipy.ndimage import filters as fi
from scipy.sparse import lil_matrix
import scipy.stats


def get_np_replicated_identity_matrix_with_noise(width, height, limit=0.05, dtype=tf.float32):
    # Same as get_replicated_identity_matrix, but adds uniform noise to the non-diagonal terms
    out = get_np_replicated_identity_matrix(width=width, height=height, dtype=dtype)

    # For xavier noise add with : limit = np.sqrt(6.0 / (width + height))
    # Get uniform random noise
    noise = np.random.uniform(low=-limit, high=limit, size=out.shape)
    noise = noise.astype(dtype.as_numpy_dtype())

    # Add the noise to all elements that are not ones in the original array
    out += noise * (1 - out)
    return out


def get_np_gradient_ellipse(xy, width, height, img_width, img_height, base_color=np.array([255, 255, 255, 255]),
                            background_color=(0, 0, 0, 0)):
    ellipse_img = np.zeros((img_height, img_width, 4), dtype=np.uint8)
    # Input is total width, but for calculations it's easier to have width from center
    width = width * 0.5
    height = height * 0.5
    xy = np.array(xy, dtype=np.float)
    max_r = np.max([height, width])
    base_alpha = base_color[3:]
    base_color_rgb = base_color[:3]

    for i in range(img_height):
        for j in range(img_width):
            if (((i - xy[1]) / height) ** 2 + ((j - xy[0]) / width) ** 2) <= 1:
                # Normalized [0,1] distance from center of ellipse
                scale = np.sqrt((i - xy[1]) ** 2 + (j - xy[0]) ** 2) / max_r
                color = np.concatenate([scale * base_color_rgb, base_alpha])
                color = color.clip(0, 255)
                ellipse_img[i, j, :] = color
            else:
                ellipse_img[i, j, :] = background_color
    return ellipse_img


def get_np_replicated_identity_matrix(width, height, dtype=tf.float32):
    # Examples with (3, 5) and (3, 2)
    # 1 0 0 1 0        1 0
    # 0 1 0 0 1        0 1
    # 0 0 1 0 0        1 0
    transpose = False
    if width > height:
        width, height = height, width
        transpose = True

    np_type = dtype.as_numpy_dtype()
    out = np.zeros((width, height), dtype=np_type)
    n_repeat = int(math.ceil(float(height) / width))

    for i in range(n_repeat):
        out += np.eye(width, height, i * width, dtype=np_type)

    if transpose:
        out = out.transpose()
    return out


def get_np_1d_gaussian_kernel(kernlen=21, sigma=3.0, location=None):
    """Returns a 2D Gaussian kernel array."""
    if location is None:
        # Middle of the kernel
        location = kernlen // 2

    # create nxn zeros
    inp = np.zeros(kernlen)
    # set element at location to one
    inp[location] = 1
    # gaussian-smooth the one, resulting in a gaussian filter mask, with mode constant the values beyond
    # the border are considered to be zero
    return fi.gaussian_filter(inp, sigma, mode='constant')


def get_np_2d_gaussian_kernel(kernlen=21, sigma=3.0, location=None):
    """Returns a 2D Gaussian kernel array."""
    if location is None:
        # Middle of the img kernel
        location = [[kernlen // 2], [kernlen // 2]]

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at location to one
    inp[location] = 1
    # gaussian-smooth the one, resulting in a gaussian filter mask, with mode constant the values beyond
    # the border are considered to be zero
    return fi.gaussian_filter(inp, sigma, mode='constant')


def get_np_gaussian_kernel(kernlen=21, sigma=3, channels=1):
    """ Returns a Gaussian filter [kernlen, kernlen, channels, 1] to be used with convolutions """
    interval = (2 * sigma + 1.) / kernlen
    x = np.linspace(-sigma - interval / 2., sigma + interval / 2., kernlen + 1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter


def get_np_line_kernel(kernlen=21, sigma=3.0, is_horizontal=True):
    """Returns a line kernel array."""
    # The kernel will have a cross shape, with the center at the center of the image
    # with exponential decaying weights. If horizontal with positive horizontal line and
    # negative vertical, and otherwise if vertical.
    kernel = np.zeros((kernlen, kernlen))

    half_lenght = kernlen // 2

    j = half_lenght
    for i in range(kernlen):
        value = math.exp(-0.5 * float(j ** 2) / sigma)

        if is_horizontal:
            value *= -1.0

        kernel[i, half_lenght] = value
        kernel[half_lenght, i] = -kernel[i, half_lenght]

        if i < half_lenght:
            j -= 1
        else:
            j += 1

    kernel[half_lenght, half_lenght] = 1.0

    return kernel


def get_np_line_smooth_kernel(kernlen=21, sigma=3.0, is_horizontal=True):
    """Returns a line kernel array.
       It's a negative Gaussian kernel with a positive row or column
    """

    kernel = get_np_2d_gaussian_kernel(kernlen, sigma)

    scale = -1.0
    for i in range(kernlen):
        if is_horizontal:
            kernel[i, :] *= scale
        else:
            kernel[:, i] *= scale
        scale *= -1.0
    return kernel


def np_make_matrix_from_kernel_list(kernels, img_size, make_sparse=False):
    """
    :param kernels: a list or numpy array of [img width * img height, filter width, filter height]
    :param img_size: an integer with the value of img width
    :return: a matrix of  [img width * img height, img width * img height] with the kernels centered on the diagonal
    """
    size = img_size * img_size

    filters_shape = kernels[0].shape
    filters_w_half = (kernels[0].shape[0] - 1) // 2
    filteres_h_half = (kernels[0].shape[1] - 1) // 2

    if make_sparse:
        # Create matrix in format that is fast for building incrementally with rows
        matrix = lil_matrix((size, size), dtype=kernels[0].dtype)
    else:
        matrix = np.zeros((size, size), dtype=kernels[0].dtype)

    k = 0
    for i in range(0, img_size):
        for j in range(0, img_size):
            filter_ij = kernels[k]
            padding_w0 = i - filters_w_half
            padding_w1 = img_size - (filters_shape[0] + padding_w0)

            padding_h0 = j - filteres_h_half
            padding_h1 = img_size - (filters_shape[1] + padding_h0)

            if padding_w0 < 0:
                filter_ij = filter_ij[np.abs(padding_w0):filters_shape[0], :]
                padding_w0 = 0

            if padding_w1 < 0:
                filter_ij = filter_ij[0:filters_shape[0] - np.abs(padding_w1), :]
                padding_w1 = 0

            if padding_h0 < 0:
                filter_ij = filter_ij[:, np.abs(padding_h0):filters_shape[1]]
                padding_h0 = 0

            if padding_h1 < 0:
                filter_ij = filter_ij[:, 0:filters_shape[1] - np.abs(padding_h1)]
                padding_h1 = 0

            padding = np.array([[padding_w0, padding_w1], [padding_h0, padding_h1]])
            filter_ij = np.pad(filter_ij, padding, mode="constant")
            filter_ij = np.reshape(filter_ij, -1)

            matrix[k] = filter_ij
            k += 1
    return matrix


def np_make_matrix_from_kernel(kernel, size):
    # Flat index of the center of the kernel
    center_i = np.ravel_multi_index(kernel.shape // np.array([2]), kernel.shape)

    # Flatten kernel image
    kernel = kernel.reshape(-1)

    # Move center of kernel to beginning of vector
    kernel = np.concatenate([kernel[-(size - center_i):], kernel[:center_i]])

    matrix = np.zeros((size, size))

    matrix[0, :] = kernel
    for i in range(1, size):
        # Move the beginning of the kernel vector so that it ends
        # up in the diagonal of the covariance matrix
        matrix[i, :] = np.concatenate([kernel[-i:], kernel[:size - i]])

    return matrix


def np_make_matrix_from_1d_kernel_fnc(kernel_fnc, size):
    matrix = np.zeros((size, size))

    for i in range(0, size):
        matrix[i, :] = kernel_fnc(i)

    return matrix


def np_make_matrix_from_2d_kernel_fnc(kernel_fnc, size):
    matrix = np.zeros((size, size))
    img_size = int(np.sqrt(size))

    for i in range(0, size):
        kernel = kernel_fnc(np.unravel_index([i], (img_size, img_size)))
        matrix[i, :] = kernel.reshape(-1)

    return matrix


def np_is_positive_definite(m):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(m)
        return True
    except np.linalg.LinAlgError:
        return False


def get_np_nearest_positive_definite_matrix(m):
    """Find the nearest positive-definite matrix to input

    N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix" (1988)
    """

    if np_is_positive_definite(m):
        return m

    # Find closest symmetric matrix of m
    b = (m + m.T) / 2.0

    # Compute symmetric polar factor h
    _, s, v = np.linalg.svd(b)
    h = np.dot(v.T, np.dot(np.diag(s), v))

    # Nearest matrix that is positive-definite of m is the mean between b and h
    m2 = (b + h) / 2.0

    # Ensure symmetry in result
    m3 = (m2 + m2.T) / 2.0

    if np_is_positive_definite(m3):
        return m3

    # Due to numerical instabilities the previous method can fail in practice. Force the matrix to be
    # positive-definite by slowly increasing the diagonal values using as scale the most negative eigenvalue
    diag_ind = np.diag_indices_from(m3)
    k = 1
    while not np_is_positive_definite(m3):
        min_eig_val = np.min(np.real(np.linalg.eigvals(m3)))
        spacing = np.spacing(min_eig_val)
        m3[diag_ind] += -min_eig_val * k ** 2 + spacing
        k += 1

    return m3
