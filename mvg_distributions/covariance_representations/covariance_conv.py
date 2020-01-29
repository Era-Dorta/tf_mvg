import numpy as np
import tensorflow as tf

from mvg_distributions.covariance_representations.covariance_matrix import Covariance, SampleMethod
from mvg_distributions.utils.variable_filter_functions import conv2d_samples_linear_combination_filters
from mvg_distributions.utils.unpooling import unpooling2d_zero_filled
import mvg_distributions.utils.numpy_ops as np_ops
from scipy.sparse.linalg import spsolve_triangular
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix, isspmatrix_csr, SparseEfficiencyWarning
from scipy.sparse import eye as sparse_eye
from scipy.linalg import LinAlgError
from warnings import warn
import os
from tqdm import tqdm


class PrecisionConvFilters(Covariance):
    def __init__(self, weights_precision, filters_precision, sample_shape, **kwargs):
        super().__init__(**kwargs)
        self.sample_shape = sample_shape
        if isinstance(self.sample_shape, np.ndarray):
            self.sample_shape = tf.TensorShape(self.sample_shape)

        if isinstance(self.sample_shape, tf.TensorShape) and self.sample_shape.is_fully_defined():
            assert self.sample_shape[3].value == 1, "Only one channel is supported"
            num_pixels = self.sample_shape[1] * self.sample_shape[2]
            self._matrix_shape = tf.TensorShape([self.sample_shape[0], num_pixels, num_pixels])
        else:
            num_pixels = self.sample_shape[1] * self.sample_shape[2]
            self._matrix_shape = tf.stack([self.sample_shape[0], num_pixels, num_pixels], axis=0)

        self.weights_precision = weights_precision
        assert self.weights_precision is not None

        if filters_precision is None:
            self.filters_precision = self._id_filters(self.weights_precision)
        else:
            self.filters_precision = filters_precision

        assert self.filters_precision is not None

        self._diag_sqrt_covariance = None
        self._diag_sqrt_precision = None
        self._recons_filters_precision = None

    def _get_epsilon_flat_shape(self, num_samples):
        return tf.stack([self.sample_shape[0], num_samples, tf.reduce_prod(self.sample_shape[1:])], axis=0)

    def _get_epsilon(self, num_samples, epsilon, seed=None):
        epsilon_shape = self._get_epsilon_flat_shape(num_samples)
        if epsilon is None:
            epsilon = self._build_epsilon(epsilon_shape, seed=seed)
        if epsilon.shape.ndims + 1 == epsilon_shape.shape[0].value:
            epsilon = tf.expand_dims(epsilon, 1)  # Epsilon should be [batch dim, 1, ...]
        return epsilon

    def _get_epsilon_img_shape(self, num_samples):
        return tf.concat([self.sample_shape[0:1], [num_samples], self.sample_shape[1:]], axis=0)

    def _get_epsilon_5_dim(self, num_samples, epsilon, seed=None):
        epsilon_shape = self._get_epsilon_img_shape(num_samples)
        if epsilon is None:
            epsilon = self._build_epsilon(epsilon_shape, seed=seed)
        if epsilon.shape.ndims == 2 or epsilon.shape.ndims == 3:
            # Input is flatten epsilon, reshape to 4 or 5 dim
            flat_shape = tf.shape(epsilon)
            dim_4_5_shape = tf.concat([flat_shape[0:epsilon.shape.ndims - 1], self.sample_shape[1:]], axis=0)
            epsilon = tf.reshape(epsilon, dim_4_5_shape)
        if epsilon.shape.ndims + 1 == epsilon_shape.shape[0].value:
            epsilon = tf.expand_dims(epsilon, 1)  # Epsilon should be [batch dim, 1, ...]
        return epsilon

    def _flatten_keep_sample_dim(self, inputs):
        if inputs.shape.ndims == 3:
            return inputs
        elif inputs.shape.ndims == 5:
            inputs_shape = tf.shape(inputs)
            flat_shape = tf.concat([inputs_shape[0:2], [tf.reduce_prod(self.sample_shape[1:])]], axis=0)
            return tf.reshape(inputs, shape=flat_shape)
        else:
            raise ValueError("Invalid number of dimensions {}".format(inputs.shape.ndims))

    def x_precision_x(self, x, mean_batch=False, no_gradients=True, **kwargs):
        if no_gradients == False:
            raise NotImplementedError("")
        # x shape should be [batch dim, num features]
        with tf.name_scope("x_precision_x"):
            # x.shape.assert_is_compatible_with(self.precision.shape[0:2])
            x_precision = self.whiten_x(x=x, flatten_output=True)
            if x_precision.shape.ndims == 2:
                x_precision = tf.expand_dims(x_precision, axis=1)
            squared_error = tf.multiply(x_precision, x_precision)
            squared_error = tf.reduce_sum(squared_error, axis=2)  # Error per sample
            if squared_error.shape[1].value == 1:
                squared_error = tf.squeeze(squared_error, axis=1, name="x_precision_x")  # Remove sample dim

            if mean_batch:
                squared_error = tf.reduce_mean(squared_error, axis=0, name="mean_x_precision_x")
            return squared_error

    @staticmethod
    def _id_filters(weights_precision):
        """
        Create filters that correspond to a basis that is identity
        :param weights_precision: [b, w, h, fs**2]
        :return: [fs**2, fs, fs, 1, 1]
        """
        with tf.name_scope('id_filters'):
            num_ch = 1
            filter_wh = weights_precision.shape[-1].value
            filter_size = int(np.sqrt(filter_wh))
            identity_basis = tf.eye(num_rows=filter_wh)
            return tf.reshape(identity_basis, (filter_wh, filter_size, filter_size, num_ch, num_ch))

    @property
    def diag_sqrt_covariance(self):
        if self._diag_sqrt_covariance is None:
            self._diag_sqrt_covariance = self._build_diag_sqrt_covariance()
        return self._diag_sqrt_covariance

    def _build_diag_sqrt_covariance(self):
        with tf.name_scope("DiagSqrtCovariance"):
            return tf.matrix_diag_part(self.sqrt_covariance, name="diag_sqrt_covariance")

    @property
    def diag_sqrt_precision(self):
        if self._diag_sqrt_precision is None:
            self._diag_sqrt_precision = self._build_diag_sqrt_precision()
        return self._diag_sqrt_precision

    def _build_diag_sqrt_precision(self):
        with tf.name_scope("DiagSqrtPrecision"):
            return self._build_diag_from_weights(name="diag_sqrt_precision")

    @property
    def recons_filters_precision(self):
        if self._recons_filters_precision is None:
            self._recons_filters_precision = self._reconstruct_basis(weights=self.weights_precision,
                                                                     basis=self.filters_precision)
        return self._recons_filters_precision

    def _build_diag_from_weights(self, name="diag_from_weights"):
        weights = self.weights_precision
        filters = self.filters_precision

        filters.shape.assert_is_fully_defined()
        filters_shape = filters.shape.as_list()
        center_i = (np.array(filters_shape[1:3]) - 1) // 2
        diag_basis = filters[:, center_i[0], center_i[1]]

        if weights.shape.is_fully_defined():
            w_flat_shape = tf.TensorShape([weights.shape[0], weights.shape[1] * weights.shape[2], weights.shape[3]])
        else:
            weights_shape = tf.shape(weights)
            w_flat_shape = tf.stack([weights_shape[0], weights_shape[1] * weights_shape[2], weights_shape[3]])
        weights_flat = tf.reshape(weights, shape=w_flat_shape)

        # [Batch dim (1 for broadcasting), filter size, num filters]
        diag_basis = tf.reshape(diag_basis, shape=(1, 1, filters_shape[0]))

        diag_covariance = tf.multiply(weights_flat, diag_basis)
        return tf.reduce_sum(diag_covariance, axis=2, name=name)

    def _build_covariance(self):
        with tf.name_scope("Covariance"):
            return self._inverse_covariance_or_precision()

    def _build_precision(self):
        with tf.name_scope("Precision"):
            return tf.matmul(self.sqrt_precision, self.sqrt_precision, transpose_b=True, name="precision")

    def _build_sqrt_covariance(self):
        with tf.name_scope("Covariance_Sqrt"):
            return tf.matrix_inverse(self.sqrt_precision, name="covariance_sqrt")

    def _build_sqrt_precision(self):
        with tf.name_scope("Precision_Sqrt"):
            return self._build_matrix_from_basis(weights=self.weights_precision, basis=self.filters_precision,
                                                 name="precision_sqrt")

    def _reconstruct_basis(self, weights, basis, name="reconstruct_basis"):
        """
        :param basis: [num basis, filter width, filter height, in channels, out channels]
        :param weights: [batch size, img width, img height, num filters]
        :return:
        """
        with tf.name_scope(name):
            basis.shape[3:5].assert_is_fully_defined()
            assert basis.shape[3] == 1 and basis.shape[4] == 1, "Only on channel is supported"

            basis = tf.squeeze(basis, axis=[3, 4])  # [num basis, filter width, filter height]

            if basis.shape.is_fully_defined():
                basis_flat_shape = tf.TensorShape([basis.shape[0], basis.shape[1] * basis.shape[2]])
            else:
                basis_shape = tf.shape(basis)
                basis_flat_shape = tf.stack([basis_shape[0], basis_shape[1] * basis_shape[2]])

            if weights.shape.is_fully_defined():
                weights_shape = weights.shape
                weights_flat_shape = tf.TensorShape(
                    [weights.shape[0], weights.shape[1] * weights.shape[2], weights.shape[3]])
            else:
                weights_shape = tf.shape(weights)
                weights_flat_shape = tf.stack([weights_shape[0], weights_shape[1] * weights_shape[2], weights_shape[3]])

            basis_flat = tf.reshape(basis, shape=basis_flat_shape)  # [num basis, filter width * filter height]

            # [batch size, num basis, filter width * filter height]
            basis_flat = tf.expand_dims(basis_flat, axis=0)
            basis_flat = tf.tile(basis_flat, multiples=tf.stack([weights_shape[0], 1, 1]))

            # [batch size, img width * img height, num basis]
            weights_flat = tf.reshape(weights, shape=weights_flat_shape)

            # [batch size, img width * img height, filter width * filter height]
            reconstructed_basis = tf.matmul(weights_flat, basis_flat)
            output_shape = [weights.shape[0], weights.shape[1] * weights.shape[2], basis.shape[1] * basis.shape[2]]
            reconstructed_basis.set_shape(output_shape)
            return reconstructed_basis

    def _build_matrix_from_basis(self, weights, basis, name=None):
        """
        :param basis: [num basis, filter width, filter height, in channels, out channels]
        :param weights: [batch size, img width, img height, num filters]
        :return:
        """
        with tf.name_scope(name, default_name="build_matrix_from_basis"):
            # [batch size, img width * img height, num filters]
            filters = self.recons_filters_precision

            filters_shape, img_h, img_w = self._compute_shapes_for_filter_matrix(basis, weights)

            # [batch size, img width * img height, filter width, filter height]
            filters = tf.reshape(filters, shape=filters_shape)

            filters_shape, filteres_h_half, filters_w_half = self._compute_shapes_for_single_filter(filters)

            matrix = list()

            k = 0
            for i in range(0, img_w):
                for j in range(0, img_h):
                    filter_ij = filters[:, k]
                    padding_w0 = i - filters_w_half
                    padding_w1 = img_w - (filters_shape[2].value + padding_w0)

                    padding_h0 = j - filteres_h_half
                    padding_h1 = img_h - (filters_shape[3].value + padding_h0)

                    if padding_w0 < 0:
                        filter_ij = filter_ij[:, np.abs(padding_w0):filters_shape[2].value, :]
                        padding_w0 = 0

                    if padding_w1 < 0:
                        filter_ij = filter_ij[:, 0:filters_shape[2].value - np.abs(padding_w1), :]
                        padding_w1 = 0

                    if padding_h0 < 0:
                        filter_ij = filter_ij[:, :, np.abs(padding_h0):filters_shape[3].value]
                        padding_h0 = 0

                    if padding_h1 < 0:
                        filter_ij = filter_ij[:, :, 0:filters_shape[3].value - np.abs(padding_h1)]
                        padding_h1 = 0

                    padding = tf.constant([[0, 0], [padding_w0, padding_w1], [padding_h0, padding_h1]])
                    filter_ij = tf.pad(filter_ij, paddings=padding)
                    filter_ij = tf.layers.flatten(filter_ij)

                    matrix.append(filter_ij)
                    k += 1

            matrix = tf.stack(matrix, axis=1)
            # The convolutions are computing the transpose matrix multiplication of the inputs
            # x M^T, to get equivalent results between the convolutions and the matrix, transpose here
            return tf.matrix_transpose(matrix)

    def _compute_shapes_for_single_filter(self, filters):
        if filters.shape[1:].is_fully_defined():
            filters_shape = filters.shape
            assert filters_shape[2].value / 2.0 != 0, "Filter width must be an odd number"
            assert filters_shape[3].value / 2.0 != 0, "Filter height must be an odd number"
            # Half of the filter not counting the central pixel
            filters_w_half = (filters_shape[2].value - 1) // 2
            filteres_h_half = (filters_shape[3].value - 1) // 2
        else:
            filters_shape = tf.shape(filters)
            filters_w_half = (filters_shape[2] - 1) // 2
            filteres_h_half = (filters_shape[3] - 1) // 2

        return filters_shape, filteres_h_half, filters_w_half

    def _compute_shapes_for_filter_matrix(self, basis, weights):
        if basis.shape.is_fully_defined() and weights.shape[1:].is_fully_defined():
            weights_shape = weights.shape.as_list()
            if weights_shape[0] is None:
                batch_size = -1
            else:
                batch_size = weights_shape[0]
            filters_shape = [batch_size, weights_shape[1] * weights_shape[2]]
            filters_shape = filters_shape + basis.shape[1:3].as_list()

            img_w = weights_shape[1]
            img_h = weights_shape[2]
        else:
            weights_shape = tf.shape(weights)
            filters_shape = tf.shape(basis)
            batch_flat_img_shape = [weights_shape[0], weights_shape[1] * weights_shape[2]]
            filters_shape = tf.concat(batch_flat_img_shape + filters_shape[1:3], axis=0)
            img_w = weights_shape[1]
            img_h = weights_shape[2]
            raise NotImplementedError("Only supported for images of known sizes")

        return filters_shape, img_h, img_w

    def _build_epsilon(self, epsilon_shape, seed=None):
        with tf.name_scope("Epsilon"):
            # Epsilon is [batch size, num samples, ...]
            return tf.random_normal(shape=epsilon_shape, dtype=self.dtype, seed=seed, name="epsilon")

    def _sample_with_net(self, epsilon, filters, weights, name="sample_with_net"):
        return conv2d_samples_linear_combination_filters(inputs=epsilon, filters=filters,
                                                         alpha=weights, name=name)

    def _sample_common(self, num_samples, epsilon, sample_method, return_epsilon, flatten_output, is_covariance):
        # Epsilon should be [batch dim, img width, img height, num channels] or
        # [batch dim, num samples, img width, img height, num channels]
        if is_covariance:
            name = "sample_covariance"
        else:
            name = "whiten_x"
        if sample_method == SampleMethod.NET:
            weights = self.weights_precision
            filters = self.filters_precision
            if not is_covariance:
                with tf.name_scope(name):
                    epsilon = self._get_epsilon_5_dim(num_samples, epsilon)
                    sample = self._sample_with_net(epsilon, filters, weights)
                    if flatten_output:
                        epsilon = self._flatten_keep_sample_dim(epsilon)
                        sample = self._flatten_keep_sample_dim(sample)
                    epsilon = self._squeeze_sample_dims(epsilon, name='epsilon')
                    sample = self._squeeze_sample_dims(sample, name='sample')
                    if return_epsilon:
                        return sample, epsilon
                    else:
                        return sample
            else:
                # Default to Cholesky sampling if NET is not defined
                sample_method = SampleMethod.CHOLESKY

        with tf.name_scope(name):
            epsilon = self._get_epsilon_5_dim(num_samples, epsilon)
            epsilon_5_dims_shape = tf.shape(epsilon)
            epsilon = self._flatten_keep_sample_dim(epsilon)

            if is_covariance:
                sample = super().sample_covariance(num_samples, epsilon, sample_method=sample_method)
            else:
                sample = super().whiten_x(num_samples, epsilon, sample_method=sample_method)

            if not flatten_output:
                epsilon = tf.reshape(epsilon, shape=epsilon_5_dims_shape)
                sample = tf.reshape(sample, shape=epsilon_5_dims_shape)
                sample = self._squeeze_sample_dims(sample, name='sample')
            epsilon = self._squeeze_sample_dims(epsilon, name='epsilon')
            if return_epsilon:
                return sample, epsilon
            else:
                return sample

    def sample_covariance(self, num_samples=1, epsilon=None, sample_method=None, return_epsilon=False,
                          flatten_output=False):
        if sample_method is None:
            sample_method = SampleMethod.NET
        return self._sample_common(num_samples, epsilon, sample_method, return_epsilon, flatten_output,
                                   is_covariance=True)

    def whiten_x(self, num_samples=1, x=None, sample_method=None, return_epsilon=False,
                 flatten_output=False):
        if sample_method is None:
            sample_method = SampleMethod.NET
        return self._sample_common(num_samples, x, sample_method, return_epsilon, flatten_output,
                                   is_covariance=False)


class PrecisionConvCholFilters(PrecisionConvFilters):
    def __init__(self, weights_precision, filters_precision, sample_shape, **kwargs):
        super().__init__(weights_precision=weights_precision, filters_precision=filters_precision,
                         sample_shape=sample_shape, **kwargs)
        self._log_diag_chol_covariance = None
        self._log_diag_chol_precision = None
        self._diag_covariance = None
        self._diag_precision = None
        self._recons_filters_precision_aligned = None

        self._build_with_covariance = False
        self.dtype = weights_precision.dtype
        self._t_indices = None
        self._l_indices = None

    def _build_chol_precision(self):
        with tf.name_scope("Precision_Chol"):
            return self._build_matrix_from_basis(weights=self.weights_precision, basis=self.filters_precision,
                                                 name="precision_chol")

    def _build_covariance(self):
        with tf.name_scope("Covariance"):
            return self._inverse_covariance_or_precision()

    def _build_precision(self):
        with tf.name_scope("Precision"):
            return tf.matmul(self.chol_precision, self.chol_precision, transpose_b=True, name="precision")

    def _build_log_det_covariance_with_chol(self):
        log_det = 2.0 * tf.reduce_sum(self.log_diag_chol_precision, axis=1)
        return tf.negative(log_det, name="log_det_covar")

    @property
    def log_diag_chol_covariance(self):
        if self._log_diag_chol_covariance is None:
            self._log_diag_chol_covariance = self._build_log_diag_chol_covariance()
        return self._log_diag_chol_covariance

    @log_diag_chol_covariance.setter
    def log_diag_chol_covariance(self, value):
        self._log_diag_chol_covariance = value

    def _build_log_diag_chol_covariance(self):
        with tf.name_scope("DiagCholCovariance"):
            diag_c = tf.matrix_diag_part(self.chol_covariance, name="diag_chol_covariance")
            return tf.log(diag_c, name="log_diag_chol_covariance")

    @property
    def log_diag_chol_precision(self):
        if self._log_diag_chol_precision is None:
            self._log_diag_chol_precision = self._build_log_diag_chol_precision()
        return self._log_diag_chol_precision

    @log_diag_chol_precision.setter
    def log_diag_chol_precision(self, value):
        self._log_diag_chol_precision = value

    def _build_log_diag_chol_precision(self):
        with tf.name_scope("DiagCholPrecision"):
            diag_p = self._build_diag_from_weights(name="diag_chol_precision")
            return tf.log(diag_p, name="log_diag_chol_precision")

    def _build_diag_sqrt_covariance(self):
        with tf.name_scope("DiagSqrtCovariance"):
            return tf.matrix_diag_part(self.sqrt_covariance, name="diag_chol_precision")

    def _build_diag_sqrt_precision(self):
        with tf.name_scope("DiagSqrtPrecision"):
            return tf.matrix_diag_part(self.sqrt_precision, name="diag_chol_precision")

    def _build_sqrt_covariance(self):
        # Do not use the parent method as it uses the weights and filters to build the sqrt covariance
        return super(PrecisionConvFilters, self)._build_sqrt_covariance()

    def _build_sqrt_precision(self):
        # Do not use the parent method as it uses the weights and filters to build the sqrt precision
        return super(PrecisionConvFilters, self)._build_sqrt_precision()

    def _conv_filter_for_diag(self, filters_shape):
        """
        Create a convolutional filter of shape (filter_width, filter_height, num_ch, 1) for computing the diagonal of
        the precision matrix. For example for a 3x3 kernel the output is

        num_ch     4         3         2         1         0

                |1 0 0|   |0 1 0|   |0 0 1|   |0 0 0|   |0 0 0|
                |0 0 0|   |0 0 0|   |0 0 0|   |1 0 0|   |0 1 0|
                |0 0 0|   |0 0 0|   |0 0 0|   |0 0 0|   |0 0 0|

        :param filters_shape:
        :return:
        """
        # TODO: These convolutional filters are just using for shifting the values in the image, this could done be
        # much more efficiently. A first optimization would be to use 1D filters, i.e. change 1 above for, for |1 0 0|
        # Another option would be to create a row/s (and/or column/s) of zeros. Concatenate on the appropriate side,
        # while discarding the row/s (and/or column/s) of the opposite side
        with tf.name_scope('Cov-Diag-Filter'):
            if isinstance(filters_shape, tf.Tensor):
                raise NotImplementedError("")

            half_size = (filters_shape[2] * filters_shape[3]) // 2 + 1
            cov_shape = [filters_shape[2], filters_shape[3], half_size]
            cov_filter = np.zeros(shape=cov_shape, dtype=np.float32)
            f_width = cov_filter.shape[1]
            i = 0
            j = 0
            for c in range(half_size):
                cov_filter[i, j, -(c + 1)] = 1
                j += 1
                if j == f_width:
                    j = 0
                    i += 1

            cov_filter = np.expand_dims(cov_filter, axis=3)  # (f_width, f_width, f_width // 2 + 1, 1)
            cov_filter = tf.constant(cov_filter)
            return cov_filter

    def _build_diag_part_with_conv(self, filters, basis, weights, name):
        """
        Optimized sparse implementation for getting the diagonal of the covariance or precision matrix from the
        weights and basis. The diagonal is the sum of the shifted squared values in the filters.
        This method is able to shift and add the values in the filters using a convolution with a kernel of
        [n_f, n_f, n_f // 2 + 1, 1]

        :param filters: tensor equal to matmul(weights, basis), [batch size, img width * img height, n_f * n_f]
        :param basis: tensor [num basis, n_f, n_f, 1, 1]
        :param weights: tensor [batch size, img width, img height, num basis]
        :param name: string
        :return:
        """
        filters_shape, img_h, img_w = self._compute_shapes_for_filter_matrix(basis, weights)

        with tf.name_scope(name + 'Diag-Part'):
            conv_diag_filter = self._conv_filter_for_diag(filters_shape)

            half_f_width = (filters_shape[2] * filters_shape[3]) // 2
            filters = filters[:, :, half_f_width:]  # Discard the zeros in the filters
            filters = tf.reshape(filters, (-1, img_w, img_h, half_f_width + 1))  # Reshape the filters to img_w, img_h

            filters_sq = tf.square(filters)

            diag_part = tf.nn.conv2d(filters_sq, conv_diag_filter, strides=(1, 1, 1, 1), padding='SAME')
            return tf.layers.flatten(diag_part)

    def _build_precision_diag_part(self):
        return self._build_diag_part_with_conv(filters=self.recons_filters_precision, basis=self.filters_precision,
                                               weights=self.weights_precision, name='Precision')

    def _align_filters_per_row(self, filters, basis, weights, name):
        """ This shifts the values in filters [b, n, nb] such that [b, i, :] now contains a row in the matrix rather
         than a column """
        filters_shape, img_h, img_w = self._compute_shapes_for_filter_matrix(basis, weights)

        with tf.name_scope(name + 'Row-Align'):
            conv_filter = self._conv_filter_for_diag(filters_shape)

            half_f_width = (filters_shape[2] * filters_shape[3]) // 2
            zeroes = filters[:, :, 0:half_f_width]  # Save the zeroes in the filters for later
            filters = filters[:, :, half_f_width:]  # Discard the zeros in the filters
            filters = tf.reshape(filters, (-1, img_w, img_h, half_f_width + 1))  # Reshape the filters to img_w, img_h

            # Shift each column, such that the same x,y position over channels represents a row in the matrix
            aligned_filters = tf.nn.depthwise_conv2d(filters, conv_filter, strides=(1, 1, 1, 1), padding='SAME')

            # Flatten back to [batch, num pixels, num_channels]
            aligned_filters = tf.reshape(aligned_filters, (-1, img_w * img_h, half_f_width + 1))

            aligned_filters = tf.concat([zeroes, aligned_filters], axis=-1)  # Add the zeros back
            return aligned_filters

    @property
    def recons_filters_precision_aligned(self):
        if self._recons_filters_precision_aligned is None:
            self._recons_filters_precision_aligned = self._align_filters_per_row(filters=self.recons_filters_precision,
                                                                                 basis=self.filters_precision,
                                                                                 weights=self.weights_precision,
                                                                                 name='Aligned-Filters')
        return self._recons_filters_precision_aligned

    def np_off_diag_mask(self):
        """
        Returns a ndarray of [n,n] that is 1 of the off-diagonal elements in L
        """
        assert self.recons_filters_precision.shape[1:3].is_fully_defined()
        n = self.recons_filters_precision.shape[1].value
        n_width = int(np.sqrt(n))
        nb = self.recons_filters_precision.shape[2].value
        nf = int(np.sqrt(nb))

        # Create a Cholesky kernel, without the element in the diagonal
        kernel = np.zeros((nf, nf), dtype=np.float32)
        kernel[nf // 2 + 1:, :] = 1
        kernel[nf // 2, nf // 2 + 1:] = 1

        kernel_list = np.tile(kernel[np.newaxis, :, :], (n, 1, 1))
        cholesky_matrix = np_ops.np_make_matrix_from_kernel_list(kernels=kernel_list, img_size=n_width)
        cholesky_matrix = cholesky_matrix.T  # Operation builds the transpose matrix
        return cholesky_matrix

    def np_off_diag_mask_compact(self):
        off_diag_mask = self.np_off_diag_mask()

        nb = self.recons_filters_precision.shape[2].value
        nf = int(np.sqrt(nb))
        n = off_diag_mask.shape[0]
        n_width = int(np.sqrt(n))

        nf2 = nf // 2

        off_diag_compact_mask = np.zeros(shape=(n, nb), dtype=np.float32)
        pix_i, pix_j = 0, 0
        for i in range(n):
            # Get a column in the matrix
            single_column = off_diag_mask[:, i]
            # Reshape to image size
            kernel = single_column.reshape((n_width, n_width))

            # Pad with zeros
            zero_row = np.zeros(shape=(n_width, nf2), dtype=off_diag_mask.dtype)
            zero_col = np.zeros(shape=(nf2, n_width + 2 * nf2), dtype=off_diag_mask.dtype)

            kernel = np.concatenate([zero_row, kernel, zero_row], axis=1)
            kernel = np.concatenate([zero_col, kernel, zero_col], axis=0)

            # Select the kernel location only
            pix_i_end = pix_i + 2 * nf2 + 1
            pix_j_end = pix_j + 2 * nf2 + 1
            kernel = kernel[pix_i:pix_i_end, pix_j:pix_j_end]

            pix_j += 1
            if pix_j == n_width:
                pix_j = 0
                pix_i += 1

            # Save the flatten kernel in the matrix
            off_diag_compact_mask[i] = kernel.reshape(-1)

        # Check that the matrix can be reconstructed from the kernels
        recons_off_diag_mask = np_ops.np_make_matrix_from_kernel_list(off_diag_compact_mask.reshape(n, nf, nf),
                                                                      img_size=n_width)
        assert np.all(off_diag_mask == recons_off_diag_mask.T), "Failed in creating off-diag mask"

        return off_diag_compact_mask

    def off_diag_mask_compact_aligned(self):
        """ Align the off_diag_mask [n, b] so that each [i, :] is a row in the matrix """
        filters_shape, img_h, img_w = self._compute_shapes_for_filter_matrix(self.filters_precision,
                                                                             self.weights_precision)

        off_diag_mask = self.np_off_diag_mask_compact()
        off_diag_mask = off_diag_mask[np.newaxis, ...]  # Add batch dimension

        conv_filter = self._conv_filter_for_diag(filters_shape)

        half_f_width = (filters_shape[2] * filters_shape[3]) // 2
        zeroes = off_diag_mask[:, :, 0:half_f_width]  # Save the zeroes in the mask for later
        off_diag_mask = off_diag_mask[:, :, half_f_width:]  # Discard the zeros in the mask
        off_diag_mask = off_diag_mask.reshape((1, img_w, img_h, half_f_width + 1))  # Reshape the img_w, img_h

        # Shift each column, such that the same x,y position over channels represents a row in the matrix
        aligned_off_diag_mask = tf.nn.depthwise_conv2d(off_diag_mask, conv_filter, strides=(1, 1, 1, 1), padding='SAME')

        # Flatten back to [batch, num pixels, num_channels]
        aligned_off_diag_mask = tf.reshape(aligned_off_diag_mask, (1, img_w * img_h, half_f_width + 1))

        aligned_off_diag_mask = tf.concat([zeroes, aligned_off_diag_mask], axis=-1)  # Add the zeros back

        return aligned_off_diag_mask[0]  # Remove batch dimension

    def sample_with_sparse_solver(self, num_samples=1, u=None, seed=None, sess=None, feed_dict=None):
        filters_shape, img_h, img_w = self._compute_shapes_for_filter_matrix(self.filters_precision,
                                                                             self.weights_precision)
        nc = (filters_shape[2] * filters_shape[3] - 1) // 2 + 1
        nf = filters_shape[2]
        n = img_h * img_w
        assert img_w == img_h, "Only supported for square images"

        # Random sample from unit Gaussian
        u = self._get_epsilon_5_dim(num_samples, epsilon=u, seed=seed)
        u = self._flatten_keep_sample_dim(u)

        t_matrix = self.recons_filters_precision
        t_matrix = t_matrix[..., :, nc - 1:]  # Discard the zeros at the start of the filter
        if not tf.executing_eagerly():
            assert isinstance(sess, tf.Session)
            u, t_matrix = sess.run([u, t_matrix], feed_dict=feed_dict)

        return self._solve_system_with_sparse_solver(t_matrix=t_matrix, u=u, img_w=img_w, n=n, nc=nc, nf=nf)

    def upper_chol_covariance_with_sparse_solver(self, sess=None, feed_dict=None, sparse_format=False):
        return self._invert_with_sparse_solver(sess=sess, feed_dict=feed_dict, use_iterative_solver=sparse_format,
                                               triang_sparse_format=sparse_format)

    def covariance_with_sparse_solver(self, only_x_rows=None, sess=None, feed_dict=None):
        if only_x_rows is None:
            chol_covariance = self._invert_with_sparse_solver(sess=sess, feed_dict=feed_dict)

            # Cov = M M^T, so do the matrix multiplication to get the covariance matrix
            covariance = chol_covariance
            for num_batch in range(covariance.shape[0]):
                covariance[num_batch] = np.matmul(covariance[num_batch], covariance[num_batch].T)
        else:
            # Use iterative solver to get the rows indicated by only_x_rows
            chol_covariance_x_rows = self._invert_with_sparse_solver(sess=sess, feed_dict=feed_dict,
                                                                     keep_idx_list=only_x_rows,
                                                                     use_iterative_solver=True,
                                                                     triang_sparse_format=True)

            batch_size = chol_covariance_x_rows.shape[0]
            num_features = chol_covariance_x_rows[0].shape[0]
            num_x_to_keep = len(only_x_rows)
            dtype = chol_covariance_x_rows[0][only_x_rows[0]].dtype

            chol_covariance_x_rows_dense = np.zeros(shape=(batch_size, num_x_to_keep, num_features), dtype=dtype)

            for b in range(batch_size):
                for i in range(num_x_to_keep):
                    j = num_features - only_x_rows[i]
                    chol_covariance_x_rows_dense[b, i, -j:] = chol_covariance_x_rows[b][only_x_rows[i]]

            chol_covariance_x_rows_dense = chol_covariance_x_rows_dense[:, np.newaxis]

            # Use iterative solver again to get the rows in the covariance for only_x_rows
            covariance = self._invert_with_sparse_solver(sess=sess, feed_dict=feed_dict,
                                                         use_iterative_solver=True,
                                                         x_for_dot=chol_covariance_x_rows_dense)

        return covariance

    def variance_with_sparse_solver(self, sess=None, feed_dict=None, use_iterative_solver=True):
        chol_covariance = self._invert_with_sparse_solver(sess=sess, feed_dict=feed_dict,
                                                          use_iterative_solver=use_iterative_solver)

        if use_iterative_solver:
            # The iterative solver does the dot products internally, so chol_covariance is diag_covariance
            return chol_covariance
        else:
            # Do dot products per row to get the diagonal of the covariance matrix
            diag_covariance = np.zeros(shape=chol_covariance.shape[:-1], dtype=chol_covariance.dtype)
            for num_batch in range(chol_covariance.shape[0]):
                diag_covariance[num_batch] = np.sum(chol_covariance[num_batch] * chol_covariance[num_batch], axis=1)

            return diag_covariance

    def _solve_system_with_sparse_solver(self, t_matrix, u, img_w, n, nc, nf, use_iterative_solver=False,
                                         triang_sparse_format=False, keep_idx_list=None, x_for_dot=None):
        """
        Solves the sparse system of equations t_matrix^T e = u for e.
        If use_iterative_solver, it solves the system t_matrix t_matrix^T E E^T = U for E, and
            it returns diag(E E^T).

        :param t_matrix: dense Cholesky precision matrix, ndarray [b, n, nc]
        :param u: dense matrix with the right hand side of the system, ndarray [b, s, n] or [b, s, n, n]
        :param img_w: the image width, sqrt(n), int
        :param n: the number of pixels, int
        :param nc: the maximum number of elements per pixel, nc = (nf**2 - 1) // 2 + 1 int
        :param nf: the neighbourhood size, int
        :param use_iterative_solver: bool, if True, it solves the system t_matrix^T E E^T = U, using an
            iterative solver that is more memory efficient (n * (nf // 2)), but slower.
        :return: e, the matrix with the solution to the system, [b, s, n] or [b, s, n, n]
        """
        # Get indices to go from dense matrix, [n,nc], to sparse matrix [n,n]
        # l_indices in tuple of destination locations in L matrix, and t_indices is array of
        # indies for flatten origin location in T matrix
        l_indices, t_indices = self._build_indices_for_sparse_matrix(img_size=img_w, n=n, nc=nc, nf=nf)
        if use_iterative_solver:
            # u is a sparse matrix, we need a dense one (epsilon) to save the dense results
            if triang_sparse_format:
                epsilon = np.empty(shape=[t_matrix.shape[0], 1], dtype=np.object)
            else:
                if x_for_dot is None:
                    epsilon = np.zeros(shape=[t_matrix.shape[0], 1, n], dtype=u.dtype)
                else:
                    epsilon = np.zeros(shape=x_for_dot.shape, dtype=u.dtype)

            num_samples = 1
            assert u.ndim == 2, "Expected a single matrix"

            # When solving the system, delete any row in E that is further than max_row_dist.
            # For example, a 3x3 kernel spans a row below the current pixel, so we need the next row,
            # and in a 5x5 kernel we need to keep two rows.
            # The indices start from the current pixel, so we need to also add the half of the values,
            # that fall to the right of the center pixel in the values that we are keeping (for a 3x3).
            max_row_dist = img_w * (nf // 2) + nf // 2
        else:
            assert keep_idx_list is None and x_for_dot is None
            num_samples = u.shape[1]
            max_row_dist = None
            epsilon = None

        for num_batch in range(t_matrix.shape[0]):
            # Indices for t_matrix are for flat matrix, so flatten it
            flat_t_matrix = t_matrix[num_batch].reshape(-1)

            # Discard neighbours outside of the image and maybe re-oder to match l_indices
            flat_t_matrix = flat_t_matrix[t_indices]

            # Build sparse matrix, csr_matrix format is good for the sparse solver,
            # but as we have to transpose, build as csc_matrix, such that when
            # transposed becomes a csr_matrix
            sparse_l_matrix = csc_matrix((flat_t_matrix, l_indices), shape=(n, n))
            sparse_l_transposed_matrix = csc_matrix.transpose(sparse_l_matrix)

            for num_sample in range(num_samples):
                # Solve the sparse system of equations: L^T epsilon = u, for epsilon
                if use_iterative_solver:
                    if x_for_dot is None:
                        x_for_dot_i = None
                    else:
                        x_for_dot_i = x_for_dot[num_batch, num_sample]

                    epsilon[num_batch, num_sample] = self._spsolve_triangular_iterative_dot(
                        A=sparse_l_transposed_matrix,
                        b=u,
                        max_row_dist=max_row_dist,
                        lower=False,
                        overwrite_A=True,
                        b_is_identity=True,
                        return_x_dot=not triang_sparse_format,
                        keep_idx_list=keep_idx_list,
                        x_for_dot=x_for_dot_i)
                else:
                    u[num_batch, num_sample] = spsolve_triangular(A=sparse_l_transposed_matrix,
                                                                  b=u[num_batch, num_sample],
                                                                  lower=False,
                                                                  overwrite_A=True,
                                                                  overwrite_b=True)

        if use_iterative_solver:
            # If using the iterative solvers, the result was saved in epsilon
            u = epsilon
        return u

    def _invert_with_sparse_solver(self, sess, feed_dict, use_iterative_solver=False, _invert_with_sparse_solver=False,
                                   triang_sparse_format=False, keep_idx_list=None, x_for_dot=None):
        filters_shape, img_h, img_w = self._compute_shapes_for_filter_matrix(self.filters_precision,
                                                                             self.weights_precision)

        nc = (filters_shape[2] * filters_shape[3] - 1) // 2 + 1
        nf = filters_shape[2]
        n = img_h * img_w
        assert img_w == img_h, "Only supported for square images"

        t_matrix = self.recons_filters_precision
        t_matrix = t_matrix[..., :, nc - 1:]  # Discard the zeros at the start of the filter
        if not tf.executing_eagerly():
            assert isinstance(sess, tf.Session)
            if use_iterative_solver:
                t_matrix = sess.run(t_matrix, feed_dict=feed_dict)
            else:
                t_matrix = sess.run(t_matrix, feed_dict=feed_dict)

        if use_iterative_solver:
            # Sparse I matrix of [b, n, n]
            identity_matrix = sparse_eye(m=n, dtype=t_matrix.dtype, format='lil')
        else:
            # Dense I matrix of [b, 1, n, n]
            # It seems really inefficient to build an identity matrix here, but the solver
            # will use this matrix to save the result
            batch_size = t_matrix.shape[0]
            identity_matrix = np.eye(N=n, dtype=t_matrix.dtype)
            identity_matrix = np.tile(identity_matrix[np.newaxis, :, :], (batch_size, 1, 1))
            identity_matrix = identity_matrix[:, np.newaxis]  # Add sample dim

        # Get the upper triangular Cholesky of the covariance matrix
        chol_covariance = self._solve_system_with_sparse_solver(t_matrix=t_matrix, u=identity_matrix, img_w=img_w, n=n,
                                                                nc=nc, nf=nf, use_iterative_solver=use_iterative_solver,
                                                                triang_sparse_format=triang_sparse_format,
                                                                keep_idx_list=keep_idx_list, x_for_dot=x_for_dot)

        # Remove number of samples dimension
        chol_covariance = chol_covariance[:, 0]

        return chol_covariance

    def _build_indices_for_sparse_matrix(self, img_size, n, nc, nf, cache_values=True):

        if self._t_indices is None or self._l_indices is None:
            indices_save_path = './L_sparse_indices_img_size_{}_nc_{}.npz'.format(img_size, nc)
            if cache_values and os.path.exists(indices_save_path):
                print('Loading indices for the sparse matrix in')
                print(indices_save_path)

                np_data = np.load(indices_save_path)
                self._l_indices = (np_data['l_indices_0'], np_data['l_indices_1'])
                self._t_indices = np_data['t_indices']

                return self._l_indices, self._t_indices

            # Array of flat indices
            t_indices = np.arange(n * nc)
            t_indices = t_indices.reshape((n, nc))

            # Add zero indices for the upper-left part of the filter that has zero values
            zero_ind = np.zeros(shape=(n, nc - 1), dtype=t_indices.dtype)
            t_indices = np.concatenate([zero_ind, t_indices], axis=1)
            t_indices = t_indices.reshape((n, nf, nf))

            # Build the dense L matrix with the indices, this could be a really big matrix
            print('Building indices for the sparse matrix')
            l_matrix = np_ops.np_make_matrix_from_kernel_list(t_indices, img_size=img_size, make_sparse=True,
                                                              verbose=True)
            l_matrix = l_matrix.T  # Make matrix makes the transpose matrix

            # Indices in L as row, col
            l_indices = l_matrix.nonzero()

            # Indices in dense matrix T, this step removes out of image indices, which
            # won't make into l_matrix, and also takes care of any reordering due to
            # the nonzero operation above. As L matrix is sparse, make the result
            # a dense matrix
            t_indices = l_matrix[l_indices].toarray()[0]

            # The first index in zero, so it wasn't counted in the nonzero step,
            # but it goes on the first position, so add it here. Numpy wants tuple indices
            self._l_indices = (np.concatenate([np.array([0]), l_indices[0]], axis=0),
                               np.concatenate([np.array([0]), l_indices[1]], axis=0))

            self._t_indices = np.concatenate([np.array([0]), t_indices], axis=0)

            if cache_values:
                print('Saving indices for the sparse matrix in')
                print(indices_save_path)
                try:
                    np.savez(indices_save_path, l_indices_0=self._l_indices[0], l_indices_1=self._l_indices[1],
                             t_indices=self._t_indices)
                except IOError as exc:
                    print('Could not save indices data file\n\t{}'.format(exc))

        return self._l_indices, self._t_indices

    @staticmethod
    def _spsolve_triangular_iterative_dot(A, b, max_row_dist, lower=True, overwrite_A=False, b_is_identity=False,
                                          return_x_dot=False, keep_idx_list=None, x_for_dot=None, verbose=True):
        """ This code is based on scipy.sparse.linalg.spsolve_triangular.
        It solves the system the following system for X
        A X = b
        It returns an ndarray with the non zero elements per row in X

        If return_x_dot is true, then it returns the diagonal of Y, where Y = X X^T

        If x_for_dot is given, then in returns Y = x_for_dot X^T

        Parameters
        ----------
        :param  A : (M, M) sparse matrix
            A sparse square triangular matrix. Should be in CSR format.
        :param  b : (M,) or (M, N) sparse matrix
            Right-hand side matrix in `A A^T X X^T = b`
        :param  max_row_dist : maximum number of rows in X to keep, while solving the system
        :param  lower : bool, optional
            Whether `A` is a lower or upper triangular matrix.
            Default is lower triangular matrix.
        :param  overwrite_A : bool, optional
            Allow changing `A`. The indices of `A` are going to be sorted and zero
            entries are going to be removed.
            Enabling gives a performance gain. Default is False.
        :param  b_is_identity : bool, optional
            Assumes `b` is an identity matrix,
            Enabling gives a performance gain. Default is False.
        :param return_x_dot, bool, if True, return the diagonal of Y.
            Enabling gives a performance gain. Default is False.
        :param keep_idx_list, list of indices with the rows to keep in X
        :param x_for_dot: (P, M) dense matrix, if provided the method returns Y = x_for_dot X^T
        :param verbose: bool, if True, it prints the progress in solving the system

        :return:
        -------
        If return_x_dot is False
        x : (M,) ndarray, where each row contains another ndarray with the non-zero
            elements in X, for the system A X = b

        If return_x_dot is True and x_for_dot is None
        x : (M,) ndarray
                Diagonal of the Y matrix for the system A A^T X X^T = b, and Y = X X^T.
                Shape of return matches shape of `b`.

        If return_x_dot is True and x_for_dot is not None
        x : (P, M) ndarray
                Returns Y = x_for_dot X, after solving the system A X = b.
                Shape of return matches shape of `x_for_dot`.
        """

        # Check the input for correct type and format.
        if not isspmatrix_csr(A):
            warn('CSR matrix format is required. Converting to CSR matrix.',
                 SparseEfficiencyWarning)
            A = csr_matrix(A)
        elif not overwrite_A:
            A = A.copy()

        if A.shape[0] != A.shape[1]:
            raise ValueError(
                'A must be a square matrix but its shape is {}.'.format(A.shape))

        # sum duplicates for non-canonical format
        A.sum_duplicates()

        if b.ndim not in [2]:
            raise ValueError(
                'b must have 2 dims but its shape is {}.'.format(b.shape))
        if A.shape[0] != b.shape[0]:
            raise ValueError(
                'The size of the dimensions of A must be equal to '
                'the size of the first dimension of b but the shape of A is '
                '{} and the shape of b is {}.'.format(A.shape, b.shape))

        x_dtype = np.result_type(A.data, b, np.float)
        common_dtype = np.common_type(A, b)

        if return_x_dot:
            assert keep_idx_list is None

            if x_for_dot is None:
                # The diagonal of Y
                ret_val = np.zeros(shape=A.shape[0], dtype=x_dtype)
            else:
                assert x_for_dot.ndim == 2
                assert x_for_dot.shape[1] == A.shape[0]

                # x_for_dot X
                ret_val = np.zeros(shape=x_for_dot.shape, dtype=x_dtype)
        else:
            assert x_for_dot is None

            # X in custom sparse triangular format
            ret_val = np.array([None] * b.shape[0])

        # Choose forward or backward order.
        min_max_idx_list = None
        if lower:
            row_indices = range(b.shape[0])
            del_row_index = -1
            if keep_idx_list is not None:
                min_max_idx_list = max(keep_idx_list)
        else:
            row_indices = range(b.shape[0] - 1, -1, -1)
            del_row_index = b.shape[0]
            if keep_idx_list is not None:
                min_max_idx_list = min(keep_idx_list)

        """ Sparse dot product, which does not create any numpy
                dense array internally. It should be faster than dense version, but the
                sorting operation makes it slower, left here for reference.
        """
        """
        def sparse_dot(x, i, A_column_indices_in_row_i, A_values_in_row_i):
            num_A_values_in_row_i = len(A_values_in_row_i)
            if num_A_values_in_row_i == 0:
                return

            # Copy the sparse data from each row, and multiply by A_values_in_row_i
            x_A_rows = [None] * num_A_values_in_row_i
            x_A_data = [None] * num_A_values_in_row_i
            for j in range(num_A_values_in_row_i):
                x_A_rows[j] = x.rows[A_column_indices_in_row_i[j]]
                x_A_data[j] = x.data[A_column_indices_in_row_i[j]].copy()
                for k in range(len(x_A_data[j])):
                    x_A_data[j][k] *= A_values_in_row_i[j]

            # Get the new sparsity pattern for x[i], it is the combination of all
            # the non-zero locations in x[A_column_indices_in_row_i]
            new_rows = set(x.rows[i])
            for j in range(num_A_values_in_row_i):
                new_rows.update(x_A_rows[j])
            # The new indices must be sorted and this is the slowest part
            new_rows = list(sorted(new_rows))

            # Substract over the number of elements in A_column_indices_in_row_i
            new_data = [0] * len(new_rows)
            for k in range(num_A_values_in_row_i):
                l = 0
                num_elem = len(x_A_rows[k])
                for j, new_row in enumerate(new_rows):
                    if new_row == x_A_rows[k][l]:
                        new_data[j] -= x_A_data[k][l]
                        l += 1
                        if l == num_elem:
                            break

            # Add the previous x[i] values
            l = 0
            num_elem = len(x.rows[i])
            for j, new_row in enumerate(new_rows):
                if new_row == x.rows[i][l]:
                    new_data[j] += x.data[i][l]
                    l += 1
                    if l == num_elem:
                        break

            # Update the sparse matrix
            x.data[i] = new_data
            x.rows[i] = new_rows
        """

        # For x we are going to use a representation that is like a simplified lil_matrix,
        # where the rows are dense so, x[i] will either be a dense row, or None
        x = np.array([None] * b.shape[0])

        # A tractable method for solving the system seems consists of creating
        # dense matrices for the few rows that we need in x.
        # The x matrix is at most [n,nf], so this is still tractable

        if verbose:
            row_indices = tqdm(row_indices, desc='spsolve_triangular_iterative_dot')

        # Fill x iteratively.
        for i in row_indices:
            # Get indices for i-th row.
            indptr_start = A.indptr[i]
            indptr_stop = A.indptr[i + 1]
            if lower:
                A_diagonal_index_row_i = indptr_stop - 1
                A_off_diagonal_indices_row_i = slice(indptr_start, indptr_stop - 1)
            else:
                A_diagonal_index_row_i = indptr_start
                A_off_diagonal_indices_row_i = slice(indptr_start + 1, indptr_stop)

            # Check regularity and triangularity of A.
            if indptr_stop <= indptr_start or A.indices[A_diagonal_index_row_i] < i:
                raise LinAlgError(
                    'A is singular: diagonal {} is zero.'.format(i))
            if A.indices[A_diagonal_index_row_i] > i:
                raise LinAlgError(
                    'A is not triangular: A[{}, {}] is nonzero.'
                    ''.format(i, A.indices[A_diagonal_index_row_i]))

            # Incorporate off-diagonal entries.
            A_column_indices_in_row_i = A.indices[A_off_diagonal_indices_row_i]
            A_values_in_row_i = A.data[A_off_diagonal_indices_row_i]

            # Make sure we didn't delete any element that we are about to use
            if len(A_values_in_row_i) > 0:
                if lower:
                    assert del_row_index < np.min(A_column_indices_in_row_i)
                else:
                    assert del_row_index > np.max(A_column_indices_in_row_i)

            # Incorporate off-diagonal entries.
            # This and the next if, computes in an efficient way this line
            # x[i] -= x[A_column_indices_in_row_i].T.dot(A_values_in_row_i)
            if len(A_values_in_row_i) > 0:
                # Do the dot product of previous rows in x, working with dense ndarrays
                x_A = x[A_column_indices_in_row_i]
                x_i = -np.dot(x_A.T, A_values_in_row_i)
            else:
                # First row when solving, there are no previous rows in x
                x_i = np.zeros(shape=b.shape[0], dtype=x_dtype)

            # Add the value for the ith row
            if b_is_identity:
                # If it's identity, current row is sparse, and only 1 at ith location
                x_i[i] += 1
            else:
                # if b is not identity, we need the dense representation of the sparse row
                x_i += b[i].toarray()[0]

            # Compute i-th entry of x.
            x_i /= A.data[A_diagonal_index_row_i]

            # Update the ith row in the x matrix
            x[i] = x_i

            if return_x_dot:
                if x_for_dot is None:
                    # The ith diagonal of Y is the dot product of the ith row in x
                    ret_val[i] = np.dot(x_i, x_i)
                else:
                    ret_val[:, i] = np.dot(x_for_dot, x_i)

            # Delete the rows in x that we do not need anymore, and optionally
            # save in ret_val the non-zero elements in the current row
            if lower:
                if not return_x_dot:
                    raise NotImplementedError()
                    # ret_val[i] = x_i[-i:]

                if i > max_row_dist:
                    del_row_index += 1
                    x[del_row_index] = None
            else:
                if not return_x_dot:
                    if keep_idx_list is None:
                        ret_val[i] = x_i[i:].astype(common_dtype)
                    elif i in keep_idx_list:
                        ret_val[i] = x_i[i:].astype(common_dtype)

                        # Do not compute for rows that we are not going to save
                        if i == min_max_idx_list:
                            if verbose:
                                # Set progress bar to the end
                                row_indices.update(b.shape[0])
                            return ret_val

                if i < b.shape[0] - max_row_dist:
                    del_row_index -= 1
                    x[del_row_index] = None

        return ret_val


class PrecisionDilatedConvCholFilters(PrecisionConvCholFilters):
    def __init__(self, weights_precision, filters_precision, sample_shape, dilation_rates, **kwargs):
        assert isinstance(weights_precision, (tuple, list))
        assert isinstance(dilation_rates, (tuple, list))

        self.dilation_rates = dilation_rates
        self.num_dilation_blocks = len(dilation_rates)

        assert self.num_dilation_blocks == len(weights_precision)
        assert self.num_dilation_blocks == len(filters_precision)

        self.weights_precision_list = weights_precision

        if filters_precision is None:
            self.filters_precision_list = self._id_filters(weights_precision)
        else:
            self.filters_precision_list = filters_precision

        assert isinstance(filters_precision, (tuple, list))

        self._dense_filters_precision = None

        weights_precision = tf.concat(self.weights_precision_list, axis=3, name='weights_precision_concat')
        filters_precision = tf.concat(self.filters_precision_list, axis=0, name='filters_precision_concat')

        for i in range(self.num_dilation_blocks):
            if np.alltrue(np.array(self.dilation_rates[i]) == 1):
                self.dilation_rates[i] = None

        super().__init__(weights_precision=weights_precision, filters_precision=filters_precision,
                         sample_shape=sample_shape, **kwargs)

    @staticmethod
    def _id_filters(weights_precision_list):
        id_filter_list = []
        for weights_precision in weights_precision_list:
            id_filter_list.append(super()._id_filters(weights_precision))
        return id_filter_list

    def _sample_with_net(self, epsilon, filters, weights, name="sample_with_net"):
        sample = list()
        for i in range(self.num_dilation_blocks):
            # Do the convolutions with the different dilation rates
            sample.append(conv2d_samples_linear_combination_filters(inputs=epsilon,
                                                                    filters=self.filters_precision_list[i],
                                                                    alpha=self.weights_precision_list[i],
                                                                    dilation_rate=self.dilation_rates[i],
                                                                    name=name + '_' + str(i)))
        # Sum all the samples together
        return tf.add_n(sample)

    @staticmethod
    def _sparse_dilated_filter_to_dense(filter, dilation_rate, name=None):
        """
        :param filter: a tensor filter of [<batch_size>, n_h, n_w, n_c_in, n_c_out]
        :param dilation_rate: a positive tuple of integers
        :return: a filter with dilated rate zeros in between each value

        Example: for a [3, 3, 1, 1] filter and dilation rate (2, 2)
        Input filter      Output filter
        | x, x, x |     | x, 0, x, 0, x |
        | x, x, x |     | 0, 0, 0, 0, 0 |
        | x, x, x |     | x, 0, x, 0, x |
                        | 0, 0, 0, 0, 0 |
                        | x, 0, x, 0, x |
        """
        with tf.name_scope(name=name, default_name="sparse_dilated_filter_to_dense"):
            filter = tf.convert_to_tensor(filter)
            assert filter.shape.ndims in [4, 5], "Kernel must have 4 or 5 dimensions, found {}".format(
                filter.shape.ndims)

            if not filter.shape.is_fully_defined():
                raise NotImplementedError("Operation only supported for kernels with known shape")

            remove_batch_dim = False
            if filter.shape.ndims == 4:
                remove_batch_dim = True
                filter = tf.expand_dims(filter, axis=0)

            if filter.shape[3].value is not 1:
                error_str = "Operation only supported for filters of [<b>, n_h, n_w, 1, n_c_out], found {}"
                raise NotImplementedError(error_str.format(filter.shape.as_list()))

            filter = tf.squeeze(filter, axis=3)  # Remove n_c_in dimension

            dilated_filter = unpooling2d_zero_filled(filter, stride=dilation_rate)

            n_h = dilated_filter.shape[1] - dilation_rate[0] + 1
            n_w = dilated_filter.shape[2] - dilation_rate[1] + 1

            dilated_filter = dilated_filter[:, 0:n_h, 0:n_w, :]

            dilated_filter = tf.expand_dims(dilated_filter, axis=3)  # Add n_c_in dimension

            if remove_batch_dim:
                dilated_filter = tf.squeeze(dilated_filter, axis=0)

            return dilated_filter

    def _build_dense_filter_matrix(self):
        with tf.name_scope('build_sum_dense_filter_list'):
            dilated_filter_list = list()
            max_filter_size = None

            # Make a list of dense filters from the (sparse) dilated filters
            for i in range(self.num_dilation_blocks):
                if self.dilation_rates[i] is None:
                    dilated_filter_list.append(self.filters_precision_list[i])
                else:
                    dilated_filter_list.append(self._sparse_dilated_filter_to_dense(self.filters_precision_list[i],
                                                                                    dilation_rate=self.dilation_rates[
                                                                                        i]))

                filter_size = dilated_filter_list[-1].shape[1:3].as_list()

                if max_filter_size is None:
                    max_filter_size = filter_size
                else:
                    if filter_size[0] > max_filter_size[0]:
                        max_filter_size[0] = filter_size[0]

                    if filter_size[1] > max_filter_size[1]:
                        max_filter_size[1] = filter_size[1]

            for i in range(self.num_dilation_blocks):
                filter_size = dilated_filter_list[i].shape[1:3].as_list()

                if filter_size[0] < max_filter_size[0] or filter_size[1] < max_filter_size[1]:
                    padding = (max_filter_size[0] - filter_size[0]) // 2, (max_filter_size[1] - filter_size[1]) // 2
                    padding = [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0], [0, 0]]

                    dilated_filter_list[i] = tf.pad(dilated_filter_list[i], paddings=padding)

            # Build a single matrix with all the dilated and padded filters together
            return tf.concat(dilated_filter_list, axis=0, name='dense_filter_matrix')

    @property
    def recons_filters_precision(self):
        if self._recons_filters_precision is None:
            self._recons_filters_precision = self._reconstruct_basis(weights=self.weights_precision,
                                                                     basis=self.dense_filters_precision)
        return self._recons_filters_precision

    @property
    def dense_filters_precision(self):
        if self._dense_filters_precision is None:
            self._dense_filters_precision = self._build_dense_filter_matrix()
        return self._dense_filters_precision

    def _build_chol_precision(self):
        with tf.name_scope("Precision_Chol"):
            return self._build_matrix_from_basis(weights=self.weights_precision, basis=self.dense_filters_precision,
                                                 name="precision_chol")

    def _build_precision_diag_part(self):
        return self._build_diag_part_with_conv(filters=self.recons_filters_precision,
                                               basis=self.dense_filters_precision,
                                               weights=self.weights_precision, name='Precision')

    def _build_indices_for_sparse_matrix(self, img_size, n, nc, nf, cache_values=True):
        raise NotImplementedError("")

    def covariance_with_sparse_solver(self, sess=None, feed_dict=None):
        raise NotImplementedError("")

    def variance_with_sparse_solver(self, sess=None, feed_dict=None, use_iterative_solver=False):
        raise NotImplementedError("")
