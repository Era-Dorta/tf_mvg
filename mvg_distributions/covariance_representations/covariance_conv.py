import numpy as np
import tensorflow as tf

from mvg_distributions.covariance_representations.covariance_matrix import Covariance, SampleMethod
from mvg_distributions.utils.variable_filter_functions import conv2d_samples_linear_combination_filters
from mvg_distributions.utils.unpooling import unpooling2d_zero_filled


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

    def _get_epsilon(self, num_samples, epsilon):
        epsilon_shape = self._get_epsilon_flat_shape(num_samples)
        if epsilon is None:
            epsilon = self._build_epsilon(epsilon_shape)
        if epsilon.shape.ndims + 1 == epsilon_shape.shape[0].value:
            epsilon = tf.expand_dims(epsilon, 1)  # Epsilon should be [batch dim, 1, ...]
        return epsilon

    def _get_epsilon_img_shape(self, num_samples):
        return tf.concat([self.sample_shape[0:1], [num_samples], self.sample_shape[1:]], axis=0)

    def _get_epsilon_5_dim(self, num_samples, epsilon):
        epsilon_shape = self._get_epsilon_img_shape(num_samples)
        if epsilon is None:
            epsilon = self._build_epsilon(epsilon_shape)
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

    def _build_epsilon(self, epsilon_shape):
        with tf.name_scope("Epsilon"):
            # Epsilon is [batch size, num samples, ...]
            return tf.random_normal(shape=epsilon_shape, dtype=self.dtype, name="epsilon")

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

        self._build_with_covariance = False
        self.dtype = weights_precision.dtype

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
