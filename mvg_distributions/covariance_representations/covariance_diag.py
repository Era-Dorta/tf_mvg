import tensorflow as tf

from mvg_distributions.covariance_representations.covariance_matrix import Covariance, DecompMethod


class _CovarianceDiagCommon(Covariance):
    def __init__(self, log_diag_covariance, log_diag_precision, inversion_method=None, **kwargs):
        if inversion_method is None:
            inversion_method = DecompMethod.CUSTOM
        assert inversion_method is DecompMethod.CUSTOM, "CovarianceDiag only supports CUSTOM inversion"
        super(_CovarianceDiagCommon, self).__init__(inversion_method=inversion_method, **kwargs)
        self._log_diag_covariance = log_diag_covariance
        self._log_diag_precision = log_diag_precision

        self._diag_shape = None
        self._chol_diag_covariance = None
        self._chol_diag_precision = None

        self.dtype = self._log_diag_covariance.dtype

    @property
    def log_diag_covariance(self):
        return self._log_diag_covariance

    @property
    def log_diag_precision(self):
        return self._log_diag_precision

    def _build_covariance_diag_part(self):
        return tf.exp(self.log_diag_covariance, name="covariance_diag")

    def _build_precision_diag_part(self):
        return tf.exp(self.log_diag_precision, name="precision_diag")

    def _build_shapes(self, input_tensor):
        with tf.name_scope("Matrix_Shape"):
            shape = tf.shape(input_tensor)
            if input_tensor.shape[1].value is not None:
                return shape, tf.concat([shape, [input_tensor.shape[1].value]], axis=0)
            else:
                return shape, tf.concat([shape, [shape[1]]], axis=0)

    def _build_covariance(self):
        with tf.name_scope("Covariance"):
            matrix = tf.zeros(self._matrix_shape, dtype=self.dtype)
            return tf.matrix_set_diag(matrix, self._diag_covariance, name="covariance")

    def _build_precision(self):
        with tf.name_scope("Precision"):
            matrix = tf.zeros(self._matrix_shape, dtype=self.dtype)
            return tf.matrix_set_diag(matrix, self._diag_precision, name="precision")

    def _build_chol_covariance(self):
        with tf.name_scope("Covariance_Chol"):
            matrix = tf.zeros(self._matrix_shape, dtype=self.dtype)
            return tf.matrix_set_diag(matrix, self._chol_diag_covariance, name="chol_covariance")

    def _build_chol_precision(self):
        with tf.name_scope("Precision_Chol"):
            matrix = tf.zeros(self._matrix_shape, dtype=self.dtype)
            return tf.matrix_set_diag(matrix, self._chol_diag_precision, name="chol_precision")

    def _build_sqrt_covariance(self):
        # For a diagonal matrix the square root matrix and the cholesky matrix are equivalent
        return self._build_chol_covariance()

    def _build_sqrt_precision(self):
        # For a diagonal matrix the square root matrix and the cholesky matrix are equivalent
        return self._build_chol_precision()

    def log_det_covariance(self, decomp_method=None):
        if decomp_method is None:
            decomp_method = DecompMethod.CUSTOM
        return super(_CovarianceDiagCommon, self).log_det_covariance(decomp_method)

    def _build_log_det_covariance(self, decomp_method):
        assert decomp_method == DecompMethod.CUSTOM, "CovarianceDiag only supports CUSTOM log det"
        with tf.name_scope("Log_Det_Covariance"):
            if self._build_with_covariance:
                return tf.reduce_sum(self._log_diag_covariance, axis=1, name="log_det_covar")
            else:
                return tf.negative(tf.reduce_sum(self._log_diag_precision, axis=1), name="log_det_covar")

    def x_precision_x(self, x, mean_batch=False, no_gradients=False):
        # x shape should be [batch dim, num features]
        with tf.name_scope("x_precision_x"):
            if no_gradients:
                diag_precision = tf.stop_gradient(self._diag_precision)
            else:
                diag_precision = self._diag_precision

            if x.shape.ndims == 2:
                x = tf.expand_dims(x, axis=1)

            diag_precision = tf.expand_dims(diag_precision, axis=1)

            squared_error = tf.square(x)
            squared_error *= diag_precision
            squared_error = tf.reduce_sum(squared_error, axis=2, name="x_precision_x")  # Error per sample
            if squared_error.shape[1].value == 1:
                squared_error = tf.squeeze(squared_error, axis=1, name="x_precision_x")

            if mean_batch:
                squared_error = tf.reduce_mean(squared_error, name="mean_x_precision_x")
            return squared_error

    def _sample_or_whiten(self, diag_sqrt, num_samples, epsilon, name="sample", **kwargs):
        # Sample with the sqrt of the diagonal
        # For diagonal matrices, the sqrt and the Cholesky decomposition are equivalent
        epsilon = self._get_epsilon(num_samples, epsilon)

        diag_sqrt = tf.expand_dims(diag_sqrt, axis=1)  # [batch dim, 1, num features]
        sample = tf.multiply(epsilon, diag_sqrt)

        return self._squeeze_sample_dims(sample, name)

    def sample_covariance(self, num_samples=1, epsilon=None, **kwargs):
        with tf.name_scope("sample_covariance"):
            return self._sample_or_whiten(diag_sqrt=self._chol_diag_covariance, num_samples=num_samples,
                                          epsilon=epsilon,
                                          name="sample_covariance")

    def whiten_x(self, num_samples=1, x=None, **kwargs):
        with tf.name_scope("whiten_x"):
            return self._sample_or_whiten(diag_sqrt=self._chol_diag_precision, num_samples=num_samples, epsilon=x,
                                          name="whiten_x")


class CovarianceDiag(_CovarianceDiagCommon):
    def __init__(self, log_diag_covariance, **kwargs):
        tf.assert_rank(log_diag_covariance, 2, message="Data must be [batch dim, num features]")
        with tf.name_scope("CovarianceDiag"):
            log_diag_precision = tf.negative(log_diag_covariance, name="log_diag_precision")

        super(CovarianceDiag, self).__init__(log_diag_covariance=log_diag_covariance,
                                             log_diag_precision=log_diag_precision, **kwargs)

        with tf.name_scope("CovarianceDiag"):
            self._diag_shape, self._matrix_shape = self._build_shapes(self._log_diag_covariance)
            self._diag_covariance = tf.exp(self._log_diag_covariance, name="diag_covariance")
            self._diag_precision = tf.exp(-self._log_diag_covariance, name="diag_precision")
            self._chol_diag_covariance = tf.exp(self._log_diag_covariance * 0.5, name="chol_diag_covariance")
            self._chol_diag_precision = tf.exp(self._log_diag_covariance * -0.5, name="chol_diag_precision")


class PrecisionDiag(_CovarianceDiagCommon):
    def __init__(self, log_diag_precision, **kwargs):
        tf.assert_rank(log_diag_precision, 2, message="Data must be [batch dim, num features]")
        with tf.name_scope("PrecisionDiag"):
            log_diag_covariance = tf.negative(log_diag_precision, name="log_diag_covariance")

        super(PrecisionDiag, self).__init__(log_diag_precision=log_diag_precision,
                                            log_diag_covariance=log_diag_covariance, **kwargs)
        self._build_with_covariance = False

        with tf.name_scope("PrecisionDiag"):
            self._diag_shape, self._matrix_shape = self._build_shapes(self._log_diag_precision)
            self._diag_precision = tf.exp(self._log_diag_precision, name="diag_precision")
            self._diag_covariance = tf.exp(-self._log_diag_precision, name="diag_covariance")
            self._chol_diag_covariance = tf.exp(self._log_diag_precision * -0.5, name="chol_diag_covariance")
            self._chol_diag_precision = tf.exp(self._log_diag_precision * 0.5, name="chol_diag_precision")
