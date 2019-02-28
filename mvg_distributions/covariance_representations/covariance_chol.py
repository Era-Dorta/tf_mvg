import tensorflow as tf

from mvg_distributions.covariance_representations.covariance_matrix import Covariance, DecompMethod


class _CovarianceCholeskyCommon(Covariance):
    def __init__(self, inversion_method=None, **kwargs):
        if inversion_method is None:
            inversion_method = DecompMethod.CHOLESKY
        super(_CovarianceCholeskyCommon, self).__init__(inversion_method=inversion_method, **kwargs)


class CovarianceCholesky(_CovarianceCholeskyCommon):
    def __init__(self, chol_covariance, **kwargs):
        super(CovarianceCholesky, self).__init__(**kwargs)
        tf.assert_rank(chol_covariance, 3, message="Size must be [batch dim, feature dim, feature dim]")
        self._chol_covariance = chol_covariance

        self.dtype = self._chol_covariance.dtype
        if self._chol_covariance.shape.is_fully_defined():
            self._matrix_shape = self._chol_covariance.shape
        else:
            self._matrix_shape = tf.shape(self._chol_covariance)

    def _build_covariance(self):
        with tf.name_scope("Covariance"):
            return tf.matmul(self.chol_covariance, self.chol_covariance, transpose_b=True, name="covariance")

    def _build_chol_covariance(self):
        return self._chol_covariance

    def _build_covariance_diag_part(self):
        with tf.name_scope("covariance_diag_part"):
            return tf.einsum('bij,bij->bi', self.chol_covariance, self.chol_covariance)

    def x_precision_x(self, x, mean_batch=False, no_gradients=False):
        """
        :param x: input, should be [batch dim, num_samples, num features], or [batch dim, num features]
        :param mean_batch: if True do the mean over the batch
        :param no_gradients: if True, do not back-propagate gradients on the Cholesky
        :return:
        """
        # , M = cholesky(covariance)
        # x (M M^T)^-1 x^T = x (M^T)^-1 * M^-1 x^T -> M^-1 x^T -> M y^T = x^T
        # Solve the M system for y^T and multiply by the solution by itself
        if x.shape.ndims == 2:
            x = tf.expand_dims(x, 2)
        else:
            x = tf.transpose(x, perm=[0, 2, 1])
        # x should be [batch dim, num features, num_samples]
        x.shape[0:2].assert_is_compatible_with(self.chol_covariance.shape[0:2])

        if no_gradients:
            chol_covariance = tf.stop_gradient(self.chol_covariance)
        else:
            chol_covariance = self.chol_covariance

        # Compute x * Cholesky
        x_chol_precision = tf.matrix_triangular_solve(chol_covariance, x)

        # Compute matmul((x * Cholesky),(x * Cholesky)) and sum over samples
        squared_error = tf.multiply(x_chol_precision, x_chol_precision)
        squared_error = tf.reduce_sum(squared_error, axis=1)  # Error per sample
        if squared_error.shape[1].value == 1:
            squared_error = tf.squeeze(squared_error, axis=1, name="x_precision_x")  # Remove sample dim

        if mean_batch:
            squared_error = tf.reduce_mean(squared_error, name="mean_x_precision_x")
        return squared_error


class PrecisionCholesky(_CovarianceCholeskyCommon):
    def __init__(self, chol_precision, **kwargs):
        super(PrecisionCholesky, self).__init__(**kwargs)
        tf.assert_rank(chol_precision, 3, message="Size must be [batch dim, feature dim, feature dim]")
        self._chol_precision = chol_precision
        self._build_with_covariance = False

        self.dtype = self._chol_precision.dtype
        if self._chol_precision.shape.is_fully_defined():
            self._matrix_shape = self._chol_precision.shape
        else:
            self._matrix_shape = tf.shape(self._chol_precision)

    def _build_covariance(self):
        return self._inverse_covariance_or_precision()

    def _build_precision(self):
        with tf.name_scope("Precision"):
            return tf.matmul(self.chol_precision, self.chol_precision, transpose_b=True, name="precision")

    def _build_chol_precision(self):
        return self._chol_precision

    def _build_precision_diag_part(self):
        with tf.name_scope("precision_diag_part"):
            return tf.einsum('bij,bij->bi', self._chol_precision, self._chol_precision)

    def x_precision_x(self, x, mean_batch=False, no_gradients=False):
        """
        :param x: input, should be [batch dim, num_samples, num features], or [batch dim, num features]
        :param mean_batch: if True do the mean over the batch
        :param no_gradients: if True, do not back-propagate gradients on the Cholesky
        :return:
        """
        # M = cholesky(covariance)
        # x M M^T x^T = (x M) (M x)^T = y y^T
        if x.shape.ndims == 2:
            x = tf.expand_dims(x, 1)
        # x should be [batch dim, num_samples, num features]
        x.shape[0:3:2].assert_is_compatible_with(self.chol_covariance.shape[0:3:2])

        if no_gradients:
            chol_precision = tf.stop_gradient(self.chol_precision)
        else:
            chol_precision = self.chol_precision

        # Compute x * Cholesky
        x_chol_precision = tf.matmul(x, chol_precision)

        # Compute matmul((x * Cholesky),(x * Cholesky)) and sum over samples
        squared_error = tf.multiply(x_chol_precision, x_chol_precision)
        squared_error = tf.reduce_sum(squared_error, axis=2)  # Error per sample
        if squared_error.shape[1].value == 1:
            squared_error = tf.squeeze(squared_error, axis=1, name="x_precision_x")  # Remove sample dim

        if mean_batch:
            squared_error = tf.reduce_mean(squared_error, name="mean_x_precision_x")
        return squared_error
