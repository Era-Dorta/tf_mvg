from enum import Enum

import tensorflow as tf

import mvg_distributions.log_likelihoods
import mvg_distributions.utils.utils as utils


class DecompMethod(Enum):
    # The method to use for computing inv(covariance) or inv(precision) and log(det(covariance))
    CUSTOM = 0  # Specific methods, used by Diag and EigDiag classes
    LU = 1
    CHOLESKY = 2
    EIGEN = 3


class SampleMethod(Enum):
    # The method to use for sampling the covariance or the precision matrices
    CHOLESKY = 0
    SQRT = 1
    NET = 2


class Covariance(object):
    # Abstract Covariance class, provides naive implementations for the precision matrix, sampling,
    # log(det(covariance)) and x*Precision*x^T computation, where x is a vector

    def __init__(self, inversion_method=None, **kwargs):
        with tf.name_scope("Covariance-Class"):
            self._build_with_covariance = True
            if inversion_method is None:
                self.inversion_method = DecompMethod.CHOLESKY
            else:
                self.inversion_method = inversion_method
            self._covariance = None
            self._eig_val_covar = None
            self._eig_val_precision = None
            self._eig_vec = None
            self._precision = None
            self._chol_covariance = None
            self._chol_precision = None
            self._chol_covariance = None
            self._chol_precision = None
            self._sqrt_covariance = None
            self._sqrt_precision = None
            self._log_det_covariance = None
            self.dtype = tf.float32
            self._matrix_shape = None
            self._diag_covariance = None
            self._diag_precision = None

    @property
    def covariance(self):
        if self._covariance is None:
            self._covariance = self._build_covariance()
        return self._covariance

    def _build_covariance(self):
        raise NotImplementedError("")

    @property
    def eig_val_covar(self):
        if self._eig_val_covar is None:
            if self._eig_val_precision is None:
                if self._eig_vec is not None:
                    raise ValueError("Eig values is unknown but eig vec is set")
                self._eig_val_covar, self._eig_val_precision, self._eig_vec = self._build_eig_decomp()
            else:
                self._eig_val_covar = tf.divide(1.0, self._eig_val_precision, name="eig_val_covar")
        return self._eig_val_covar

    @property
    def eig_val_precision(self):
        if self._eig_val_precision is None:
            if self._eig_val_covar is None:
                if self._eig_vec is not None:
                    raise ValueError("Eig values is unknown but eig vec is set")
                self._eig_val_covar, self._eig_val_precision, self._eig_vec = self._build_eig_decomp()
            else:
                self._eig_val_precision = tf.divide(1.0, self._eig_val_covar, name="eig_val_covar")
        return self._eig_val_precision

    @property
    def eig_vec(self):
        if self._eig_vec is None:
            if self._eig_val_covar is not None or self._eig_val_precision is not None:
                raise ValueError("Eig vector is unknown but eig value is set")
            self._eig_val_covar, self._eig_val_precision, self._eig_vec = self._build_eig_decomp()
        return self._eig_vec

    def _build_eig_decomp(self):
        with tf.name_scope("Eig_Decomp"):
            if self._build_with_covariance:
                eig_val_covar, eig_vec = tf.self_adjoint_eig(self.covariance)
                eig_val_precision = tf.divide(1.0, eig_val_covar, name="eig_val_precision")
            else:
                eig_val_precision, eig_vec = tf.self_adjoint_eig(self.precision)
                eig_val_covar = tf.divide(1.0, eig_val_precision, name="eig_val_covariance")
            return eig_val_covar, eig_val_precision, eig_vec

    @property
    def precision(self):
        if self._precision is None:
            self._precision = self._build_precision()
        return self._precision

    def _build_precision(self):
        return self._inverse_covariance_or_precision()

    @property
    def covariance_diag_part(self):
        if self._diag_covariance is None:
            self._diag_covariance = self._build_covariance_diag_part()
        return self._diag_covariance

    def _build_covariance_diag_part(self):
        return tf.matrix_diag_part(self.covariance, name="covariance_diag")

    @property
    def precision_diag_part(self):
        if self._diag_precision is None:
            self._diag_precision = self._build_precision_diag_part()
        return self._diag_precision

    def _build_precision_diag_part(self):
        return tf.matrix_diag_part(self.precision, name="precision_diag")

    @property
    def chol_covariance(self):
        if self._chol_covariance is None:
            self._chol_covariance = self._build_chol_covariance()
        return self._chol_covariance

    def _build_chol_covariance(self):
        with tf.name_scope("Covariance_Chol"):
            return tf.cholesky(self.covariance, name="covariance_chol")

    @property
    def chol_precision(self):
        if self._chol_precision is None:
            self._chol_precision = self._build_chol_precision()
        return self._chol_precision

    def _build_chol_precision(self):
        with tf.name_scope("Precision_Chol"):
            return tf.cholesky(self.precision, name="precision_chol")

    @property
    def sqrt_covariance(self):
        if self._sqrt_covariance is None:
            self._sqrt_covariance = self._build_sqrt_covariance()
        return self._sqrt_covariance

    def _build_sqrt_covariance(self):
        with tf.name_scope("Covariance_Sqrt"):
            return utils.sqrtm_eig(eig_vals=self.eig_val_covar, eig_vec=self.eig_vec)

    @property
    def sqrt_precision(self):
        if self._sqrt_precision is None:
            self._sqrt_precision = self._build_sqrt_precision()
        return self._sqrt_precision

    def _build_sqrt_precision(self):
        with tf.name_scope("Precision_Sqrt"):
            return utils.sqrtm_eig(eig_vals=self.eig_val_precision, eig_vec=self.eig_vec)

    def log_det_covariance(self, decomp_method=None):
        if self._log_det_covariance is None:
            if decomp_method is None:
                decomp_method = DecompMethod.CHOLESKY
            self._log_det_covariance = self._build_log_det_covariance(decomp_method)
        return self._log_det_covariance

    def _build_log_det_covariance_with_eig(self):
        with tf.name_scope("Log_Det_Covariance"):
            if self._build_with_covariance:
                return tf.reduce_sum(tf.log(self.eig_val_covar), axis=1, name="log_det_covar")
            else:
                return tf.negative(tf.reduce_sum(tf.log(self.eig_val_precision), axis=1), name="log_det_covar")

    def _build_log_det_covariance_with_chol(self):
        if self._build_with_covariance:
            return mvg_distributions.log_likelihoods._log_det_with_cholesky(cholesky=self.chol_covariance,
                                                                            dtype=self.dtype,
                                                                            out_name="log_det_covar")
        else:
            log_det = mvg_distributions.log_likelihoods._log_det_with_cholesky(cholesky=self.chol_precision,
                                                                               dtype=self.dtype)
            return tf.negative(log_det, name="log_det_covar")

    def _build_log_det_covariance_with_lu(self):
        if self._build_with_covariance:
            return tf.log(tf.matrix_determinant(self.covariance, name="log_det_covar"))
        else:
            return tf.negative(tf.log(tf.matrix_determinant(self.precision)), name="log_det_covar")

    def _build_log_det_covariance(self, decomp_method):
        if decomp_method == DecompMethod.CHOLESKY:
            return self._build_log_det_covariance_with_chol()
        elif decomp_method == DecompMethod.LU:
            return self._build_log_det_covariance_with_lu()
        elif decomp_method == DecompMethod.EIGEN:
            return self._build_log_det_covariance_with_eig()
        else:
            raise ValueError("Decomp method is not supported")

    def _inverse_covariance_or_precision(self):
        # Create the precision(covariance) matrix by inverting the covariance(precision)
        if self._build_with_covariance:
            name, out_name = "Precision", "precision"
        else:
            name, out_name = "Covariance", "covariance"
        if self.inversion_method == DecompMethod.CHOLESKY:
            if self._build_with_covariance:
                matrix = self.chol_covariance
            else:
                matrix = self.chol_precision
            with tf.name_scope(name):
                return self._matrix_inverse_with_cholesky(matrix, name=out_name)
        elif self.inversion_method == DecompMethod.LU:
            if self._build_with_covariance:
                matrix = self.covariance
            else:
                matrix = self.precision
            with tf.name_scope(name):
                return tf.matrix_inverse(matrix, name=out_name)
        elif self.inversion_method == DecompMethod.EIGEN:
            if self._build_with_covariance:
                eig_val = self.eig_val_covar
            else:
                eig_val = self.eig_val_precision
            return utils.symmetric_matrix_from_eig_decomp(eig_val, self.eig_vec, name=name, do_inv=True,
                                                          out_name=out_name)
        else:
            raise ValueError("Invalid inversion method")

    @staticmethod
    def _matrix_inverse_with_cholesky(cholesky_input, name="matrix_inverse"):
        # Invert with cholesky, C = Cholesky(A), find inverse(A) by solving C X = I
        if cholesky_input.shape.is_fully_defined():
            matrix_shape = cholesky_input.shape.as_list()
        else:
            matrix_shape = tf.shape(cholesky_input)
        eye = tf.eye(batch_shape=[matrix_shape[0]], num_rows=matrix_shape[1], num_columns=matrix_shape[2],
                     dtype=cholesky_input.dtype)
        return tf.cholesky_solve(cholesky_input, eye, name=name)

    def _build_epsilon(self, num_samples, seed=None):
        with tf.name_scope("Epsilon"):
            # Epsilon is [batch size, num_samples, num features]
            if isinstance(num_samples, tf.Tensor) or isinstance(self._matrix_shape, tf.Tensor):
                if isinstance(self._matrix_shape, tf.TensorShape):
                    matrix_shape = tf.convert_to_tensor(self._matrix_shape)
                else:
                    matrix_shape = self._matrix_shape
                epsilon_shape = tf.stack([matrix_shape[0], num_samples, matrix_shape[1]], axis=0)
            else:
                assert num_samples >= 1, "Number of samples must be positive"
                epsilon_shape = tf.TensorShape([self._matrix_shape[0], num_samples, self._matrix_shape[1]])

            return tf.random_normal(shape=epsilon_shape, dtype=self.dtype, seed=seed, name="epsilon")

    def x_precision_x(self, x, mean_batch=False, no_gradients=False):
        # x shape should be [batch dim, num features]
        with tf.name_scope("x_precision_x"):
            if x.shape.ndims == 2:
                x = tf.expand_dims(x, axis=1)
            x.shape[0].assert_is_compatible_with(self.precision.shape[0])
            x.shape[2].assert_is_compatible_with(self.precision.shape[1])
            if no_gradients:
                precision = tf.stop_gradient(self.precision)
            else:
                precision = self.precision
            squared_error = mvg_distributions.log_likelihoods._batch_squared_error_with_covariance(predictions=x,
                                                                                                   labels=None,
                                                                                                   inv_covariance=precision,
                                                                                                   name="impl",
                                                                                                   out_name="x_precision_x")
            if mean_batch:
                squared_error = tf.reduce_mean(squared_error, name="mean_x_precision_x")
            return squared_error

    def _get_epsilon(self, num_samples, epsilon, seed=None):
        if epsilon is None:
            epsilon = self._build_epsilon(num_samples, seed=seed)
        if epsilon.shape.ndims == 2:
            epsilon = tf.expand_dims(epsilon, 1)  # Epsilon should be [batch dim, 1, num features]
        epsilon.shape[0:3:2].assert_is_compatible_with(self.covariance.shape[0:3:2])
        return epsilon

    def _squeeze_sample_dims(self, sample, name):
        if sample.shape[1].value is not None and sample.shape[1].value == 1:
            return tf.squeeze(sample, axis=1, name=name)  # [batch dim, num features]
        else:
            return sample  # [batch dim, num samples, num features]

    def _sample_or_whiten(self, chol_matrix, num_samples, epsilon, sample=True, name="sample"):
        epsilon = self._get_epsilon(num_samples, epsilon)
        if sample:
            epsilon = tf.matrix_transpose(epsilon)
            sample = tf.matmul(chol_matrix, epsilon)
            sample = tf.matrix_transpose(sample)
        else:
            sample = tf.matmul(epsilon, chol_matrix)
        return self._squeeze_sample_dims(sample, name)

    def _sample_or_whiten_with_inv_chol(self, chol_matrix, num_samples, epsilon, sample=True,
                                        name="sample_with_inv_chol"):
        # (cholesky(covariance) * cholesky(covariance)^T)^-1 = (cholesky(covariance)^T)^-1 * cholesky(covariance)^-1
        # (cholesky(covariance)^-1)^T * epsilon = sample -> (cholesky(covariance)^-1)^T * sample = epsilon
        # Implicit inverse using a system of equations as we know that precision_cholesky is lower triangular
        # Tensorflow solves the system Ax=b, so the formulation above fits in by making epsilon a column vector
        epsilon = self._get_epsilon(num_samples, epsilon)
        epsilon = tf.matrix_transpose(epsilon)
        sample = tf.matrix_triangular_solve(chol_matrix, epsilon, adjoint=sample)
        sample = tf.matrix_transpose(sample)
        return self._squeeze_sample_dims(sample, name=name)

    def sample_covariance(self, num_samples=1, epsilon=None, sample_method=None, **kwargs):
        # Epsilon should be [batch dim, num features]
        if sample_method is None:
            sample_method = SampleMethod.CHOLESKY
        with tf.name_scope("sample_covariance"):
            if sample_method == SampleMethod.SQRT:
                sample_matrix = self.sqrt_covariance
            elif sample_method == SampleMethod.CHOLESKY:
                if self._build_with_covariance:
                    sample_matrix = self.chol_covariance
                else:
                    return self._sample_or_whiten_with_inv_chol(chol_matrix=self.chol_precision,
                                                                num_samples=num_samples,
                                                                epsilon=epsilon)
            else:
                raise ValueError("Sampling can only be done with Sqrt or Cholesky")
            return self._sample_or_whiten(chol_matrix=sample_matrix, num_samples=num_samples, epsilon=epsilon,
                                          name="sample_covariance")

    def whiten_x(self, num_samples=1, x=None, sample_method=None, **kwargs):
        # Epsilon should be [batch dim, num features]
        if sample_method is None:
            sample_method = SampleMethod.CHOLESKY
        with tf.name_scope("whiten_x"):
            if sample_method == SampleMethod.SQRT:
                sample_matrix = self.sqrt_precision
            elif sample_method == SampleMethod.CHOLESKY:
                if self._build_with_covariance:
                    return self._sample_or_whiten_with_inv_chol(chol_matrix=self.chol_covariance,
                                                                num_samples=num_samples,
                                                                epsilon=x, sample=False)
                else:
                    sample_matrix = self.chol_precision
            else:
                raise ValueError("Sampling can only be done with Sqrt or Cholesky")
            return self._sample_or_whiten(chol_matrix=sample_matrix, num_samples=num_samples, epsilon=x, sample=False,
                                          name="whiten_x")


class CovarianceFull(Covariance):
    def __init__(self, covariance, **kwargs):
        super(CovarianceFull, self).__init__(**kwargs)
        tf.assert_rank(covariance, 3, message="Data must be [batch dim, num features, num features]")
        self._covariance = covariance
        self.dtype = self.covariance.dtype
        if self.covariance.shape.is_fully_defined():
            self._matrix_shape = self.covariance.shape
        else:
            self._matrix_shape = tf.shape(self.covariance)

    def _build_covariance(self):
        return self._covariance


class PrecisionFull(Covariance):
    def __init__(self, precision, **kwargs):
        super(PrecisionFull, self).__init__(**kwargs)
        tf.assert_rank(precision, 3, message="Data must be [batch dim, num features, num features]")
        self._precision = precision
        self.dtype = self._precision.dtype
        if self.precision.shape.is_fully_defined():
            self._matrix_shape = self.precision.shape
        else:
            self._matrix_shape = tf.shape(self.precision)
        self._build_with_covariance = False

    def _build_covariance(self):
        return self._inverse_covariance_or_precision()

    def _build_precision(self):
        return self._precision
