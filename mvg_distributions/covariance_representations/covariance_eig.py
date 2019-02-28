import tensorflow as tf

import mvg_distributions.utils.utils as utils
from mvg_distributions.covariance_representations.covariance_matrix import Covariance, DecompMethod, \
    SampleMethod


class _CovarianceEigCommon(Covariance):
    def __init__(self, log_eig_val_covar, log_eig_val_precision, eig_vec, inversion_method=None, **kwargs):
        if inversion_method is None:
            inversion_method = DecompMethod.EIGEN
        super(_CovarianceEigCommon, self).__init__(inversion_method=inversion_method, **kwargs)
        tf.assert_rank(eig_vec, 3, message="Data must be [batch dim, num features, num features]")
        self._eig_vec = eig_vec
        self._log_eig_val_covar = log_eig_val_covar
        self._log_eig_val_precision = log_eig_val_precision

        self.dtype = self._eig_vec.dtype
        if self._eig_vec.shape.is_fully_defined():
            # Eig vec shape is [batch size, num features, num eig vectors]
            eig_vec_shape = self._eig_vec.shape
            self._matrix_shape = tf.TensorShape([eig_vec_shape[0], eig_vec_shape[1], eig_vec_shape[1]])
        else:
            eig_vec_shape = tf.shape(self._eig_vec)
            self._matrix_shape = tf.stack([eig_vec_shape[0], eig_vec_shape[1], eig_vec_shape[1]], axis=0)

    @property
    def log_eig_val_covar(self):
        return self._log_eig_val_covar

    @property
    def log_eig_val_precision(self):
        return self._log_eig_val_precision

    def log_det_covariance(self, decomp_method=None):
        if decomp_method is None:
            decomp_method = DecompMethod.EIGEN
        return super(_CovarianceEigCommon, self).log_det_covariance(decomp_method)

    def sample_covariance(self, num_samples=1, epsilon=None, sample_method=None, **kwargs):
        if sample_method is None:
            sample_method = SampleMethod.SQRT
        return super(_CovarianceEigCommon, self).sample_covariance(num_samples=num_samples, epsilon=epsilon,
                                                                   sample_method=sample_method)

    def whiten_x(self, num_samples=1, x=None, sample_method=None, **kwargs):
        if sample_method is None:
            sample_method = SampleMethod.SQRT
        return super(_CovarianceEigCommon, self).whiten_x(num_samples=num_samples, x=x,
                                                          sample_method=sample_method)

    def x_precision_x(self, x, mean_batch=False, no_gradients=False):
        # x shape should be [batch dim, num features]
        with tf.name_scope("x_precision_x"):
            if x.shape.ndims == 2:
                x = tf.expand_dims(x, axis=1)
            x.shape[0].assert_is_compatible_with(self.eig_vec.shape[0])
            x.shape[2].assert_is_compatible_with(self.eig_vec.shape[1])
            if no_gradients:
                eig_vec = tf.stop_gradient(self.eig_vec)
                eig_val_precision = tf.stop_gradient(self.eig_val_precision)
            else:
                eig_vec = self.eig_vec
                eig_val_precision = self.eig_val_precision

            eig_val_precision = tf.expand_dims(eig_val_precision, axis=1)

            # Compute ((x * eig_vec)**2) * eig_val
            squared_error = tf.matmul(x, eig_vec)
            squared_error = tf.square(squared_error)
            squared_error = tf.multiply(squared_error, eig_val_precision)

            squared_error = tf.reduce_sum(squared_error, axis=2)  # Error per sample
            if squared_error.shape[1].value == 1:
                squared_error = tf.squeeze(squared_error, axis=1, name="x_precision_x") # Remove sample dim

            if mean_batch:
                squared_error = tf.reduce_mean(squared_error, name="mean_x_precision_x")
            return squared_error


class CovarianceEig(_CovarianceEigCommon):
    def __init__(self, log_eig_val_covar, eig_vec, **kwargs):
        tf.assert_rank(log_eig_val_covar, 2, message="Data must be [batch dim, num features]")
        with tf.name_scope("CovarianceEig"):
            log_eig_val_precision = tf.negative(log_eig_val_covar, name="log_eig_val_precision")

        super(CovarianceEig, self).__init__(log_eig_val_covar=log_eig_val_covar,
                                            log_eig_val_precision=log_eig_val_precision, eig_vec=eig_vec, **kwargs)

        with tf.name_scope("EigDecomp"):
            self._eig_val_covar = tf.exp(self.log_eig_val_covar, name='eig_vals')
            self._eig_val_precision = tf.exp(-self.log_eig_val_covar, name='inv_eig_val')

    def _build_covariance(self):
        with tf.name_scope("Covariance"):
            return utils.symmetric_matrix_from_eig_decomp(self.eig_val_covar, self.eig_vec, name="Covariance",
                                                          out_name="covariance")


class PrecisionEig(_CovarianceEigCommon):
    def __init__(self, log_eig_val_precision, eig_vec, **kwargs):
        tf.assert_rank(log_eig_val_precision, 2, message="Data must be [batch dim, num features]")
        with tf.name_scope("PrecisionEig"):
            log_eig_val_covar = tf.negative(log_eig_val_precision, name="log_eig_val_covar")

        super(PrecisionEig, self).__init__(log_eig_val_covar=log_eig_val_covar,
                                           log_eig_val_precision=log_eig_val_precision, eig_vec=eig_vec, **kwargs)

        self._build_with_covariance = False
        with tf.name_scope("EigDecomp"):
            self._eig_val_covar = tf.exp(-self.log_eig_val_precision, name='eig_vals')
            self._eig_val_precision = tf.exp(self.log_eig_val_precision, name='inv_eig_val')

    def _build_covariance(self):
        return self._inverse_covariance_or_precision()

    def _build_precision(self):
        return utils.symmetric_matrix_from_eig_decomp(self.eig_val_precision, self.eig_vec, name="Precision",
                                                      out_name="precision")


class _CovarianceEigDiagCommon(_CovarianceEigCommon):
    def __init__(self, diag_a, inversion_method=None, **kwargs):
        if inversion_method is None:
            inversion_method = DecompMethod.CUSTOM
        super(_CovarianceEigDiagCommon, self).__init__(inversion_method=inversion_method, **kwargs)
        tf.assert_rank(diag_a, 2, message="Data must be [batch dim, num features]")

        self._diag_a = diag_a
        self._covariance_no_diag = None
        self._precision_no_diag = None

        self.__diag_eig_val_covar = None
        self.__diag_eig_val_precision = None
        self.__diag_eig_vec_covar = None
        self.__diag_eig_vec_precision = None

    @property
    def diag_a(self):
        return self._diag_a

    @property
    def covariance_no_diag(self):
        if self._covariance_no_diag is None:
            self._covariance_no_diag = self._build_covariance_no_diag()
        return self._covariance_no_diag

    def _build_covariance_no_diag(self):
        with tf.name_scope("CovarianceNoDiag"):
            return utils.symmetric_matrix_from_eig_decomp(self.eig_val_covar, self.eig_vec,
                                                          name="Covariance_No_Diag",
                                                          out_name="covariance_no_diag")

    @property
    def precision_no_diag(self):
        if self._precision_no_diag is None:
            self._precision_no_diag = self._build_precision_no_diag()
        return self._precision_no_diag

    def _build_precision_no_diag(self):
        with tf.name_scope("PrecisionNoDiag"):
            return utils.symmetric_matrix_from_eig_decomp(self.eig_val_precision, self.eig_vec,
                                                          name="Precision_No_Diag",
                                                          out_name="precision_no_diag")

    @property
    def _diag_eig_val_covar(self):
        if self.__diag_eig_val_covar is None:
            if self.__diag_eig_vec_covar is not None:
                raise ValueError("Eig values is unknown but eig vec is set")
            self.__diag_eig_val_covar, self.__diag_eig_vec_covar = self._build___eig_decomp_covar()
        return self.__diag_eig_val_covar

    @property
    def _diag_eig_vec_covar(self):
        if self.__diag_eig_vec_covar is None:
            if self.__diag_eig_val_covar is not None:
                raise ValueError("Eig vec is unknown but eig value is set")
            self.__diag_eig_val_covar, self.__diag_eig_vec_covar = self._build___eig_decomp_covar()
        return self.__diag_eig_vec_covar

    def _build___eig_decomp_covar(self):
        with tf.name_scope("EigDecompCovariance"):
            return tf.self_adjoint_eig(self.covariance)

    @property
    def _diag_eig_val_precision(self):
        if self.__diag_eig_val_precision is None:
            if self.__diag_eig_vec_precision is not None:
                raise ValueError("Eig values is unknown but eig vec is set")
            self.__diag_eig_val_precision, self.__diag_eig_vec_precision = self._build___eig_decomp_precision()
        return self.__diag_eig_val_precision

    @property
    def _diag_eig_vec_precision(self):
        if self.__diag_eig_vec_precision is None:
            if self.__diag_eig_val_precision is not None:
                raise ValueError("Eig vec is unknown but eig value is set")
            self.__diag_eig_val_precision, self.__diag_eig_vec_precision = self._build___eig_decomp_precision()
        return self.__diag_eig_vec_precision

    def _build___eig_decomp_precision(self):
        with tf.name_scope("EigDecompPrecision"):
            return tf.self_adjoint_eig(self.precision)

    def _build_sqrt_covariance(self):
        with tf.name_scope("Covariance_Sqrt"):
            return utils.sqrtm_eig(eig_vals=self._diag_eig_val_covar, eig_vec=self._diag_eig_vec_covar)

    def _build_sqrt_precision(self):
        with tf.name_scope("Precision_Chol"):
            return utils.sqrtm_eig(eig_vals=self._diag_eig_val_precision, eig_vec=self._diag_eig_vec_precision)

    def _build_log_det_covariance_with_eig(self):
        with tf.name_scope("Log_Det_Covariance"):
            if self._build_with_covariance:
                return tf.reduce_sum(tf.log(self._diag_eig_val_covar), axis=1, name="log_det_covar")
            else:
                return tf.negative(tf.reduce_sum(tf.log(self._diag_eig_val_precision), axis=1), name="log_det_covar")

    def log_det_covariance(self, decomp_method=None):
        if decomp_method is None:
            decomp_method = DecompMethod.CHOLESKY
        return super(_CovarianceEigDiagCommon, self).log_det_covariance(decomp_method)

    def _inverse_covariance_or_precision(self):
        # Create the precision or the covariance matrix by inverting the covariance or the precision
        if self.inversion_method == DecompMethod.EIGEN:
            if self._build_with_covariance:
                name, out_name = "Precision", "precision"
                eig_val, eig_vec = self._diag_eig_val_covar, self._diag_eig_vec_covar
            else:
                name, out_name = "Covariance", "covariance"
                eig_val, eig_vec = self._diag_eig_val_precision, self._diag_eig_vec_precision
            return utils.symmetric_matrix_from_eig_decomp(eig_val, eig_vec, name=name, do_inv=True, out_name=out_name)
        else:
            return super(_CovarianceEigDiagCommon, self)._inverse_covariance_or_precision()

    def sample_covariance(self, num_samples=1, epsilon=None, sample_method=None, **kwargs):
        if sample_method is None:
            sample_method = SampleMethod.CHOLESKY
        return super(_CovarianceEigDiagCommon, self).sample_covariance(num_samples=num_samples, epsilon=epsilon,
                                                                       sample_method=sample_method)

    def whiten_x(self, num_samples=1, x=None, sample_method=None, **kwargs):
        if sample_method is None:
            sample_method = SampleMethod.CHOLESKY
        return super(_CovarianceEigDiagCommon, self).whiten_x(num_samples=num_samples, x=x,
                                                              sample_method=sample_method)

    def x_precision_x(self, x, mean_batch=False, no_gradients=False):
        # Do not use the _CovarianceEigCommon x_precision_x, as it doesn't work with the diagonal term
        return Covariance.x_precision_x(self, x, mean_batch, no_gradients)


class CovarianceEigDiag(_CovarianceEigDiagCommon):
    def __init__(self, diag_a, log_eig_val_covar, eig_vec, **kwargs):
        tf.assert_rank(log_eig_val_covar, 2, message="Data must be [batch dim, num features]")
        with tf.name_scope("CovarianceEig"):
            log_eig_val_precision = tf.negative(log_eig_val_covar, name="log_eig_val_precision")

        super(CovarianceEigDiag, self).__init__(diag_a=diag_a, log_eig_val_covar=log_eig_val_covar,
                                                log_eig_val_precision=log_eig_val_precision,
                                                eig_vec=eig_vec, **kwargs)

        with tf.name_scope("EigDecomp"):
            self._eig_val_covar = tf.exp(self.log_eig_val_covar, name='eig_vals')
            self._eig_val_precision = tf.exp(-self.log_eig_val_covar, name='inv_eig_val')

    def _build_covariance(self):
        return utils.symmetric_matrix_from_eig_decomp_with_diag(self.eig_val_covar, self.eig_vec, self.diag_a,
                                                                name="Covariance", out_name="covariance")

    def _build_precision(self):
        if self.inversion_method == DecompMethod.CUSTOM:
            return utils.symmetric_matrix_from_eig_decomp_with_diag(self.eig_val_covar, self.eig_vec, self.diag_a,
                                                                    do_inv=True, name="Precision", out_name="precision")
        else:
            return super(CovarianceEigDiag, self)._build_precision()


class PrecisionEigDiag(_CovarianceEigDiagCommon):
    def __init__(self, diag_a, log_eig_val_precision, eig_vec, **kwargs):
        tf.assert_rank(log_eig_val_precision, 2, message="Data must be [batch dim, num features]")
        with tf.name_scope("PrecisionEig"):
            log_eig_val_covar = tf.negative(log_eig_val_precision, name="log_eig_val_covar")

        super(PrecisionEigDiag, self).__init__(diag_a=diag_a, log_eig_val_covar=log_eig_val_covar,
                                               log_eig_val_precision=log_eig_val_precision,
                                               eig_vec=eig_vec, **kwargs)
        self._build_with_covariance = False
        with tf.name_scope("EigDecomp"):
            self._eig_val_covar = tf.exp(-self.log_eig_val_precision, name='eig_vals')
            self._eig_val_precision = tf.exp(self.log_eig_val_precision, name='inv_eig_val')

    def _build_covariance(self):
        if self.inversion_method == DecompMethod.CUSTOM:
            return utils.symmetric_matrix_from_eig_decomp_with_diag(self.eig_val_precision, self.eig_vec, self.diag_a,
                                                                    do_inv=True, name="Covariance",
                                                                    out_name="covariance")
        else:
            return self._inverse_covariance_or_precision()

    def _build_precision(self):
        return utils.symmetric_matrix_from_eig_decomp_with_diag(self.eig_val_precision, self.eig_vec, self.diag_a,
                                                                name="Precision", out_name="precision")

    def x_precision_x(self, x, mean_batch=False, no_gradients=False):
        # x shape should be [batch dim, num features]
        with tf.name_scope("x_precision_x"):
            if x.shape.ndims == 2:
                x = tf.expand_dims(x, axis=1)
            x.shape[0].assert_is_compatible_with(self.eig_vec.shape[0])
            x.shape[2].assert_is_compatible_with(self.eig_vec.shape[1])
            if no_gradients:
                eig_vec = tf.stop_gradient(self.eig_vec)
                eig_val_precision = tf.stop_gradient(self.eig_val_precision)
                diag_a = tf.stop_gradient(self.diag_a)
            else:
                eig_vec = self.eig_vec
                eig_val_precision = self.eig_val_precision
                diag_a = self.diag_a

            eig_val_precision = tf.expand_dims(eig_val_precision, axis=1)
            diag_a = tf.expand_dims(diag_a, axis=1)

            # Compute ((x * eig_vec)**2) * eig_val
            squared_error = tf.matmul(x, eig_vec)
            squared_error = tf.square(squared_error)
            squared_error = tf.multiply(squared_error, eig_val_precision)

            # Add the diagonal term
            squared_error += tf.square(x) * diag_a

            squared_error = tf.reduce_sum(squared_error, axis=2)  # Error per sample
            if squared_error.shape[1].value == 1:
                squared_error = tf.squeeze(squared_error, axis=1, name="x_precision_x") # Remove sample dim

            if mean_batch:
                squared_error = tf.reduce_mean(squared_error, name="mean_x_precision_x")
            return squared_error
