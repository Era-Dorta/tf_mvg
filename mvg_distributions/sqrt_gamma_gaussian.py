import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import seed_stream
import tensorflow_probability as tfp
import mvg_distributions.covariance_representations as cov_rep
from  mvg_distributions.gamma import SqrtGamma

tfd = tfp.distributions
tfb = tfp.bijectors


class SqrtGammaGaussian(tfd.Distribution):
    def __init__(self, df, log_diag_scale, add_mode_correction=False, validate_args=False, allow_nan_stats=True,
                 name="SqrtGammaGaussian"):
        """
        Square root Gamma-Gaussian distribution, this is equivalent to a Cholesky-Wishart distribution with a
        diagonal scale matrix. Thus it has the same hyper-parameters, as a the Cholesky-Wishart distribution.

        This distribution expects as input Cholesky Precision matrices. Moreover, it assumes that the diagonal elements
        in the matrix are log(values).
        Args:
            The distribution is defined for batch (b) of M (pxp) matrices, forming a tensor of [b, p, p]

            df: degrees of freedom, a tensor of [b], the values in it must be df > p - 1
            log_diag_scale: a tensor of [b, p] with the log diagonal values of the matrix S
            add_mode_correction: bool, if using the distribution as a prior, setting this to True will add
                a correction factor to log_diag_scale, such that the log_prob will have the maximum in S
            validate_args:
            allow_nan_stats:
            name:
        """
        parameters = locals()

        with tf.name_scope(name=name):
            df = tf.convert_to_tensor(df)
            log_diag_scale = tf.convert_to_tensor(log_diag_scale)

            assert df.shape.ndims == 1
            assert log_diag_scale.shape.ndims == 2

        self._df = df
        self._log_diag_scale = log_diag_scale
        graph_parents = [df, log_diag_scale]

        self.p = self.log_diag_scale.shape[1].value
        if self.p is None:
            self.p = tf.shape(self.log_diag_scale)[1]

        self._mode_correction_factor(add_mode_correction)

        self._sqrt_gamma_dist = None
        self._normal_dist = None

        super().__init__(dtype=self.df.dtype, reparameterization_type=tf.distributions.FULLY_REPARAMETERIZED,
                         validate_args=validate_args, allow_nan_stats=allow_nan_stats, parameters=parameters,
                         graph_parents=graph_parents, name=name)

    @property
    def sqrt_gamma_dist(self):
        if self._sqrt_gamma_dist is None:
            half_df = 0.5 * self.df  # [b]
            # 0.0 to 0.5 - 0.5 p, then add 0.5 * df to all
            a = np.linspace(0.0, 0.5 - 0.5 * self.p, self.p, dtype=np.float32)  # [n]
            a = a[np.newaxis, :] + half_df[:, tf.newaxis]  # [b, n]

            b = 0.5 / tf.exp(self.log_diag_scale)  # [b, n]
            self._sqrt_gamma_dist = SqrtGamma(concentration=a, rate=b)

        return self._sqrt_gamma_dist

    @property
    def normal_dist(self):
        if self._normal_dist is None:
            sqrt_diag_scale = tf.exp(0.5 * self.log_diag_scale)
            sqrt_diag_scale = tf.tile(sqrt_diag_scale[:, :, tf.newaxis], (1, 1, self.p))  # [b, n, n]
            self._normal_dist = tfd.Normal(loc=0, scale=sqrt_diag_scale)  # [b, n, n]

        return self._normal_dist

    @property
    def log_diag_scale(self):
        return self._log_diag_scale

    @property
    def df(self):
        return self._df

    def _mode_correction_factor(self, add_mode_correction):
        if add_mode_correction:
            # corrected_diag_scale =  diag_scale * ((p - 1)/(tf.range(p) * (1 - p) + (p - 1) * (df - 1)))
            correction_factor = tf.log(self.p - 1.)
            p_range = tf.range(self.p, dtype=self._log_diag_scale.dtype)[tf.newaxis, :]
            correction_factor -= tf.log(p_range * (1. - self.p) + (self.p - 1.) * (self.df[:, tf.newaxis] - 1.))
            self._log_diag_scale += correction_factor

    def _log_prob_sqrt_gamma(self, x):
        log_diag_prob = self.sqrt_gamma_dist.log_prob(tf.matrix_diag_part(x))
        return tf.reduce_sum(log_diag_prob, axis=1)

    def _log_prob_normal(self, x):
        log_off_diag_prob = self.normal_dist.log_prob(x)

        off_diag_mask = tf.ones(shape=tf.shape(x))
        off_diag_mask = tf.matrix_band_part(off_diag_mask, -1, 0)
        off_diag_mask = tf.matrix_set_diag(off_diag_mask, tf.zeros(shape=tf.shape(x)[:-1]))

        log_off_diag_prob *= off_diag_mask
        return tf.reduce_sum(log_off_diag_prob, axis=[1, 2])

    def _log_prob(self, x):
        log_diag_prob = self._log_prob_sqrt_gamma(x)

        log_off_diag_prob = self._log_prob_normal(x)

        return log_diag_prob + log_off_diag_prob

    def _batch_shape_tensor(self):
        return tf.shape(self.log_diag_scale)[0]

    def _batch_shape(self):
        return self.log_diag_scale.shape[0:1]

    def _event_shape_tensor(self):
        event_dim = tf.shape(self.log_diag_scale)[1]
        return tf.stack([event_dim, event_dim])

    def _event_shape(self):
        event_dim = self.log_diag_scale.shape[1]
        return tf.TensorShape([event_dim, event_dim])

    def _sample_n(self, n, seed=None):
        stream = seed_stream.SeedStream(seed=seed, salt="Wishart")

        # Sample a normal full matrix
        x = self.normal_dist.sample(sample_shape=n, seed=stream())

        # Sample the log diagonal
        log_g = self.sqrt_gamma_dist.sample(sample_shape=n, seed=stream())

        # Discard the upper triangular part
        x = tf.matrix_band_part(x, -1, 0)

        # Set the diagonal
        x = tf.matrix_set_diag(x, log_g)

        return x


class SparseSqrtGammaGaussian(SqrtGammaGaussian):
    def __init__(self, df, log_diag_scale, add_mode_correction=False, validate_args=False, allow_nan_stats=True,
                 name="SparseSqrtGammaGaussian"):
        """
        Sparse square root Gamma-Gaussian distribution, this is equivalent to a Cholesky-Wishart distribution with a
        diagonal scale matrix and with a sparsity correction factor. Thus it has the same hyper-parameters, as a the
        Cholesky-Wishart distribution.
        Args:
            The distribution is defined for batch (b) of M (pxp) matrices, forming a tensor of [b, p, p]

            df: degrees of freedom, a tensor of [b], the values in it must be df > p - 1
            log_diag_scale: a tensor of [b, p] with the log diagonal values of the matrix S
            add_mode_correction: bool, if using the distribution as a prior, setting this to True will add
                a correction factor to log_diag_scale, such that the log_prob will have the maximum in S
            validate_args:
            allow_nan_stats:
            name:
        """
        super().__init__(df, log_diag_scale, add_mode_correction=add_mode_correction, validate_args=validate_args,
                         allow_nan_stats=allow_nan_stats, name=name)

    @staticmethod
    def _convert_to_cov_obj(value):
        if not isinstance(value, cov_rep.PrecisionConvCholFilters):
            value = tf.convert_to_tensor(value, name="value")
            log_prob_shape = ()
            if value.shape.ndims == 2:
                # Add batch dimension
                value = tf.expand_dims(value, axis=0)
            if value.shape.ndims == 3:
                log_prob_shape = tf.shape(value)[0:1]
            if value.shape.ndims == 4:
                # Collapse batch and sample dimension
                shape = tf.shape(value)
                log_prob_shape = shape[0:2]
                new_shape = [log_prob_shape[0] * log_prob_shape[1]]
                new_shape = tf.concat((new_shape, shape[2:]), axis=0)
                value = tf.reshape(value, new_shape)
            value = cov_rep.PrecisionCholesky(chol_precision=value)
        else:
            log_prob_shape = value.sample_shape[0:1]

        return value, log_prob_shape

    @property
    def normal_dist(self):
        if self._normal_dist is None:
            sqrt_diag_scale = tf.exp(0.5 * self.log_diag_scale)
            sqrt_diag_scale = sqrt_diag_scale[:, :, tf.newaxis]  # [b, n, 1]
            self._normal_dist = tfd.Normal(loc=0, scale=sqrt_diag_scale)  # [b, n, 1]

        return self._normal_dist

    def _call_log_prob(self, value, name, **kwargs):
        with self._name_scope(name):
            value, log_prob_shape = self._convert_to_cov_obj(value)
            try:
                log_prob = self._log_prob(value)
                return tf.reshape(log_prob, log_prob_shape)
            except NotImplementedError as original_exception:
                try:
                    log_prob = tf.log(self._prob(value))
                    return tf.reshape(log_prob, log_prob_shape)
                except NotImplementedError:
                    raise original_exception

    def _log_prob_sqrt_gamma(self, x):
        log_diag_prob = self.sqrt_gamma_dist.log_prob(x.log_diag_chol_precision)
        return tf.reduce_sum(log_diag_prob, axis=1)

    def _log_prob_normal(self, x):
        if isinstance(x, cov_rep.PrecisionConvCholFilters):
            nb = x.recons_filters_precision.shape[2].value

            # Get the elements in matrix [b, n, n] after they've been aligned per row, this is a [b, n, nb] tensor
            # that if it were reshaped to [b, n_w, n_h, n_b], the vector [b, i, j, :] contain the values of
            # the kth row in the matrix, where k corresponds to the i,j pixel.
            # For each row, we discard the leading zeros and the diagonal element
            off_diag_elements_aligned = x.recons_filters_precision_aligned[:, :, nb // 2 + 1:]

            log_off_diag_prob = self.normal_dist.log_prob(off_diag_elements_aligned)

            # Some elements in recons_filters_precision get zeroed out due to the zero padding for elements out of the
            # image in the convolution operator, thus they are not part of the Cholesky matrix.
            # Do not take into account those elements for the log probability computation
            off_diag_mask_aligned = x.off_diag_mask_compact_aligned()

            # log_off_diag_prob is [b, n, nb // 2 + 1], off_diag_mask is [n, nb]
            log_off_diag_prob *= off_diag_mask_aligned[tf.newaxis, :, nb // 2 + 1:]

            log_off_diag_prob = tf.reduce_sum(log_off_diag_prob, axis=[1, 2])
        else:
            log_off_diag_prob = super()._log_prob_normal(x.chol_precision)
        return log_off_diag_prob
