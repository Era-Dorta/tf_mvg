import tensorflow as tf
import tensorflow_probability
import numpy as np

import mvg_distributions.covariance_representations as cov_rep
import mvg_distributions.log_likelihoods as ll
from mvg_distributions.kl_divergence import kl_divergence_unit_gaussian, kl_divergence_mv_gaussian_v2
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow_probability.python.distributions import MultivariateNormalLinearOperator

import math
import abc

tfd = tensorflow_probability.distributions


class DistSummary(abc.ABC):
    """ Interface for distributions that implement tensorboard summaries
        Subclasses must implement create_summaries method that outputs a list of tensorboard summaries
    """

    @abc.abstractmethod
    def create_summaries(self):
        pass


class MultivariateNormal(tf.distributions.Distribution, DistSummary):
    def __init__(self, loc, cov_obj, validate_args=False, allow_nan_stats=True, name="MultivariateNormal"):
        """
        Multivariate Normal distribution using the Covariance class

        :param loc: The mean of the distribution [batch, n]
        :param cov_obj: A Covariance object for covariance matrices with shape [batch, n, n]
        :param validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        :param allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
        :param name: Python `str` name prefixed to Ops created by this class.
        """
        parameters = locals()
        with tf.name_scope(name, values=[loc]):
            self._loc = tf.identity(loc, name="loc")
            tf.assert_rank(self.loc, 2, message="loc must be a tensor of [batch size, event size]")
        self._cov_obj = cov_obj
        self._log_det_covar = None
        self.__r_precision_r = None
        graph_parents = [self._loc]

        assert isinstance(self._cov_obj, cov_rep.Covariance)

        super().__init__(dtype=self._loc.dtype, reparameterization_type=tf.distributions.FULLY_REPARAMETERIZED,
                         validate_args=validate_args, allow_nan_stats=allow_nan_stats, parameters=parameters,
                         graph_parents=graph_parents, name=name)

    @property
    def loc(self):
        """Distribution parameter for the mean."""
        return self._loc

    def _covariance(self):
        """Distribution parameter for covariance matrix."""
        return self.cov_obj.covariance

    def _stddev(self):
        return tf.sqrt(self.cov_obj.covariance_diag_part)

    def _variance(self):
        return self.cov_obj.covariance_diag_part

    @property
    def cov_obj(self):
        """The covariance matrix object."""
        return self._cov_obj

    @property
    def scale(self):
        return self.cov_obj.chol_covariance

    @property
    def log_det_covar(self):
        """The log determinant of the covariance matrix."""
        if self._log_det_covar is None:
            self._log_det_covar = self.cov_obj.log_det_covariance()
        return self._log_det_covar

    def _validate_input(self, x, batch_first=False):
        """
        Args:
            x: tensor of [batch size, num features] or [num samples, batch size, num features]

        Returns:
            tensor of [batch size, num features] or [batch size, num samples, num features]
        """
        x = tf.convert_to_tensor(x)
        if x.shape.ndims == 2:
            return x
        if x.shape.ndims == 3:
            if batch_first:
                return tf.transpose(x, [1, 0, 2])
            else:
                return x
        raise RuntimeError("Tensor must be rank 2 or 3, found {}".format(x.shape.ndims))

    @staticmethod
    def _expand_if_x_rank_3(value, x, axis):
        if x.shape.ndims == 2:
            return value
        else:
            return tf.expand_dims(value, axis=axis)

    def _r_precision_r(self, x):
        """
        Computes (x - mu) inv(Sigma) (x - mu)^T
        Args:
            x: tensor of [batch size, num samples, num features]

        Returns:
            A tensor of [num samples, batch size]
        """
        loc = self._expand_if_x_rank_3(self.loc, x, axis=1)

        # x_precision_x expects data in [batch size, num samples, num features]
        r_precision_r = self.cov_obj.x_precision_x(x - loc)

        if x.shape.ndims == 3:
            if r_precision_r.shape.ndims == 1:
                # x_precision_x removes sample dimensions if sample dim is 1, add it again
                r_precision_r = tf.expand_dims(r_precision_r, axis=1)

            # Transpose to [num samples, batch size]
            r_precision_r = tf.transpose(r_precision_r, [1, 0])
        self.__r_precision_r = r_precision_r
        return r_precision_r

    def _k_log_2_pi(self, x):
        with tf.name_scope('k_log_2_pi'):
            k = tf.cast(tf.shape(x)[-1], x.dtype)
            return k * np.log(2.0 * np.pi)

    def _log_prob(self, x):
        """
        log p(x) = - 0.5 * [ log(det(Sigma)) + (x - mu) inv(Sigma) (x - mu)^T + k log(2 pi) ]
        Args:
            x: tensor of [batch size, num features] or [num samples, batch size, num features]

        Returns:
            log p(x) tensor of [num samples, batch size, num features]
        """
        x = self._validate_input(x, batch_first=True)

        r_precision_r = self._r_precision_r(x)

        k_log_2_pi = self._k_log_2_pi(x)

        log_det_cov = self._expand_if_x_rank_3(self.log_det_covar, x, axis=0)

        return - 0.5 * (log_det_cov + r_precision_r + k_log_2_pi)

    def _prob(self, x):
        x = self._validate_input(x)
        return tf.exp(self._log_prob(x))

    def _sample_n(self, n, seed=None, epsilon=None):
        cov_sample = self.cov_obj.sample_covariance(num_samples=n, epsilon=epsilon, flatten_output=True)

        if cov_sample.shape.ndims == 2:
            # Sample covariance might remove the sample dim, add it again
            cov_sample = tf.expand_dims(cov_sample, axis=1)

        # cov_sample outputs a tensor of [batch size, num samples, num features]
        # reorder to [num samples, batch size, num features]
        cov_sample = tf.transpose(cov_sample, perm=(1, 0, 2))

        return tf.expand_dims(self.loc, axis=0) + cov_sample

    def sample_with_epsilon(self, sample_shape=(), epsilon=None, name="sample"):
        return self._call_sample_n(sample_shape, seed=None, name=name, epsilon=epsilon)

    def _batch_shape_tensor(self):
        return tf.shape(self.loc)[0]

    def _batch_shape(self):
        return self.loc.shape[0:1]

    def _event_shape_tensor(self):
        return tf.shape(self.loc)[1]

    def _event_shape(self):
        return self.loc.shape[1:]

    def _entropy(self):
        # 0.5 * log det(2 pi e Sigma) = 0.5 * (k * log(2 pi e) + log(det(Sigma)))
        k = tf.cast(tf.shape(self.loc)[1], self.loc.dtype)
        return 0.5 * (k * tf.log(2. * math.pi * math.e) + self.log_det_covar)

    def _mean(self):
        return self.loc

    def create_summaries(self):
        # Add summaries of log det(Sigma) and (x - mu) inv(Sigma) (x - mu)^T
        # but only if they already exists
        summaries = []
        if self._log_det_covar is not None:
            summaries.append(tf.summary.scalar(tensor=tf.reduce_mean(self.log_det_covar), name='log_det_covar'))

        if self.__r_precision_r is not None:
            summaries.append(tf.summary.scalar(tensor=tf.reduce_mean(self.__r_precision_r), name='r_precision_r'))

        return summaries


class MultivariateNormalDiag(MultivariateNormal):
    def __init__(self, loc, log_diag_covariance=None, log_diag_precision=None, validate_args=False,
                 allow_nan_stats=True, name="MultivariateNormalDiag"):
        parameters = locals()

        cov_obj = None

        if log_diag_covariance is not None:
            log_diag_covariance = tf.convert_to_tensor(log_diag_covariance)
            cov_obj = cov_rep.CovarianceDiag(log_diag_covariance=log_diag_covariance)

            assert log_diag_precision is None

        if log_diag_precision is not None:
            log_diag_precision = tf.convert_to_tensor(log_diag_precision)
            cov_obj = cov_rep.PrecisionDiag(log_diag_precision=log_diag_precision)

            assert log_diag_covariance is None

        if cov_obj is None:
            raise RuntimeError('Must provide log_diag_covariance or log_diag_precision')

        super().__init__(loc=loc, cov_obj=cov_obj, validate_args=validate_args, allow_nan_stats=allow_nan_stats,
                         name=name)
        self._parameters = parameters

    @property
    def log_diag_covariance(self):
        return self.cov_obj.log_diag_covariance

    @property
    def log_diag_precision(self):
        return self.cov_obj.log_diag_precision


class MultivariateNormalChol(MultivariateNormal):
    def __init__(self, loc, chol_covariance=None, chol_precision=None, validate_args=False, allow_nan_stats=True,
                 name="MultivariateNormalChol"):
        parameters = locals()

        cov_obj = None
        graph_parents = None

        if chol_covariance is not None:
            chol_covariance = tf.convert_to_tensor(chol_covariance)
            cov_obj = cov_rep.CovarianceCholesky(chol_covariance=chol_covariance)
            graph_parents = [chol_covariance]

            assert chol_precision is None

        if chol_precision is not None:
            chol_precision = tf.convert_to_tensor(chol_precision)
            cov_obj = cov_rep.PrecisionCholesky(chol_precision=chol_precision)
            graph_parents = [chol_precision]

            assert chol_covariance is None

        if cov_obj is None:
            raise RuntimeError('Must provide chol_covariance or chol_precision')

        super().__init__(loc=loc, cov_obj=cov_obj, validate_args=validate_args, allow_nan_stats=allow_nan_stats,
                         name=name)
        self._parameters = parameters


class MultivariateNormalPrecCholFilters(MultivariateNormal):
    def __init__(self, loc, weights_precision, filters_precision, log_diag_chol_precision, sample_shape,
                 validate_args=False, allow_nan_stats=True, name="MultivariateNormalCholFilters"):
        """
        Multivariate normal distribution for gray-scale images. Assumes an batch of images
            with shape [batch, img_w, img_h, 1]

            It models the distribution as N(mu, inv(L L.T)), where L is the Cholesky decomposition of the
            inverse of the covariance matrix.

        :param loc: The mean of the distribution [batch, img_w * img_h]
        :param weights_precision: Weight factors [batch, img_w, img_h, nb]
        :param filters_precision: Basis matrix (optionally it can be None) [nb, fs, fs, 1, 1]
        :param log_diag_chol_precision: The log values of the diagonal of L [batch, img_w * img_h]
        :param sample_shape:  A list or tensor indicating the shape [batch, img_w, img_h, 1]
        :param validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        :param allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
        :param name: Python `str` name prefixed to Ops created by this class.

        There are two modes of operation
        1) Without basis functions, which is set by filters_precision = None. This internally creates a
        filters_precision of identity matrix

        nb must be a squared number and weights precision must follow
        weights_precision[..., 0:nb2] = 0
        weights_precision[..., nb2] must be positive
         where nb2 = nb // 2

        Example of sparsity pattern for nb = 9, and looking at a slice [0, 0, :, :]

        | 0 0 0 0 d x x x x|
        | 0 0 0 0 d x x x x|
                ...
        | 0 0 0 0 d x x x x|

        where 'd' must be positive.

        Use example

            batch = 10
            img_w, img_h = 5, 5
            fs = 3
            nb2 = (fs**2) // 2

            loc = tf.zeros((batch, img_w * img_h))

            zeros = tf.zeros((batch, img_w, img_h, nb2))
            weights_precision_right = tf.random_normal((batch, img_w, img_h, nb2))
            log_diag_chol_precision = tf.random_normal((batch, img_w, img_h, 1))

            diag_chol_precision = tf.exp(log_diag_chol_precision)

            weights_precision = tf.concat([zeros, diag_chol_precision, weights_precision_right], axis=3)

            mvg_dist = MultivariateNormalPrecCholFilters(loc, weights_precision, None, log_diag_chol_precision,
                                                        (batch, img_w, img_h, 1))

        2) With a basis matrix, where weights_precision and filters_precision are given

        weights_precision must be positive

        filters_precision top half and left half of the center row must be zero
        and the center values must be positive.

        Example for fs = 3, and looking at a slice [0, :, :, 0, 0]

        | 0 0 0 |
        | 0 d x |
        | x x x |


        Use example

            batch = 10
            img_w, img_h = 5, 5
            fs = 3
            nb = 4
            fs2 = (fs ** 2) // 2

            loc = tf.zeros((batch, img_w * img_h))

            log_weights_precision = tf.random_normal((batch, img_w, img_h, nb))
            weights_precision = tf.exp(log_weights_precision)

            left_filters = tf.zeros((nb, fs2, 1, 1))
            log_center_filters = tf.random_normal((nb, 1, 1, 1))
            right_filters = tf.random_normal((nb, fs2, 1, 1))

            center_filters = tf.exp(log_center_filters)

            filters_precision = tf.concat([left_filters, center_filters, right_filters], axis=1)
            filters_precision = tf.reshape(filters_precision, (nb, fs, fs, 1, 1))

            log_center_filters = tf.reshape(log_center_filters, (1, 1, 1, -1))

            log_diag_chol_precision = tf.reduce_logsumexp(log_center_filters + log_weights_precision, axis=3)
            log_diag_chol_precision = tf.reshape(log_diag_chol_precision, (batch, img_w * img_h))

            mvg_dist = MultivariateNormalPrecCholFilters(loc, weights_precision, filters_precision,
                                                         log_diag_chol_precision, (batch, img_w, img_h, 1))

        Enforcing positiveness could be done in all cases by employing the exp operation.

        TODO: Add operations to validate args
        """
        parameters = locals()

        cov_obj = None

        with tf.name_scope(name=name):
            weights_precision = tf.convert_to_tensor(weights_precision)
            if filters_precision is not None:
                filters_precision = tf.convert_to_tensor(filters_precision)

            cov_obj = cov_rep.PrecisionConvCholFilters(weights_precision=weights_precision,
                                                       filters_precision=filters_precision,
                                                       sample_shape=sample_shape)
            cov_obj.log_diag_chol_precision = log_diag_chol_precision

        super().__init__(loc=loc, cov_obj=cov_obj, validate_args=validate_args, allow_nan_stats=allow_nan_stats,
                         name=name)
        self._parameters = parameters


class MultivariateNormalPrecCholFiltersDilation(MultivariateNormal):
    def __init__(self, loc, weights_precision, filters_precision, log_diag_chol_precision, sample_shape,
                 dilation_rates, validate_args=False, allow_nan_stats=True, name="MultivariateNormalCholFilters"):
        """
        Multivariate normal distribution for gray-scale images. Assumes an batch of images
            with shape [batch, img_w, img_h, 1]

            It models the distribution as N(mu, inv(L L.T)), where L is the Cholesky decomposition of the
            inverse of the covariance matrix.

        :param loc: The mean of the distribution [batch, img_w * img_h]
        :param weights_precision: A list of weight factors [batch, img_w, img_h, nb]
        :param filters_precision: A list of basis matrix (optionally it can be None) [nb, fs, fs, 1, 1]
        :param log_diag_chol_precision: The log values of the diagonal of L [batch, img_w * img_h]
        :param sample_shape:  A list or tensor indicating the shape [batch, img_w, img_h, 1]
        :param dilation_rates:  A list or tuple with dilation rates
        :param validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        :param allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
        :param name: Python `str` name prefixed to Ops created by this class.

        See examples in MultivariateNormalPrecCholFilters, where this class extends it to support a list of
        weights and filters, that are associated with the dilation rates.

        To easily evaluate the log_diag_chol_precision, concatenate all the filters and employ the
        logsumexp operator.

        log_weights_d = tf.concat(log_weights_precision, axis=3)

        fs2 = fs // 2

        log_filters = []
        for filter_i in log_filters_precision:
            log_filters.append(tf.reshape(filter_i[:, fs2, fs2, 0, 0]), (1, 1, 1, -1))

        log_filters_precision = tf.concat(log_filters_precision, axis=3)

        log_diag_chol_precision = tf.reduce_logsumexp(log_center_filters + log_weights_precision, axis=3)
        log_diag_chol_precision = tf.reshape(log_diag_chol_precision, (batch, img_w * img_h))
        """
        parameters = locals()

        cov_obj = None

        with tf.name_scope(name=name):
            for i, elem in enumerate(weights_precision):
                weights_precision[i] = tf.convert_to_tensor(elem)

            if filters_precision is not None:
                for i, elem in enumerate(filters_precision):
                    filters_precision[i] = tf.convert_to_tensor(elem)

            cov_obj = cov_rep.PrecisionDilatedConvCholFilters(weights_precision=weights_precision,
                                                              filters_precision=filters_precision,
                                                              sample_shape=sample_shape,
                                                              dilation_rates=dilation_rates)
            cov_obj.log_diag_chol_precision = log_diag_chol_precision

        super().__init__(loc=loc, cov_obj=cov_obj, validate_args=validate_args, allow_nan_stats=allow_nan_stats,
                         name=name)
        self._parameters = parameters


class IsotropicMultivariateNormal(MultivariateNormalDiag):
    """
    x ~ N(0,I)
    """

    def __init__(self, shape, dtype, validate_args=False, allow_nan_stats=True, name="IsotropicMultivariateNormal"):
        params = locals()
        with tf.name_scope(name=name):
            loc = tf.zeros(shape=shape, dtype=dtype, name='loc')
            log_diag_covar = tf.zeros(shape=shape, dtype=dtype, name='log_scale')

        super().__init__(loc=loc, log_diag_covariance=log_diag_covar, log_diag_precision=None,
                         validate_args=validate_args, allow_nan_stats=allow_nan_stats,
                         name="IsotropicMultivariateNormal")

        self._params = params

    def _log_prob(self, x):
        x = self._validate_input(x, batch_first=True)
        log_prob = - ll.neg_log_likelihood_unit_gaussian(predictions=x, mean_batch=False)
        if x.shape.ndims == 3:
            log_prob = tf.transpose(log_prob, [1, 0])
        return log_prob


class LogNormal(tfd.TransformedDistribution):
    """
    Given a random variable x, its log values are normally distributed
    log(x) = y ~ N(mu, sigma^2)
    x = exp(y)
    Thus, all samples x from this distribution are positive
    """

    def __init__(self, loc, scale, validate_args=False, allow_nan_stats=True, name="LogNormal"):
        params = locals()
        # y = exp(x) and y = N(mu, sigma^2)
        # x = log(y) and x = log N(mu, sigma^2)
        normal_dist = tf.distributions.Normal(loc=loc, scale=scale, validate_args=validate_args,
                                              allow_nan_stats=allow_nan_stats)
        super().__init__(distribution=normal_dist, bijector=tfd.bijectors.Exp(), name=name)

        self._parameters = params


@kullback_leibler.RegisterKL(MultivariateNormal, MultivariateNormal)
def _kl_mvnd_mvnd(a, b, name=None):
    """Batched KL divergence `KL(a || b)` for multivariate Normals."""
    return kl_divergence_mv_gaussian_v2(mu1=a.loc, mu2=b.loc, sigma1=a.cov_obj, sigma2=b.cov_obj, mean_batch=False,
                                        name=name)


@kullback_leibler.RegisterKL(MultivariateNormal, MultivariateNormalLinearOperator)
def _kl_mvnd_tfmvnd(a, b, name=None):
    """Batched KL divergence `KL(a || b)` for multivariate Normals, when "b" is a
    tf.contrib.distributions.MultivariateNormal* distribution"""
    b_cov_obj = cov_rep.CovarianceCholesky(chol_covariance=b.scale.to_dense())
    return kl_divergence_mv_gaussian_v2(mu1=a.loc, mu2=b.loc, sigma1=a.cov_obj, sigma2=b_cov_obj, mean_batch=False,
                                        name=name)


@kullback_leibler.RegisterKL(MultivariateNormalLinearOperator, MultivariateNormal)
def _kl_mvnd_tfmvnd(a, b, name=None):
    """Batched KL divergence `KL(a || b)` for multivariate Normals, when "a" is a
    tf.contrib.distributions.MultivariateNormal* distribution"""
    a_cov_obj = cov_rep.CovarianceCholesky(chol_covariance=a.scale.to_dense())
    return kl_divergence_mv_gaussian_v2(mu1=a.loc, mu2=b.loc, sigma1=a_cov_obj, sigma2=b.cov_obj, mean_batch=False,
                                        name=name)


@kullback_leibler.RegisterKL(MultivariateNormalDiag, IsotropicMultivariateNormal)
def _kl_diag_unit(a, b, name=None):
    """Special case of batched KL divergence `KL(a || b)` for multivariate Normals,
    where "a" is diagonal and "b" is the isotropic Gaussian distribution"""
    return kl_divergence_unit_gaussian(mu=a.loc, log_sigma_sq=a.log_diag_covariance, mean_batch=False, name=name)
