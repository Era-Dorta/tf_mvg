import tensorflow as tf
import tensorflow_probability
import numpy as np
import numpy.testing as npt
from tqdm import trange
from enum import Enum

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
        if seed is not None and epsilon is None:
            epsilon = self.cov_obj._get_epsilon(num_samples=n, seed=seed, epsilon=None)
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
    def __init__(self, loc, chol_covariance=None, chol_precision=None, log_diag_chol_covariance=None,
                 log_diag_chol_precision=None, validate_args=False, allow_nan_stats=True,
                 name="MultivariateNormalChol"):
        parameters = locals()

        cov_obj = None
        graph_parents = None

        if chol_covariance is not None:
            assert log_diag_chol_covariance is not None, 'Must provide log_diag of Cholesky matrix'

            chol_covariance = tf.convert_to_tensor(chol_covariance)
            log_diag_chol_covariance = tf.convert_to_tensor(log_diag_chol_covariance)

            cov_obj = cov_rep.CovarianceCholesky(chol_covariance=chol_covariance)
            cov_obj.log_diag_chol_covariance = log_diag_chol_covariance

            graph_parents = [chol_covariance, log_diag_chol_covariance]

            assert chol_precision is None

        if chol_precision is not None:
            assert log_diag_chol_precision is not None, 'Must provide log_diag of Cholesky matrix'

            chol_precision = tf.convert_to_tensor(chol_precision)
            log_diag_chol_precision = tf.convert_to_tensor(log_diag_chol_precision)

            cov_obj = cov_rep.PrecisionCholesky(chol_precision=chol_precision)
            cov_obj.log_diag_chol_precision = log_diag_chol_precision

            graph_parents = [chol_precision, log_diag_chol_precision]

            assert chol_covariance is None

        if cov_obj is None:
            raise RuntimeError('Must provide chol_covariance or chol_precision')

        super().__init__(loc=loc, cov_obj=cov_obj, validate_args=validate_args, allow_nan_stats=allow_nan_stats,
                         name=name)
        self._parameters = parameters


class MultivariateNormalPrecCholFilters(MultivariateNormal):
    class CondMeanSolver(Enum):
        SIMPLE = 0
        FAST = 1
        MEMORY = 2

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
            log_diag_chol_precision = tf.convert_to_tensor(log_diag_chol_precision)
            if filters_precision is not None:
                filters_precision = tf.convert_to_tensor(filters_precision)

            cov_obj = cov_rep.PrecisionConvCholFilters(weights_precision=weights_precision,
                                                       filters_precision=filters_precision,
                                                       sample_shape=sample_shape)
            cov_obj.log_diag_chol_precision = log_diag_chol_precision

        super().__init__(loc=loc, cov_obj=cov_obj, validate_args=validate_args, allow_nan_stats=allow_nan_stats,
                         name=name)
        self._parameters = parameters

    def sample_with_sparse_solver(self, sample_shape=(), seed=None, name="sample", sess=None, feed_dict=None):
        """ Sample from the distribution using a sparse solver. As TensorFlow lacks support for sparse
            solver this is done on CPU with scipy. That means that gradients are not back-propagated
            and the output of this operation is a numpy array """
        with tf.name_scope(name=name):
            sample_shape = tf.convert_to_tensor(sample_shape, dtype=tf.int32, name="sample_shape")
            sample_shape, n = self._expand_sample_shape_to_vector(sample_shape, "sample_shape")

            if not tf.executing_eagerly():
                tensor_list = []
                if isinstance(n, tf.Tensor):
                    tensor_list.append(n)
                if isinstance(sample_shape, tf.Tensor):
                    tensor_list.append(sample_shape)

                if not (tensor_list is []):
                    assert isinstance(sess, tf.Session)
                    tensor_list = sess.run(tensor_list, feed_dict=feed_dict)

                if isinstance(n, tf.Tensor) and isinstance(sample_shape, tf.Tensor):
                    n, sample_shape = tensor_list
                elif isinstance(n, tf.Tensor):
                    n = tensor_list[0]
                else:
                    sample_shape = tensor_list[0]

            epsilon = self.cov_obj.sample_with_sparse_solver(num_samples=n, seed=seed,
                                                             sess=sess, feed_dict=feed_dict)
            # Transpose to [num samples, batch size, num features]
            epsilon = np.transpose(epsilon, axes=(1, 0, 2))

            loc = tf.expand_dims(self.loc, axis=0)  # Add num samples dim
            if not tf.executing_eagerly():
                assert isinstance(sess, tf.Session)
                loc = sess.run(self.loc, feed_dict=feed_dict)

            samples = epsilon + loc

            # Reshape back to sample_shape + [num_features]
            batch_event_shape = np.shape(samples)[1:]
            final_shape = np.concatenate([sample_shape, batch_event_shape], 0)
            samples = np.reshape(samples, final_shape)

            return samples

    def covariance_with_sparse_solver(self, name="covariance", sess=None, feed_dict=None):
        """ Get the covariance matrix using a sparse solver. As TensorFlow lacks support for sparse
            solver this is done on CPU with scipy. That means that gradients are not back-propagated
            and the output of this operation is a numpy array """
        with tf.name_scope(name=name):
            return self.cov_obj.covariance_with_sparse_solver(sess=sess, feed_dict=feed_dict)

    def variance_with_sparse_solver(self, name="variance", sess=None, feed_dict=None,
                                    use_iterative_solver=True):
        """ Get the diagonal of the covariance matrix using a sparse solver. As TensorFlow lacks
            support for sparse solver this is done on CPU with scipy. That means that gradients
            are not back-propagated and the output of this operation is a numpy array.

             The iterative solver is slower but decreases the memory consumption to from
             n**2 to  n * (nf // 2), where n is the dimensionality of loc"""
        with tf.name_scope(name=name):
            return self.cov_obj.variance_with_sparse_solver(sess=sess, feed_dict=feed_dict,
                                                            use_iterative_solver=use_iterative_solver)

    def upper_chol_covariance_with_sparse_solver(self, name="covariance", sess=None, feed_dict=None,
                                                 use_sparse_format=False):
        """ Get the upper triangular cholesky of the covariance matrix using a sparse solver.
            S = M M^T, where S is the covariance matrix and M is an upper triangular matrix.
            This method returns the M matrix, [batch, num features, num features].

            As TensorFlow lacks support for sparse solver this is done on CPU with scipy. That means
            that gradients are not back-propagated and the output of this operation is a numpy array """
        with tf.name_scope(name=name):
            return self.cov_obj.upper_chol_covariance_with_sparse_solver(sess=sess, feed_dict=feed_dict,
                                                                         sparse_format=use_sparse_format)

    def conditional_mean(self, x_known, x_known_idx, name='conditional_mean', sess=None, feed_dict=None,
                         solver_method=CondMeanSolver.FAST):
        """ Evaluate the mean of the distribution given that we know the ground truth value of x_know,
            which is the element located at x_known_idx.

            Following notation from the book "Computer Vision: models and learning and inference"
            where x1 are the unknown variables, and x2 are the known ones.
            Equation 5.13 says that the new_mu1 = mu_1 + (covariance_21)^T (covariance_22)^{-1} (x_2 - mu_2)

            As TensorFlow lacks support for sparse solver covariance_21 and covariance_22 are evaluated on
            on CPU with scipy. That means that gradients are not back-propagated and the output of this
            operation is a numpy array

        :param x_known: the known value of the elements in x per batch, [batch, m]
        :param x_known_idx: the indices of the known values, [m]
        :param name: name for the operation, string
        :param sess: a tensorflow session, optional
        :param feed_dict: feed_dict for the tensorflow session, optional
        :param solver_method: the solver method to use:
            SIMPLE: slow and memory hungry method, but with very a simple implementation, used for verification
            FAST: faster method, but with a memory cost of O(n * n)
            MEMORY: method that is roughly twice as slow as FAST, but that it has a is a memory cost of
                O(n * nc + n * num_x_known)
        :return:
            It returns the updated mean of the distribution, [batch, num_features - m]
        """
        with tf.name_scope(name=name):
            assert isinstance(x_known, np.ndarray) and isinstance(x_known_idx, np.ndarray)
            assert x_known.ndim == 2 and x_known_idx.ndim == 1
            assert isinstance(solver_method, self.CondMeanSolver)

            np_loc = self.loc
            if not tf.executing_eagerly():
                assert isinstance(sess, tf.Session)
                np_loc = sess.run(self.loc, feed_dict=feed_dict)

            x2 = x_known
            x2_idx = x_known_idx

            # The x1 indices are the indices that are not in x2
            n = np_loc.shape[1]
            x1_idx = np.arange(n)
            x1_idx = np.delete(x1_idx, x2_idx)

            np_loc1 = np_loc[:, x1_idx]
            np_loc2 = np_loc[:, x2_idx]

            if solver_method == self.CondMeanSolver.SIMPLE:
                solver_fnc = self._conditional_mean_covariance_21_22_simple
            elif solver_method == self.CondMeanSolver.FAST:
                solver_fnc = self._conditional_mean_covariance_21_22_fast
            elif solver_method == self.CondMeanSolver.MEMORY:
                solver_fnc = self._conditional_mean_covariance_21_22_memory
            else:
                raise RuntimeError("Invalid solver method {}".format(solver_method))

            np_covariance_21, np_covariance_22 = solver_fnc(x1_idx=x1_idx, x2_idx=x2_idx,
                                                            sess=sess, feed_dict=feed_dict)

            # x1_new_mean = mu_1 + (covariance_21)^T (covariance_22)^{-1} (x_2 - mu_2)
            np_precision_22 = np.linalg.inv(np_covariance_22)
            np_x2_effect = np.matmul(np_precision_22, (x2 - np_loc2)[:, :, np.newaxis])
            np_x2_effect = np.matmul(np_covariance_21.transpose((0, 2, 1)), np_x2_effect)[:, :, 0]
            np_loc1 += np_x2_effect

            return np_loc1

    def _conditional_mean_covariance_21_22_simple(self, x1_idx, x2_idx, sess, feed_dict):
        # Compute the full covariance matrix, and then get covariance_21 and covariance_22
        np_covariance = self.cov_obj.covariance_with_sparse_solver(sess=sess, feed_dict=feed_dict)

        np_covariance_x2_rows = np_covariance[:, x2_idx]

        np_covariance_22 = np_covariance_x2_rows[:, :, x2_idx]
        np_covariance_21 = np_covariance_x2_rows[:, :, x1_idx]

        return np_covariance_21, np_covariance_22

    def _conditional_mean_covariance_21_22_fast(self, x1_idx, x2_idx, sess, feed_dict):
        # This is a faster implementation that avoids computing rows in Sigma that will not be used
        # Get the upper triangular Cholesky matrix of the Covariance, it comes in a sparse format of
        # ndarray with [batch size, num rows], where each row is another ndarray with the non zero elements
        # as the matrix is dense and upper triangular, no indices are needed
        np_chol_covariance = self.cov_obj.upper_chol_covariance_with_sparse_solver(sess=sess, feed_dict=feed_dict,
                                                                                   sparse_format=True)

        batch = np_chol_covariance.shape[0]
        num_x1 = len(x1_idx)
        num_x2 = len(x2_idx)
        n = num_x1 + num_x2
        dtype = np_chol_covariance[0][0].dtype

        # Get the rows containing the values for x2, equivalent to the line below for a dense matrix
        # np_chol_covariance_21_22 = np_chol_covariance[:, x2_idx]
        np_chol_covariance_21_22 = np.zeros([batch, num_x2, n], dtype=dtype)
        for i in range(num_x2):
            num_elems = np_chol_covariance[0][x2_idx[i]].shape[0]
            for b in range(batch):
                np_chol_covariance_21_22[b, i, -num_elems:] = np_chol_covariance[b][x2_idx[i]]

        # Matmul those rows to get the rows in the covariance for x2, equivalent to the line below
        # np_covariance_21_22 = np.matmul(np_chol_covariance_21_22, np_chol_covariance.transpose((0, 2, 1)))
        np_covariance_21_22 = np.zeros([batch, num_x2, n], dtype=dtype)
        for k in trange(num_x2, desc='dot_product_chol_covariance_to_covariance'):
            for b in range(batch):
                j = n
                for i in range(n):
                    # Dot product ignoring the zero elements as required per iteration in np_chol_covariance_21_22
                    np_covariance_21_22[b, k, i] = np.dot(np_chol_covariance_21_22[b, k, -j:],
                                                          np_chol_covariance[b][i])
                    j -= 1

        # Delete this cholesky matrix as soon as it is no longer needed
        del np_chol_covariance

        # Covariance_22 is the how the x2 elements affect each other, [batch, num_x2, num_x2]
        np_covariance_22 = np_covariance_21_22[:, :, x2_idx]
        # Covariance 21 is how x2 affects the x1 values, [batch, num_x2, num_x1]
        np_covariance_21 = np_covariance_21_22[:, :, x1_idx]

        return np_covariance_21, np_covariance_22

    def _conditional_mean_covariance_21_22_memory(self, x1_idx, x2_idx, sess, feed_dict):
        # This is implementation that avoids computing rows in Sigma that will not be used,
        # and uses a double solve approach to keep the memory costs in check
        np_covariance_21_22 = self.cov_obj.covariance_with_sparse_solver(sess=sess, feed_dict=feed_dict,
                                                                         only_x_rows=x2_idx)

        # Covariance_22 is the how the x2 elements affect each other, [batch, num_x2, num_x2]
        np_covariance_22 = np_covariance_21_22[:, :, x2_idx]
        # Covariance 21 is how x2 affects the x1 values, [batch, num_x2, num_x1]
        np_covariance_21 = np_covariance_21_22[:, :, x1_idx]

        return np_covariance_21, np_covariance_22


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


@tfd.RegisterKL(MultivariateNormal, MultivariateNormal)
def _kl_mvnd_mvnd(a, b, name=None):
    """Batched KL divergence `KL(a || b)` for multivariate Normals."""
    return kl_divergence_mv_gaussian_v2(mu1=a.loc, mu2=b.loc, sigma1=a.cov_obj, sigma2=b.cov_obj, mean_batch=False,
                                        name=name)


@tfd.RegisterKL(MultivariateNormal, MultivariateNormalLinearOperator)
def _kl_mvnd_tfmvnd(a, b, name=None):
    """Batched KL divergence `KL(a || b)` for multivariate Normals, when "b" is a
    tf.contrib.distributions.MultivariateNormal* distribution"""
    b_cov_obj = cov_rep.CovarianceCholesky(chol_covariance=b.scale.to_dense())
    return kl_divergence_mv_gaussian_v2(mu1=a.loc, mu2=b.loc, sigma1=a.cov_obj, sigma2=b_cov_obj, mean_batch=False,
                                        name=name)


@tfd.RegisterKL(MultivariateNormalLinearOperator, MultivariateNormal)
def _kl_mvnd_tfmvnd(a, b, name=None):
    """Batched KL divergence `KL(a || b)` for multivariate Normals, when "a" is a
    tf.contrib.distributions.MultivariateNormal* distribution"""
    a_cov_obj = cov_rep.CovarianceCholesky(chol_covariance=a.scale.to_dense())
    return kl_divergence_mv_gaussian_v2(mu1=a.loc, mu2=b.loc, sigma1=a_cov_obj, sigma2=b.cov_obj, mean_batch=False,
                                        name=name)


@tfd.RegisterKL(MultivariateNormalDiag, IsotropicMultivariateNormal)
def _kl_diag_unit(a, b, name=None):
    """Special case of batched KL divergence `KL(a || b)` for multivariate Normals,
    where "a" is diagonal and "b" is the isotropic Gaussian distribution"""
    return kl_divergence_unit_gaussian(mu=a.loc, log_sigma_sq=a.log_diag_covariance, mean_batch=False, name=name)
