import unittest

import numpy as np
import tensorflow as tf
import tensorflow_probability

import mvg_distributions.covariance_representations as cov_rep
from mvg_distributions.sqrt_gamma_gaussian import SqrtGammaGaussian, SparseSqrtGammaGaussian
from mvg_distributions.test.test_losses_base import LossesTestBase

tfd = tensorflow_probability.distributions
tfb = tensorflow_probability.bijectors


class TestSqrtGammaGaussian(LossesTestBase):
    def setUp(self):
        super().setUp()
        self.x, self.x_cov_obj, self.sqrt_w_tfd, self.sqrt_gamma_gaussian = self._create_single_sqrt_wishart_pair()

    def _create_single_sqrt_wishart_pair(self, add_sparse_gamma=False):
        # Create a random scale matrix for the Wishart distribution
        diag_precision_prior = np.abs(np.random.normal(size=(self.batch_size, self.features_size)))
        diag_precision_prior = diag_precision_prior.astype(self.dtype.as_numpy_dtype)
        precision_prior = np.zeros(shape=(self.batch_size, self.features_size, self.features_size),
                                   dtype=self.dtype.as_numpy_dtype)
        for i in range(self.batch_size):
            precision_prior[i][np.diag_indices_from(precision_prior[i])] = diag_precision_prior[i]
        log_diag_precision_prior = np.log(diag_precision_prior)

        # Create a random vector of degrees of freedom, whose values must be larger than features_size
        df = np.random.uniform(low=self.features_size, high=self.features_size * 10, size=self.batch_size)
        df = df.astype(self.dtype.as_numpy_dtype)

        # Create a square root Wishart distribution using bijectors
        wishart = tfd.Wishart(scale=precision_prior, df=df)
        cholesky_bijector = tfb.Invert(tfb.CholeskyOuterProduct())
        sqrt_wishart_tfd = tfd.TransformedDistribution(distribution=wishart, bijector=cholesky_bijector)

        # Create our custom square root Wishart distribution with the same parameters
        sqrt_gamma_gaussian = SqrtGammaGaussian(df=df, log_diag_scale=log_diag_precision_prior)
        if add_sparse_gamma:
            sparse_sqrt_gamma_gaussian = SparseSqrtGammaGaussian(df=df, log_diag_scale=log_diag_precision_prior)

        # Create a random Cholesky matrix to test the probability density functions
        _, __, x_covariance, x_weights, x_basis, log_diag = self._random_normal_params(cov_rep.PrecisionConvCholFilters)
        x = np.linalg.cholesky(np.linalg.inv(x_covariance))

        # Our custom square root Wishart is optimized to work with PrecisionConvCholFilters, it will measure
        # the pdf of the Cholesky of the Precision
        img_w = int(np.sqrt(self.features_size))
        sample_shape = tf.TensorShape((self.batch_size, img_w, img_w, 1))
        x_cov_obj = cov_rep.PrecisionConvCholFilters(weights_precision=tf.constant(x_weights),
                                                     filters_precision=tf.constant(x_basis),
                                                     sample_shape=sample_shape)
        x_cov_obj.log_diag_chol_precision = log_diag

        if add_sparse_gamma:
            return x, x_cov_obj, sqrt_wishart_tfd, sqrt_gamma_gaussian, sparse_sqrt_gamma_gaussian
        else:
            return x, x_cov_obj, sqrt_wishart_tfd, sqrt_gamma_gaussian

    def test_log_prob(self):
        # Test that square root Gamma Gaussian is the same as a Cholesky Wishart
        log_prob1 = self.sqrt_w_tfd.log_prob(self.x)

        x_with_log_diag = tf.matrix_set_diag(self.x, self.x_cov_obj.log_diag_chol_precision)
        log_prob2 = self.sqrt_gamma_gaussian.log_prob(x_with_log_diag)

        x_with_log_diag = tf.matrix_set_diag(self.x_cov_obj.chol_precision, self.x_cov_obj.log_diag_chol_precision)
        log_prob4 = self.sqrt_gamma_gaussian.log_prob(x_with_log_diag)

        self._asset_allclose_tf_feed(log_prob1, log_prob2)
        self._asset_allclose_tf_feed(log_prob1, log_prob4)

    def test_samples(self):
        # Test that square root Gamma Gaussian is the same as a Cholesky Wishart
        sample1 = self.sqrt_w_tfd.sample(seed=0)

        sample2 = self.sqrt_gamma_gaussian.sample(seed=0)
        sample2 = tf.matrix_set_diag(sample2, tf.exp(tf.matrix_diag_part(sample2)))

        self._asset_allclose_tf_feed(sample1, sample2)


class TestSparseSqrtGammaGaussian(TestSqrtGammaGaussian):
    def setUp(self):
        LossesTestBase.setUp(self)
        outputs = self._create_single_sqrt_wishart_pair(add_sparse_gamma=True)
        self.x, self.x_cov_obj, self.sqrt_w_tfd, self.sqrt_gamma_gaussian_dense, self.sqrt_gamma_gaussian = outputs

    def test_log_prob(self):
        # Test that square root Gamma Gaussian with dense matrices is the same as a Cholesky Wishart
        log_prob1 = self.sqrt_w_tfd.log_prob(self.x)

        log_prob2 = self.sqrt_gamma_gaussian.log_prob(self.x)
        log_prob4 = self.sqrt_gamma_gaussian.log_prob(self.x_cov_obj.chol_precision)

        self._asset_allclose_tf_feed(log_prob1, log_prob2)
        self._asset_allclose_tf_feed(log_prob1, log_prob4)

    def test_log_prob_sparse(self):
        # Test that square root Gamma Gaussian with sparse matrices is the same as a the dense version,
        # when the sparse elements are removed afterwards
        x_with_log_diag = tf.matrix_set_diag(self.x, self.x_cov_obj.log_diag_chol_precision)
        log_prob1_gamma = self.sqrt_gamma_gaussian_dense._log_prob_sqrt_gamma(x_with_log_diag)

        log_prob1_normal = self.sqrt_gamma_gaussian_dense.normal_dist.log_prob(self.x)
        off_diag_mask = self.x_cov_obj.np_off_diag_mask()  # Zero out off-diagonal terms
        log_prob1_normal = tf.reduce_sum(log_prob1_normal * off_diag_mask, axis=[1, 2])

        log_prob1 = log_prob1_gamma + log_prob1_normal

        log_prob2 = self.sqrt_gamma_gaussian.log_prob(self.x_cov_obj)

        self._asset_allclose_tf_feed(log_prob1, log_prob2)

    @unittest.skip
    def test_samples(self):
        pass


if __name__ == '__main__':
    unittest.main()
