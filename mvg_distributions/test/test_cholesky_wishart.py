import unittest
import numpy as np
import tensorflow as tf
import tensorflow_probability
import mvg_distributions.covariance_representations as cov_rep
from mvg_distributions.cholesky_wishart import CholeskyWishart
from mvg_distributions.test.test_losses_base import LossesTestBase
from mvg_distributions.sqrt_gamma_gaussian import SqrtGammaGaussian, SparseSqrtGammaGaussian

tfd = tensorflow_probability.distributions
tfb = tensorflow_probability.bijectors


class TestCholeskyWishart(LossesTestBase):
    def setUp(self):
        super().setUp()
        self.x, self.x_cov_obj, self.sqrt_w_tfd, self.sqrt_w = self._create_single_sqrt_wishart_pair()

    def _create_single_sqrt_wishart_pair(self, add_sparsity_correction=False):
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
        sqrt_wishart = CholeskyWishart(df=df, log_diag_scale=log_diag_precision_prior,
                                       add_sparsity_correction=add_sparsity_correction)

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

        return x, x_cov_obj, sqrt_wishart_tfd, sqrt_wishart

    def test_log_prob(self):
        log_prob1 = self.sqrt_w_tfd.log_prob(self.x)
        log_prob2 = self.sqrt_w.log_prob(self.x)
        log_prob3 = self.sqrt_w.log_prob(self.x_cov_obj)
        log_prob4 = self.sqrt_w.log_prob(self.x_cov_obj.chol_precision)

        self._asset_allclose_tf_feed(log_prob1, log_prob2)
        self._asset_allclose_tf_feed(log_prob1, log_prob3)
        self._asset_allclose_tf_feed(log_prob1, log_prob4)

    def test_sample(self):
        x1 = self.sqrt_w_tfd.sample(seed=0)
        x2 = self.sqrt_w.sample(seed=0)

        self._asset_allclose_tf_feed(x1, x2)

    def test_sample_sparse(self):
        x1_dense = self.sqrt_w_tfd.sample(seed=0)
        x2_sparse = self.sqrt_w.sample_sparse(kw=3, seed=0)
        x2_sparse = x2_sparse.chol_precision

        # Ignore the values in x1_dense that are zero in x2_sparse
        valid_mask = tf.equal(x2_sparse, tf.zeros(tf.shape(x2_sparse)))
        valid_mask = tf.logical_not(valid_mask)
        x1_sparse = x1_dense * tf.cast(valid_mask, x1_dense.dtype)

        # FIXME
        # This test fails because internally the random normal distributions have
        # different sizes so the draw for the off-diagonal elements do not match
        # self._asset_allclose_tf_feed(x1_sparse, x2_sparse)

        # Checking that at least the diagonal part of the matrices do match
        self._asset_allclose_tf_feed(tf.matrix_diag_part(x1_sparse), tf.matrix_diag_part(x2_sparse))

    def test_gamma_gaussian_equivalent(self):
        # Check that the Cholesky-Wishart distribution is equivalent to a SquareRootGamma-Gaussian distribution
        sqrt_gamma_gaussian = SqrtGammaGaussian(df=self.sqrt_w.df, log_diag_scale=self.sqrt_w.log_diag_scale)

        x_with_log_diag = tf.matrix_set_diag(self.x, self.x_cov_obj.log_diag_chol_precision)
        log_prob_gg1 = sqrt_gamma_gaussian.log_prob(x_with_log_diag)

        x_with_log_diag = tf.matrix_set_diag(self.x_cov_obj.chol_precision, self.x_cov_obj.log_diag_chol_precision)
        log_prob_gg2 = sqrt_gamma_gaussian.log_prob(x_with_log_diag)

        log_prob_wishart = self.sqrt_w.log_prob(self.x_cov_obj)

        self._asset_allclose_tf_feed(log_prob_gg1, log_prob_wishart)
        self._asset_allclose_tf_feed(log_prob_gg2, log_prob_wishart)


class TestCholeskyWishartConv(TestCholeskyWishart):
    def _create_single_sqrt_wishart_pair(self, add_sparsity_correction=True):
        return super()._create_single_sqrt_wishart_pair(add_sparsity_correction=add_sparsity_correction)

    def test_log_prob(self):
        # The log prob contains the sparsity correction factor, thus it won't match the one from
        # the tensorflow Wishart distribution
        pass

    def test_gamma_gaussian_equivalent(self):
        # Check that the Cholesky-Wishart distribution with the sparsity correction factor is equivalent to a
        # SquareRootGamma-Gaussian distribution after removing the log probability of the zero terms in the off diagonal
        sqrt_gamma_gaussian = SqrtGammaGaussian(df=self.sqrt_w.df, log_diag_scale=self.sqrt_w.log_diag_scale)
        x_with_log_diag = tf.matrix_set_diag(self.x, self.x_cov_obj.log_diag_chol_precision)
        log_prob1_gamma = sqrt_gamma_gaussian._log_prob_sqrt_gamma(x_with_log_diag)

        log_prob1_normal = sqrt_gamma_gaussian.normal_dist.log_prob(self.x)
        off_diag_mask = self.x_cov_obj.np_off_diag_mask()
        log_prob1_normal = tf.reduce_sum(log_prob1_normal * off_diag_mask, axis=[1, 2])

        log_prob_gg = log_prob1_gamma + log_prob1_normal

        log_prob_wishart = self.sqrt_w.log_prob(self.x_cov_obj)

        self._asset_allclose_tf_feed(log_prob_gg, log_prob_wishart)

    def test_gamma_gaussian_equivalent_sparse(self):
        # Check that the sparse Cholesky-Wishart distribution is equivalent to a
        # SparseSquareRootGamma-Gaussian distribution
        sqrt_gamma_gaussian = SparseSqrtGammaGaussian(df=self.sqrt_w.df, log_diag_scale=self.sqrt_w.log_diag_scale)
        log_prob_gg = sqrt_gamma_gaussian.log_prob(self.x_cov_obj)

        log_prob_wishart = self.sqrt_w.log_prob(self.x_cov_obj)

        self._asset_allclose_tf_feed(log_prob_gg, log_prob_wishart)


if __name__ == '__main__':
    unittest.main()
