import unittest

import numpy as np
import tensorflow as tf
import tensorflow_probability

import mvg_distributions.gamma as custom_dist
from mvg_distributions.test.test_losses_base import LossesTestBase

tf_dist = tensorflow_probability.distributions
tf_bij = tensorflow_probability.bijectors


class TestGamma(LossesTestBase):
    def setUp(self):
        super().setUp()
        self.x, self.log_x, self.gamma_base, self.gamma_test = self._create_single_gamma_pair()
        _, _, self.gamma_base2, self.gamma_test2 = self._create_single_gamma_pair()

    def _random_gamma_params(self):
        eps = 1e-5

        concentration = np.random.normal(size=(self.batch_size, self.features_size))
        concentration = np.abs(concentration) + eps
        concentration = concentration.astype(self.dtype.as_numpy_dtype)

        rate = np.random.normal(size=(self.batch_size, self.features_size))
        rate = np.abs(rate) + eps
        rate = rate.astype(self.dtype.as_numpy_dtype)

        inv_variance = np.random.normal(size=(self.batch_size, self.features_size))
        inv_variance = np.abs(inv_variance) + eps
        inv_variance = inv_variance.astype(self.dtype.as_numpy_dtype)

        return inv_variance, concentration, rate

    def _create_single_gamma_pair(self):
        x, concentration, rate = self._random_gamma_params()
        log_x = np.log(x)

        gamma_base = tf_dist.Gamma(concentration=concentration, rate=rate)
        gamma_test = custom_dist.Gamma(concentration=concentration, rate=rate)

        return x, log_x, gamma_base, gamma_test

    def test__log_prob(self):
        # i = 0, without sample, i = 1, with sample dim, i = 2, with sample dim == 2
        for i in range(3):
            if i == 1:
                self.x = np.expand_dims(self.x, axis=0)
                self.log_x = np.expand_dims(self.log_x, axis=0)
            if i == 2:
                self.x = np.concatenate((self.x, self.x * 2), axis=0)
                self.log_x = np.concatenate((self.log_x, np.log(np.exp(self.log_x) * 2)), axis=0)

            log_prob1 = self.gamma_test.log_prob(self.log_x)
            log_prob2 = self.gamma_base.log_prob(self.x)

            self._asset_allclose_tf_feed(log_prob1, log_prob2)

    def test__prob(self):
        for i in range(3):
            if i == 1:
                self.x = np.expand_dims(self.x, axis=0)
                self.log_x = np.expand_dims(self.log_x, axis=0)
            if i == 2:
                self.x = np.concatenate((self.x, self.x * 2), axis=0)
                self.log_x = np.concatenate((self.log_x, np.log(np.exp(self.log_x) * 2)), axis=0)

            prob1 = self.gamma_test.prob(self.log_x)
            prob2 = self.gamma_base.prob(self.x)

            self._asset_allclose_tf_feed(prob1, prob2)

    def test_stddev(self):
        scale1 = self.gamma_test.stddev()
        scale2 = self.gamma_base.stddev()

        self._asset_allclose_tf_feed(scale1, scale2)

    def test_variance(self):
        variance1 = self.gamma_test.variance()
        variance2 = self.gamma_base.variance()

        self._asset_allclose_tf_feed(variance1, variance2)

    def test__entropy(self):
        entropy1 = self.gamma_test.entropy()
        entropy2 = self.gamma_base.entropy()

        self._asset_allclose_tf_feed(entropy1, entropy2)

    def test__kl_div1(self):
        # Test kl between MultivariateNormal and MultivariateNormal
        kl1 = tf_dist.kl_divergence(self.gamma_test, self.gamma_test2)
        kl2 = tf_dist.kl_divergence(self.gamma_base, self.gamma_base2)

        self._asset_allclose_tf_feed(kl1, kl2)

    def test__kl_div2(self):
        # Test kl between MultivariateNormal and MultivariateNormalLinearOperator
        kl1 = tf_dist.kl_divergence(self.gamma_test, self.gamma_base2)
        kl2 = tf_dist.kl_divergence(self.gamma_base, self.gamma_test2)

        self._asset_allclose_tf_feed(kl1, kl2)

        # Check that it's also the same as between tf distributions
        kl3 = tf_dist.kl_divergence(self.gamma_base, self.gamma_base2)
        self._asset_allclose_tf_feed(kl1, kl3)

    def test_sample(self):
        log_sample1 = self.gamma_test.sample(seed=0)
        sample1 = tf.exp(log_sample1)

        sample2 = self.gamma_base.sample(seed=0)

        self._asset_allclose_tf_feed(sample1, sample2)


class TestSqrtGamma(TestGamma):
    def _create_single_gamma_pair(self):
        x, concentration, rate = self._random_gamma_params()
        log_x = np.log(x)

        gamma_base = tf_dist.Gamma(concentration=concentration, rate=rate)
        sqrt_bij = tf_bij.Invert(tf_bij.Square())  # Square root
        sqrt_gamma_base = tf_dist.TransformedDistribution(distribution=gamma_base,
                                                          bijector=sqrt_bij)

        sqrt_gamma_test = custom_dist.SqrtGamma(concentration=concentration, rate=rate)

        return x, log_x, sqrt_gamma_base, sqrt_gamma_test

    @unittest.skip("Not implemented in SqrtGamma")
    def test__entropy(self):
        pass

    @unittest.skip("Not implemented in SqrtGamma")
    def test__kl_div1(self):
        pass

    @unittest.skip("Not implemented in SqrtGamma")
    def test__kl_div2(self):
        pass

    @unittest.skip("Not implemented in SqrtGamma")
    def test_stddev(self):
        pass

    @unittest.skip("Not implemented in SqrtGamma")
    def test_variance(self):
        pass


if __name__ == '__main__':
    unittest.main()
