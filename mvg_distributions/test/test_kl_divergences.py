import unittest
import numpy as np
import tensorflow_probability

import tensorflow as tf
import mvg_distributions.covariance_representations as cov_rep
from mvg_distributions.kl_divergence import kl_divergence_gaussian, kl_divergence_mv_gaussian, \
    kl_divergence_unit_gaussian, kl_divergence_mv_gaussian_v2
from mvg_distributions.test.test_losses_base import LossesTestBase

dist = tensorflow_probability.distributions


class KLDivergenceTest(LossesTestBase):

    def _convert_to_tensor(self, *args):
        out_vals = []
        for arg in args:
            out_vals.append(tf.convert_to_tensor(arg))
        return out_vals

    def test_kl_divergence_gaussian(self):
        _, mu1, sigma_sq1 = self._random_normal_params(cov_rep.CovarianceDiag)
        _, mu2, sigma_sq2 = self._random_normal_params(cov_rep.CovarianceDiag)

        tf_mvnd1 = dist.MultivariateNormalDiag(loc=mu1, scale_diag=np.sqrt(sigma_sq1))
        tf_mvnd2 = dist.MultivariateNormalDiag(loc=mu2, scale_diag=np.sqrt(sigma_sq2))

        tf_kldiv = dist.kl_divergence(tf_mvnd1, tf_mvnd2)

        mu1_tf, mu2_tf = self._convert_to_tensor(mu1, mu2)
        covar_kldiv = kl_divergence_gaussian(mu1=mu1_tf, log_sigma_sq1=tf.log(sigma_sq1), mu2=mu2_tf,
                                             log_sigma_sq2=tf.log(sigma_sq2), mean_batch=False)

        self._asset_allclose_tf_feed(tf_kldiv, covar_kldiv)

        tf_kldiv = tf.reduce_mean(tf_kldiv)
        covar_kldiv = kl_divergence_gaussian(mu1=mu1_tf, log_sigma_sq1=tf.log(sigma_sq1), mu2=mu2_tf,
                                             log_sigma_sq2=tf.log(sigma_sq2))
        self._asset_allclose_tf_feed(tf_kldiv, covar_kldiv)

    def test_kl_divergence_unit_gaussian(self):
        _, mu1, sigma_sq1 = self._random_normal_params(cov_rep.CovarianceDiag)
        mu2 = mu1.copy()
        sigma_sq2 = sigma_sq1.copy()
        mu2[:] = 0
        sigma_sq2[:] = 1

        tf_mvnd1 = dist.MultivariateNormalDiag(loc=mu1, scale_diag=np.sqrt(sigma_sq1))
        tf_mvnd2 = dist.MultivariateNormalDiag(loc=mu2, scale_diag=np.sqrt(sigma_sq2))

        tf_kldiv = dist.kl_divergence(tf_mvnd1, tf_mvnd2)

        mu1_tf = tf.convert_to_tensor(mu1)
        covar_kldiv = kl_divergence_unit_gaussian(mu=mu1_tf, log_sigma_sq=tf.log(sigma_sq1), mean_batch=False)
        self._asset_allclose_tf_feed(tf_kldiv, covar_kldiv)

        tf_kldiv = tf.reduce_mean(tf_kldiv)
        covar_kldiv = kl_divergence_unit_gaussian(mu=mu1_tf, log_sigma_sq=tf.log(sigma_sq1))
        self._asset_allclose_tf_feed(tf_kldiv, covar_kldiv)

    def test_kl_divergence_mv_gaussian_v2_full(self):
        _, mu1, covar1 = self._random_normal_params(cov_rep.CovarianceFull)
        _, mu2, covar2 = self._random_normal_params(cov_rep.CovarianceFull)

        tf_mvnd1 = dist.MultivariateNormalFullCovariance(loc=mu1, covariance_matrix=covar1)
        tf_mvnd2 = dist.MultivariateNormalFullCovariance(loc=mu2, covariance_matrix=covar2)

        tf_kldiv = dist.kl_divergence(tf_mvnd1, tf_mvnd2)

        mu1_tf, mu2_tf = self._convert_to_tensor(mu1, mu2)
        covar1 = cov_rep.CovarianceFull(covariance=tf.convert_to_tensor(covar1))
        covar2 = cov_rep.CovarianceFull(covariance=tf.convert_to_tensor(covar2))
        covar_kldiv = kl_divergence_mv_gaussian_v2(sigma1=covar1, sigma2=covar2, mu1=mu1_tf, mu2=mu2_tf,
                                                   mean_batch=False)

        self._asset_allclose_tf_feed(tf_kldiv, covar_kldiv)

        tf_kldiv = tf.reduce_mean(tf_kldiv)
        covar_kldiv = kl_divergence_mv_gaussian_v2(sigma1=covar1, sigma2=covar2, mu1=mu1_tf, mu2=mu2_tf)

        self._asset_allclose_tf_feed(tf_kldiv, covar_kldiv)

    def test_kl_divergence_mv_gaussian_v2_diag(self):
        _, mu1, sigma_sq1 = self._random_normal_params(cov_rep.CovarianceDiag)
        _, mu2, sigma_sq2 = self._random_normal_params(cov_rep.CovarianceDiag)

        tf_mvnd1 = dist.MultivariateNormalDiag(loc=mu1, scale_diag=np.sqrt(sigma_sq1))
        tf_mvnd2 = dist.MultivariateNormalDiag(loc=mu2, scale_diag=np.sqrt(sigma_sq2))

        tf_kldiv = dist.kl_divergence(tf_mvnd1, tf_mvnd2)

        mu1_tf, mu2_tf = self._convert_to_tensor(mu1, mu2)
        sigma_sq1 = cov_rep.CovarianceDiag(log_diag_covariance=tf.log(sigma_sq1))
        covar2 = cov_rep.CovarianceDiag(log_diag_covariance=tf.log(sigma_sq2))
        covar_kldiv = kl_divergence_mv_gaussian_v2(sigma1=sigma_sq1, sigma2=covar2, mu1=mu1_tf, mu2=mu2_tf,
                                                   mean_batch=False)

        self._asset_allclose_tf_feed(tf_kldiv, covar_kldiv)

        tf_kldiv = tf.reduce_mean(tf_kldiv)
        covar_kldiv = kl_divergence_mv_gaussian_v2(sigma1=sigma_sq1, sigma2=covar2, mu1=mu1_tf, mu2=mu2_tf)

        self._asset_allclose_tf_feed(tf_kldiv, covar_kldiv)

    def test_kl_divergence_mv_gaussian_v2_chol(self):
        _, mu1, covar1 = self._random_normal_params(cov_rep.CovarianceFull)
        chol_covariance1 = np.linalg.cholesky(covar1)
        _, mu2, covar2 = self._random_normal_params(cov_rep.CovarianceFull)
        chol_covariance2 = np.linalg.cholesky(covar2)

        tf_mvnd1 = dist.MultivariateNormalFullCovariance(loc=mu1, covariance_matrix=covar1)
        tf_mvnd2 = dist.MultivariateNormalFullCovariance(loc=mu2, covariance_matrix=covar2)

        tf_kldiv = dist.kl_divergence(tf_mvnd1, tf_mvnd2)

        mu1_tf, mu2_tf = self._convert_to_tensor(mu1, mu2)
        covar1 = cov_rep.CovarianceCholesky(chol_covariance=tf.convert_to_tensor(chol_covariance1))
        covar2 = cov_rep.CovarianceCholesky(chol_covariance=tf.convert_to_tensor(chol_covariance2))
        covar_kldiv = kl_divergence_mv_gaussian_v2(sigma1=covar1, sigma2=covar2, mu1=mu1_tf, mu2=mu2_tf,
                                                   mean_batch=False)

        self._asset_allclose_tf_feed(tf_kldiv, covar_kldiv)

        tf_kldiv = tf.reduce_mean(tf_kldiv)
        covar_kldiv = kl_divergence_mv_gaussian_v2(sigma1=covar1, sigma2=covar2, mu1=mu1_tf, mu2=mu2_tf)

        self._asset_allclose_tf_feed(tf_kldiv, covar_kldiv)

    def test_kl_divergence_mv_gaussian_conv_filters_chol(self):
        _, mu1, covar1, weights1, filters1, log_diag1 = self._random_normal_params(cov_rep.PrecisionConvCholFilters)
        _, mu2, covar2, weights2, filters2, log_diag2 = self._random_normal_params(cov_rep.PrecisionConvCholFilters)

        tf_mvnd1 = dist.MultivariateNormalFullCovariance(loc=mu1, covariance_matrix=covar1)
        tf_mvnd2 = dist.MultivariateNormalFullCovariance(loc=mu2, covariance_matrix=covar2)

        tf_kldiv = dist.kl_divergence(tf_mvnd1, tf_mvnd2)

        mu1_tf, weights1, filters1 = self._convert_to_tensor(mu1, weights1, filters1)
        mu2_tf, weights2, filters2 = self._convert_to_tensor(mu2, weights2, filters2)

        img_size = int(np.sqrt(self.features_size))
        img_shape = (self.batch_size, img_size, img_size, 1)
        covar1 = cov_rep.PrecisionConvCholFilters(weights_precision=weights1,
                                                  filters_precision=filters1,
                                                  sample_shape=img_shape)
        covar1.log_diag_chol_precision = log_diag1

        covar2 = cov_rep.PrecisionConvCholFilters(weights_precision=weights2,
                                                  filters_precision=filters2,
                                                  sample_shape=img_shape)
        covar2.log_diag_chol_precision = log_diag2

        covar_kldiv = kl_divergence_mv_gaussian_v2(sigma1=covar1, sigma2=covar2, mu1=mu1_tf, mu2=mu2_tf,
                                                   mean_batch=False)

        self._asset_allclose_tf_feed(tf_kldiv, covar_kldiv)

        tf_kldiv = tf.reduce_mean(tf_kldiv)
        covar_kldiv = kl_divergence_mv_gaussian_v2(sigma1=covar1, sigma2=covar2, mu1=mu1_tf, mu2=mu2_tf)

        self._asset_allclose_tf_feed(tf_kldiv, covar_kldiv)

    def test_kl_divergence_mv_gaussian(self):
        _, mu1, covar1 = self._random_normal_params(cov_rep.CovarianceFull)
        _, mu2, covar2 = self._random_normal_params(cov_rep.CovarianceFull)

        tf_mvnd1 = dist.MultivariateNormalFullCovariance(loc=mu1, covariance_matrix=covar1)
        tf_mvnd2 = dist.MultivariateNormalFullCovariance(loc=mu2, covariance_matrix=covar2)

        tf_kldiv = dist.kl_divergence(tf_mvnd1, tf_mvnd2)

        mu1_tf, mu2_tf, covar1, covar2 = self._convert_to_tensor(mu1, mu2, covar1, covar2)

        covar_kldiv = kl_divergence_mv_gaussian(sigma1=covar1, sigma2=covar2, mu1=mu1_tf, mu2=mu2_tf, mean_batch=False)

        self._asset_allclose_tf_feed(tf_kldiv, covar_kldiv)

        tf_kldiv = tf.reduce_mean(tf_kldiv)
        covar_kldiv = kl_divergence_mv_gaussian(sigma1=covar1, sigma2=covar2, mu1=mu1_tf, mu2=mu2_tf)

        self._asset_allclose_tf_feed(tf_kldiv, covar_kldiv)


if __name__ == '__main__':
    unittest.main()
