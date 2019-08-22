import unittest

import tensorflow_probability

import mvg_distributions.covariance_representations as cov_rep
from mvg_distributions.log_likelihoods import *
from mvg_distributions.test.test_losses_base import LossesTestBase

tfd = tensorflow_probability.distributions


class LogLikelihoodsTest(LossesTestBase):
    def test_neg_log_likelihood_mv_gaussian_diag(self):
        x, mu, sigma_sq = self._random_normal_params(cov_rep.CovarianceDiag)

        tf_mvnd = tfd.MultivariateNormalDiag(loc=mu, scale_diag=np.sqrt(sigma_sq))
        tf_nll = - tf_mvnd.log_prob(x)

        covar = cov_rep.CovarianceDiag(log_diag_covariance=tf.log(sigma_sq))
        r_tf = tf.convert_to_tensor(x - mu)
        nll = neg_log_likelihood_mv_gaussian(r_tf, x_precision_x=covar.x_precision_x(r_tf),
                                             log_det_cov=covar.log_det_covariance(),
                                             mean_batch=False)

        self._asset_allclose_tf_feed(nll, tf_nll)

    def test_neg_log_likelihood_mv_gaussian_diag2(self):
        x, mu, sigma_sq = self._random_normal_params(cov_rep.CovarianceDiag)

        tf_mvnd = tfd.MultivariateNormalDiag(loc=mu, scale_diag=np.sqrt(sigma_sq))
        tf_nll = - tf_mvnd.log_prob(x)

        nll = neg_log_likelihood_diag_gaussian(predictions=tf.convert_to_tensor(mu),
                                               labels=tf.convert_to_tensor(x),
                                               log_sigma=tf.log(sigma_sq), mean_batch=False)

        self._asset_allclose_tf_feed(nll, tf_nll)

    def test_neg_log_likelihood_mv_gaussian_chol(self):
        x, mu, covariance = self._random_normal_params(cov_rep.CovarianceCholesky)
        chol_covariance = np.linalg.cholesky(covariance)

        tf_mvnd = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=covariance)
        tf_nll = - tf_mvnd.log_prob(x)

        covar = cov_rep.CovarianceCholesky(chol_covariance=tf.convert_to_tensor(chol_covariance))
        r_tf = tf.convert_to_tensor(x - mu)
        nll = neg_log_likelihood_mv_gaussian(r_tf, x_precision_x=covar.x_precision_x(r_tf),
                                             log_det_cov=covar.log_det_covariance(),
                                             mean_batch=False)

        self._asset_allclose_tf_feed(nll, tf_nll)

    def test_neg_log_likelihood_mv_gaussian_conv_filters_chol(self):
        x, mu, covariance, weights, filters, log_diag = self._random_normal_params(cov_rep.PrecisionConvCholFilters)

        tf_mvnd = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=covariance)
        tf_nll = - tf_mvnd.log_prob(x)

        img_size = int(np.sqrt(self.features_size))
        img_shape = (self.batch_size, img_size, img_size, 1)
        covar = cov_rep.PrecisionConvCholFilters(weights_precision=tf.convert_to_tensor(weights),
                                                 filters_precision=tf.convert_to_tensor(filters),
                                                 sample_shape=img_shape)
        covar.log_diag_chol_precision = log_diag

        r_tf = tf.convert_to_tensor(x - mu)
        r_tf_img = tf.reshape(r_tf, img_shape)
        nll = neg_log_likelihood_mv_gaussian(r_tf, x_precision_x=covar.x_precision_x(r_tf_img),
                                             log_det_cov=covar.log_det_covariance(),
                                             mean_batch=False)

        self._asset_allclose_tf_feed(nll, tf_nll)

    def test_neg_log_likelihood_unit_gaussian(self):
        # Create isotropic Gaussian N(0,I)
        x, mu, sigma_sq = self._random_normal_params(cov_rep.CovarianceDiag)
        mu[:] = 0
        sigma_sq[:] = 1

        tf_mvnd = tfd.MultivariateNormalDiag(loc=mu, scale_diag=np.sqrt(sigma_sq))
        tf_nll = - tf_mvnd.log_prob(x)

        nll = neg_log_likelihood_unit_gaussian(tf.convert_to_tensor(x), mean_batch=False)

        self._asset_allclose_tf_feed(nll, tf_nll)

    def test_neg_log_likelihood_spherical_gaussian(self):
        x, mu, sigma_sq = self._random_normal_params(cov_rep.CovarianceDiag)
        mu = tf.zeros((self.batch_size, self.features_size))
        sigma_sq = tf.random_gamma((self.batch_size, 1), 10.0, 0.1)
        sigma_sq = tf.concat([sigma_sq for i in range(self.features_size)], axis=1)
        x = mu + sigma_sq * tf.random_normal(sigma_sq.shape)

        tf_mvnd = tfd.MultivariateNormalDiag(loc=mu, scale_diag=tf.sqrt(sigma_sq))
        tf_nll = - tf_mvnd.log_prob(x)

        nll = neg_log_likelihood_spherical_gaussian(x, tf.log(sigma_sq[:, 0]), mean_batch=False)

        self._asset_allclose_tf_feed(nll, tf_nll)

    def test_neg_log_likelihood_bernoulli(self):
        # x ~ Bernoulli(mu)
        x, mu = self._random_normal_params(tf.distributions.Bernoulli)

        tf_bern = tf.distributions.Bernoulli(probs=mu)
        tf_nll = - tf.reduce_sum(tf_bern.log_prob(x), axis=1)

        nll = neg_log_likelihood_bernoulli(predictions=mu, labels=tf.convert_to_tensor(x), mean_batch=False)

        self._asset_allclose_tf_feed(nll, tf_nll)

    def test_neg_log_likelihood_bernoulli_logits(self):
        # x ~ Bernoulli(mu)
        x, mu = self._random_normal_params(tf.distributions.Bernoulli)
        logit_mu = - np.log(1 / mu - 1)  # Inverse of sigmoid

        tf_bern = tf.distributions.Bernoulli(probs=tf.sigmoid(logit_mu))
        tf_nll = - tf.reduce_sum(tf_bern.log_prob(x), axis=1)

        nll = neg_log_likelihood_bernoulli(predictions=None, logit_predictions=logit_mu, labels=tf.convert_to_tensor(x),
                                           mean_batch=False)

        self._asset_allclose_tf_feed(nll, tf_nll)

    def test_neg_log_likelihood_bernoulli_logits2(self):
        # x ~ Bernoulli(mu)
        x, mu = self._random_normal_params(tf.distributions.Bernoulli)
        logit_mu = - np.log(1 / mu - 1)  # Inverse of sigmoid
        x_tf = tf.convert_to_tensor(x)

        nll1 = neg_log_likelihood_bernoulli(predictions=tf.sigmoid(logit_mu), labels=x_tf, mean_batch=False)

        nll2 = neg_log_likelihood_bernoulli(predictions=None, logit_predictions=logit_mu, labels=x_tf, mean_batch=False)

        self._asset_allclose_tf_feed(nll1, nll2)


if __name__ == '__main__':
    unittest.main()
