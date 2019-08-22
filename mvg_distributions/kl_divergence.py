import tensorflow as tf


def kl_divergence_unit_gaussian(mu, log_sigma_sq, mean_batch=True, name='kl_divergence_unit_gaussian'):
    # KL divergence between a multivariate Gaussian distribution with diagonal covariance and an
    # isotropic Gaussian distribution
    with tf.name_scope(name):
        latent_loss = -0.5 * tf.reduce_sum(1 + log_sigma_sq - tf.square(mu) - tf.exp(log_sigma_sq), axis=1)
        if mean_batch:
            latent_loss = tf.reduce_mean(latent_loss, axis=0)
            tf.losses.add_loss(latent_loss)
        return latent_loss


def kl_divergence_gaussian(mu1, log_sigma_sq1, mu2, log_sigma_sq2, mean_batch=True, name='kl_divergence_gaussian'):
    # KL divergence between two multivariate Gaussian distributions with diagonal covariance
    # All inputs must be matrices of [batch size, number of features]
    with tf.name_scope(name):
        k = tf.cast(tf.shape(mu1), mu1.dtype)[1]  # Number of features
        kl_div = 0.5 * (
            tf.reduce_sum(log_sigma_sq2, axis=1) - tf.reduce_sum(log_sigma_sq1, axis=1) -  # log(|sigma1|/|sigma2|)
            k +  # -k
            tf.reduce_sum(tf.exp(log_sigma_sq1 - log_sigma_sq2), axis=1) +  # trace(inv(sigma1) sigma2)
            tf.einsum('bi,bi->b', (mu2 - mu1) ** 2, tf.exp(-log_sigma_sq2))  # (mu2 - mu1)^T inv(sigma2) (mu2 - mu1)
        )
        if mean_batch:
            return tf.reduce_mean(kl_div, axis=0)
        else:
            return kl_div


def kl_divergence_mv_gaussian(mu1, mu2, sigma1, sigma2, mean_batch=True, name='kl_divergence_mv_gaussian'):
    # KL divergence between two multivariate Gaussian distributions
    # KL(N(mu1, sigma1) | N(mu2, sigma2))
    with tf.name_scope(name):
        from mvg_distributions.covariance_representations import CovarianceFull

        covar1 = CovarianceFull(covariance=sigma1)
        covar2 = CovarianceFull(covariance=sigma2)

        return kl_divergence_mv_gaussian_v2(sigma1=covar1, sigma2=covar2, mu1=mu1, mu2=mu2, mean_batch=mean_batch)


def kl_divergence_mv_gaussian_v2(sigma1, sigma2, mu1=None, mu2=None, mean_batch=True, name='kl_divergence_mv_gaussian'):
    # KL divergence between two multivariate Gaussian distributions
    # KL(N(mu1, sigma1) | N(mu2, sigma2))
    # sigma1 and sigma2 are Covariance objects
    # mu1 and mu2 tensors of [batch size, num features], if None, they are assumed to be zero
    with tf.name_scope(name):
        from mvg_distributions.covariance_representations import Covariance

        assert isinstance(sigma1, Covariance)
        assert isinstance(sigma2, Covariance)
        if mu1 is None:
            assert mu2 is None
        if mu2 is None:
            assert mu1 is None

        # This is equivalent to
        # tr_sig1_2 = tf.trace(tf.matmul(sigma2.precision, sigma1.covariance))
        # but it avoids doing the matmul for the off-diagonal elements
        tr_sig1_2 = tf.einsum('bij,bji->b', sigma2.precision, sigma1.covariance)

        k = tf.cast(tf.shape(sigma1.covariance)[1], sigma1.covariance.dtype)
        log_det = sigma2.log_det_covariance() - sigma1.log_det_covariance()

        if mu1 is not None:
            tf.assert_rank_at_least(mu1, 2)  # [Batch size, num features]
            tf.assert_rank_at_least(mu2, 2)

            sq_error = sigma2.x_precision_x(mu2 - mu1)

            kl_div = 0.5 * (tr_sig1_2 + sq_error - k + log_det)
        else:
            kl_div = 0.5 * (tr_sig1_2 - k + log_det)

        if mean_batch:
            kl_div = tf.reduce_mean(kl_div, axis=0)

        return kl_div
