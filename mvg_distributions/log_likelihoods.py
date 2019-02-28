import numpy as np
import tensorflow as tf


def _batch_squared_error_with_covariance(predictions, labels, inv_covariance,
                                         name='batch_squared_error_with_covariance', out_name=None):
    with tf.name_scope(name):
        if labels is None:
            labels_m_pred = predictions
        else:
            labels_m_pred = labels - predictions

        if labels_m_pred.shape.ndims == 2:
            # Add number of samples dimension
            labels_m_pred = tf.expand_dims(labels_m_pred, 1)

        # Matrix with the left side: (x-mu) Sigma^-1
        if len(inv_covariance.shape) == 2:
            # Shared covariance matrix
            left_side = tf.matmul(labels_m_pred, inv_covariance)
        else:
            # A covariance matrix per element in the batch
            left_side = tf.matmul(labels_m_pred, inv_covariance)

        # Explicitly multiply each element and sum over the rows, i.e. batch-wise dot product
        batch_mse = tf.multiply(left_side, labels_m_pred)
        batch_mse = tf.reduce_sum(batch_mse, axis=2)  # Error per sample
        if batch_mse.shape[1].value == 1:
            batch_mse = tf.squeeze(batch_mse, axis=1, name=out_name)  # Remove sample dimension

        return batch_mse


def squared_error_with_covariance(predictions, labels, inv_covariance, name='squared_error_with_covariance'):
    """ Loss = (labels - predictions) inv(Covariance) (labels - predictions)^t
        Predictions and labels are batches of NxK data and
        Covariance is a matrix of (K*K)x(K*K)
        This function can only be used if the covariance is shared for all the labels
    """
    with tf.name_scope(name):
        batch_mse = _batch_squared_error_with_covariance(predictions, labels, inv_covariance)

        # Mean over batch
        return tf.reduce_mean(batch_mse)


def _get_inv_covariance(inv_covariance=None, covariance=None, dtype=tf.float32, name='inv_covariance'):
    # Compute the inverse of a covariance matrix
    if inv_covariance is None:
        if covariance is not None:
            if isinstance(covariance, tf.Tensor):
                with tf.name_scope(name):
                    inv_covariance = tf.matrix_inverse(covariance)
                    inv_covariance = tf.cast(inv_covariance, dtype)
            else:
                inv_covariance = np.linalg.inv(covariance)
                inv_covariance = inv_covariance.astype(dtype.as_numpy_dtype)
        else:
            raise RuntimeError("Must provide covariance matrix if inv(Sigma) is not given")

    return inv_covariance


def _log_det_with_cholesky(cholesky, dtype=tf.float32, name='compute_log_det_with_cholesky', out_name=None):
    with tf.name_scope(name):
        log_det_cov = tf.log(tf.matrix_diag_part(cholesky))
        if len(cholesky.shape) == 2:
            log_det_cov = 2.0 * tf.reduce_sum(log_det_cov)
        else:
            log_det_cov = 2.0 * tf.reduce_sum(log_det_cov, axis=1)
        log_det_cov = tf.cast(log_det_cov, dtype, name=out_name)
    return log_det_cov


def _get_log_det_covariance(log_det_cov, covariance, dtype=tf.float32, name='log_det_covariance'):
    # Compute the log determinant a covariance matrix or matrices
    if log_det_cov is None:
        if covariance is not None:
            if isinstance(covariance, tf.Tensor):
                with tf.name_scope(name):
                    log_det_cov = _log_det_with_cholesky(tf.cholesky(covariance), dtype)
            else:
                _, log_det_cov = np.linalg.slogdet(covariance)
                log_det_cov = log_det_cov.astype(dtype.as_numpy_dtype)
        else:
            raise RuntimeError("Must provide covariance matrix if log(det(Sigma)) is not given")

    return log_det_cov

def _get_k(x, name='k'):
    # Compute the dimensionality of x
    with tf.name_scope(name):
        # Dimensionality of x is known
        k = x.shape.as_list()[-1]
        if k is None:
            # Dimensionality of x is not known
            k = tf.cast(tf.shape(x)[-1], x.dtype)
        return k

def _get_k_log_2_pi(k_log_2_pi, x, name='k_log_2_pi'):
    # Compute k * log(2*pi), where k is the dimensionality of x
    if k_log_2_pi is None:
        with tf.name_scope(name):
            k = _get_k(x)
            k_log_2_pi = k * np.log(2.0 * np.pi)
    return k_log_2_pi


def neg_log_likelihood_mv_gaussian(predictions, labels=None, covariance=None, inv_covariance=None, log_det_cov=None,
                                   k_log_2_pi=None, x_precision_x=None, mean_batch=True,
                                   name='neg_log_likelihood_mv_gaussian'):
    """ Negative log likelihood of a multivariate Gaussian distribution
        nll = 0.5 * [ log(|Sigma|) + (labels - predictions) inv(Sigma) (labels - predictions)^T + k*log(2*pi) ]
        Predictions and labels are batches of NxK data and
        Covariance is a matrix (K*K)x(K*K) or a batch of matrices of Nx(K*K)x(K*K)
        If inv_covariance or log_det_cov are not given, they will be computed from covariance
        If both are given, covariance will not be used
    """
    tf.assert_rank(predictions, 2, message="predictions must have rank 2")
    if labels is not None:
        tf.assert_rank(labels, 2, message="labels must have rank 2")

    with tf.name_scope(name):
        log_det_cov = _get_log_det_covariance(log_det_cov, covariance, predictions.dtype)
        if x_precision_x is None:
            inv_covariance = _get_inv_covariance(inv_covariance, covariance, predictions.dtype)
            is_batch = len(inv_covariance.shape) == 3
        else:
            x_precision_x.shape.assert_is_compatible_with(log_det_cov.shape)
            is_batch = len(x_precision_x.shape) == 1
        k_log_2_pi = _get_k_log_2_pi(k_log_2_pi, predictions)

        if x_precision_x is None:
            x_precision_x = _batch_squared_error_with_covariance(predictions, labels, inv_covariance)

        if not is_batch:
            # Shared covariance matrix
            if mean_batch:
                x_precision_x = tf.reduce_mean(x_precision_x)
            return 0.5 * (log_det_cov + x_precision_x + k_log_2_pi)
        else:
            # Batch of covariance matrices
            if mean_batch:
                return 0.5 * (tf.reduce_mean(log_det_cov + x_precision_x) + k_log_2_pi)
            else:
                return 0.5 * (log_det_cov + x_precision_x + k_log_2_pi)


def _batch_squared_error_with_diag_covariance(predictions, labels, log_diag_covariance,
                                              name='batch_squared_error_with_diag_covariance'):
    """ Loss = (labels - predictions) inv(Covariance) (labels - predictions)^t
        Predictions and labels are batches of NxK data and
        Covariance is a vector NxK with the log(Covariance) diagonal values
    """
    with tf.name_scope(name):
        if labels is None:
            squared_error = tf.square(predictions)
        else:
            squared_error = tf.squared_difference(labels, predictions)
        squared_error *= (1.0 / tf.exp(log_diag_covariance))
        return tf.reduce_sum(squared_error, axis=1)


def neg_log_likelihood_diag_gaussian(predictions, log_sigma, labels=None, mean_batch=True,
                                     name='neg_log_likelihood_diag_gaussian'):
    """ Negative log likelihood of a multivariate diagonal Gaussian distribution
        nll = 0.5 * [ log(|Sigma|) + (labels - predictions) inv(Sigma) (labels - predictions)^T + k*log(2*pi) ]
        Predictions and labels are batches of NxK data and
        Sigma is vector with the diagonal NxK
    """
    with tf.name_scope(name):
        tf.assert_rank(predictions, 2, message="predictions must be [batch dim, num features]")
        tf.assert_rank(log_sigma, 2, message="log_sigma must have rank [batch dim, num features**2]")
        if labels is not None:
            tf.assert_rank(labels, 2, message="labels must have rank [batch dim, num features]")

        k_log_2_pi = _get_k_log_2_pi(None, predictions)
        log_det_sigma = tf.reduce_sum(log_sigma, axis=1)
        squared_error = _batch_squared_error_with_diag_covariance(predictions=predictions, labels=labels,
                                                                  log_diag_covariance=log_sigma)
        if mean_batch:
            return 0.5 * (tf.reduce_mean(log_det_sigma + squared_error) + k_log_2_pi)
        else:
            return 0.5 * (log_det_sigma + squared_error + k_log_2_pi)


def neg_log_likelihood_unit_gaussian(predictions, labels=None, mean_batch=True,
                                     name='neg_log_likelihood_unit_gaussian'):
    """ Negative log likelihood of a unit Gaussian distribution
        nll = 0.5 * [ (labels - predictions) (labels - predictions)^T + k*log(2*pi) ]
        Predictions and labels are batches of Nx[S]xK data
    """
    with tf.name_scope(name):
        tf.assert_rank_in(predictions, [2, 3], message="predictions must have rank 2 or 3")
        if labels is not None:
            tf.assert_rank_in(labels, [2, 3], message="labels must have rank 2 or 3")

        k_log_2_pi = _get_k_log_2_pi(None, predictions)
        if labels is None:
            squared_error = tf.square(predictions)
        else:
            squared_error = tf.squared_difference(labels, predictions)
        squared_error = tf.reduce_sum(squared_error, axis=-1)
        if mean_batch:
            squared_error = tf.reduce_mean(squared_error)
        return 0.5 * (squared_error + k_log_2_pi)

def neg_log_likelihood_spherical_gaussian(predictions, log_variance, labels=None, mean_batch=True,
                                     name='neg_log_likelihood_unit_gaussian'):
    """ Negative log likelihood of a Gaussian distribution with a single log variance parameter per image
        Predictions and labels are batches of Nx[S]xK data
    """
    with tf.name_scope(name):
        tf.assert_rank(predictions, 2, message="predictions must have rank 2")
        if labels is not None:
            tf.assert_rank(labels, 2, message="labels must have rank 2")
        tf.assert_rank(log_variance, 1, message="log_variance must have rank 1")

        k_log_2_pi = _get_k_log_2_pi(None, predictions)
        if labels is None:
            squared_error = tf.square(predictions)
        else:
            squared_error = tf.squared_difference(labels, predictions)
        squared_error *= tf.reshape(tf.exp(log_variance*-1.0), [-1,1])
        squared_error = tf.reduce_sum(squared_error, axis=-1)
        if mean_batch:
            squared_error = tf.reduce_mean(squared_error) + tf.reduce_mean(log_variance)
        else:
            squared_error += _get_k(predictions)*log_variance
        return 0.5 * (squared_error + k_log_2_pi)


def neg_log_likelihood_bernoulli(predictions, labels, logit_predictions=None, mean_batch=True,
                                 name='neg_log_likelihood_bernoulli'):
    """ Negative log likelihood of a Bernoulli distribution
        nll = labels * log(predictions) + (1 - labels) * log(1 - predictions)
        Predictions and labels are batches of NxK data
        Predictions must be probabilities values in the range [0, 1], and labels should be binary values [0, 1]
        Optionally, predictions can be None if logit_predictions is given, such that
        predictions = tf.sigmoid(logit_predictions)
    """
    with tf.name_scope(name):
        if predictions is None:
            assert logit_predictions is not None
            tf.assert_rank(logit_predictions, 2, message="logit_predictions must have rank 2")

        if logit_predictions is None:
            assert predictions is not None
            tf.assert_rank(predictions, 2, message="predictions must have rank 2")

        tf.assert_rank(labels, 2, message="labels must have rank 2")

        if logit_predictions is None:
            log_prob = labels * tf.log(predictions) + (1 - labels) * tf.log(1 - predictions)
            neg_log_prob = - log_prob
        else:
            neg_log_prob = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logit_predictions)

        neg_log_prob = tf.reduce_sum(neg_log_prob, axis=1)

        if mean_batch:
            neg_log_prob = tf.reduce_mean(neg_log_prob)

        return neg_log_prob
