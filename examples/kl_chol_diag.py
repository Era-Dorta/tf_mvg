import mvg_distributions as mvg_dist
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

np.random.seed(0)
dtype = np.float32

b = 3  # batch size
n = 5  # feature size

# Create two random mean vectors
loc0 = np.random.normal(size=(b, n)).astype(dtype)
loc1 = np.random.normal(size=(b, n)).astype(dtype)

# Create a random diagonal covariance
log_var0 = np.random.normal(size=(b, n)).astype(dtype)

# Create a random Cholesky matrix
cholesky_covar = np.random.normal(size=(b, n, n))  # Random matrix
cholesky_covar = cholesky_covar.astype(dtype)
cholesky_covar = np.tril(cholesky_covar)  # Discard the upper triangular part
log_diag_covar = np.diagonal(cholesky_covar, axis1=1, axis2=2)  # Save the log_diagonal
for i in range(b):
    # Exponentiate the diagonal to remove the log
    cholesky_covar[i][np.diag_indices_from(cholesky_covar[i])] = np.exp(log_diag_covar[i])

# Create two multivariate Gaussian distributions with the mean and diagonal covariances
mvg0 = mvg_dist.MultivariateNormalDiag(loc=loc0, log_diag_covariance=log_var0)
mvg1 = mvg_dist.MultivariateNormalChol(loc=loc1, chol_covariance=cholesky_covar,
                                       log_diag_chol_covariance=log_diag_covar)

# Compute the KL divergence between them
kl_div = tfp.distributions.kl_divergence(mvg0, mvg1)

# Evaluate
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

np_kl_div = sess.run(kl_div)

# Check that the value corresponds to the expected one
ground_truth_kl = np.array([539.61863537, 269.49346953, 23.63169979])
assert np.all(np.isclose(np_kl_div, ground_truth_kl))
