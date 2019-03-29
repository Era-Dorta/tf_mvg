import mvg_distributions as mvg_dist
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

np.random.seed(0)
dtype = np.float32

b = 3  # batch size
n = 5  # feature size

# Create two random mean vectors
loc = np.random.normal(size=(b, n)).astype(dtype)

# Create two random diagonal covariances
log_var = np.random.normal(size=(b, n)).astype(dtype)

# Create two multivariate Gaussian distributions with the mean and diagonal covariances
mvg0 = mvg_dist.MultivariateNormalDiag(loc=loc, log_diag_covariance=log_var)
mvg1 = mvg_dist.IsotropicMultivariateNormal(shape=(b, n), dtype=np.float32)

# Compute the KL divergence between them
kl_div = tfp.distributions.kl_divergence(mvg0, mvg1)

# Evaluate
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

np_kl_div = sess.run(kl_div)

# Check that the value corresponds to the expected one
ground_truth_kl = np.array([7.55843158, 5.54717439, 3.80372443])
assert np.all(np.isclose(np_kl_div, ground_truth_kl))
