import mvg_distributions as mvg_dist
import numpy as np
import tensorflow as tf

np.random.seed(0)
dtype = np.float32

b = 3  # batch size
w, h = 5, 5  # Image height and width
k = 3  # 3x3 sparsity pattern

nb = k ** 2  # Number of basis
n = h * w  # feature size
img_shape = (b, w, h, 1)  # The image shape, not that only one channel images are supported

# Create two random mean vectors
loc = np.random.normal(size=(b, n)).astype(dtype)

# Create a random sparse Cholesky matrix
weights_precision = np.random.normal(size=(b, w, h, nb))  # Random matrix
weights_precision = weights_precision.astype(dtype)
weights_precision[..., 0:nb // 2] = 0  # Equivalent of upper triangular to zero
log_diag_chol_precision = weights_precision[..., nb // 2]  # Get the diagonal
weights_precision[..., nb // 2] = np.exp(log_diag_chol_precision)  # Exponentiate to remove log
log_diag_chol_precision = np.reshape(log_diag_chol_precision, (b, w * h))  # Flatten

# Create multivariate Gaussian distributions with sparse cholesky precision matrix
mvg = mvg_dist.MultivariateNormalPrecCholFilters(loc=loc, weights_precision=weights_precision,
                                                 filters_precision=None,
                                                 log_diag_chol_precision=log_diag_chol_precision,
                                                 sample_shape=img_shape)

# Create a batch of flattened random gray scale images
x0 = np.random.normal(size=(b, w * h)).astype(dtype)

# Evaluate the log probability of the random images
# This uses an efficient approach based on convolutions
log_prob = mvg.log_prob(x0)

# Evaluate
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

np_log_prob = sess.run(log_prob)

# Check that the value corresponds to the expected one
ground_truth_log_prob = np.array([-122.61849, -105.790565, -98.64827])
assert np.all(np.isclose(np_log_prob, ground_truth_log_prob))
