import mvg_distributions as mvg_dist
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Set the session so that it doesn't take all available memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, _), (test_images, _) = fashion_mnist.load_data()

# Take only a subset of the dataset for faster training
n_subsample_dataset = 10000
train_images = train_images[0:n_subsample_dataset, ...]
test_images = test_images[0:n_subsample_dataset, ...]

# Convert from [0, 255] to [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

b = 64  # batch size
w, h = train_images.shape[1:3]  # Image height and width
k = 3  # 3x3 sparsity pattern

nb = k ** 2  # Number of basis
n = h * w  # feature size
img_shape = (b, w, h, 1)  # The image shape, not that only one channel images are supported


def encoder(input_tensor, n_z=256, n_h=256):
    # Encode image to low dim
    h0 = keras.layers.Flatten(input_shape=(w, h))(input_tensor)
    h1 = keras.layers.Dense(n_h, activation=tf.nn.relu)(h0)
    z = keras.layers.Dense(n_z, activation=tf.nn.relu, name='z')(h1)
    return z


def decoder_mean(z, n_h=256):
    # Decoder for the means of the Gaussian distribution
    # from low dim (n_z) to image space (w, h), use sigmoid as data is in [0, 1]
    l0 = keras.layers.Dense(n_h, activation=tf.nn.relu)(z)
    out_mean = keras.layers.Dense((w * h), activation=tf.nn.sigmoid)(l0)
    return keras.layers.Reshape((w, h), name='out_mean')(out_mean)


def decoder_covar(z, n_h=256):
    # Decoder for the covariance of the Gaussian distribution
    # from low dim (n_z) to image space with extra channels (w, h, (nb // 2) + 1)
    h0 = keras.layers.Dense(n_h, activation=tf.nn.relu)(z)

    chol_half_weights = keras.layers.Dense((w * h * ((nb // 2) + 1)), activation=None)(h0)
    chol_half_weights = keras.layers.Reshape((w, h, (nb // 2) + 1))(chol_half_weights)

    # The first channel contains the log_diagonal of the cholesky matrix
    log_diag_chol_precision = keras.layers.Lambda(lambda x: x[..., 0])(chol_half_weights)
    log_diag_chol_precision = keras.layers.Flatten(input_shape=(w, h), name='log_diag_chol_precision')(
        log_diag_chol_precision)

    def exponentiate_diag(x):
        x0 = tf.exp(x[..., 0:1])
        return tf.concat([x0, x[..., 1:]], axis=-1)

    # Exponentiate to remove the log from the diagonal
    chol_half_weights = keras.layers.Lambda(lambda x: exponentiate_diag(x))(chol_half_weights)

    def concat_with_zeros(x):
        zeros_shape = tf.shape(x)
        zeros_shape -= np.array([0, 0, 0, 1])
        zeros = tf.zeros(zeros_shape)
        zeros.set_shape((None, w, h, nb // 2))
        return tf.concat([zeros, x], axis=-1)

    # Concatenate with zeros to have a nb = k*k kernel per pixel, output size is (w, h, nb)
    chol_precision_weights = keras.layers.Lambda(lambda x: concat_with_zeros(x), name='chol_precision_weights')(
        chol_half_weights)

    return chol_precision_weights, log_diag_chol_precision


# Build the network, one encoder, and two decoders
model_input = keras.layers.Input(shape=(w, h))
z = encoder(model_input)
out_mean = decoder_mean(z)
chol_precision_weights, log_diag_chol_precision = decoder_covar(z)

############
# Train the encoder and the decoder for the means with a mean squared error loss
model_means = keras.Model(inputs=model_input, outputs=out_mean)

opt = keras.optimizers.Adam(lr=0.01)
model_means.compile(optimizer=opt,
                    loss='mse',
                    )

model_means.fit(train_images,
                {'out_mean': train_images},
                epochs=5, batch_size=b,
                )
#
############

# Freeze the weights for the encoder and decoder for the means
for i in range(len(model_means.layers)):
    model_means.layers[i].trainable = False


# Define the loss of the model as the negative log likelihood of a multivariate Gaussian distribution
def log_prob_loss(chol_precision_weights, log_diag_chol_precision):
    def loss(y_true, y_pred):
        y_mean = keras.layers.Flatten(input_shape=(w, h))(y_pred)
        mvg = mvg_dist.MultivariateNormalPrecCholFilters(loc=y_mean, weights_precision=chol_precision_weights,
                                                         filters_precision=None,
                                                         log_diag_chol_precision=log_diag_chol_precision,
                                                         sample_shape=img_shape)

        # The probability of the input data under the model
        y_true_flat = keras.layers.Flatten(input_shape=(w, h))(y_true)
        log_prob = mvg.log_prob(y_true_flat)
        neg_log_prob = - tf.reduce_mean(log_prob)

        return neg_log_prob

    return loss


############
# Train the decoder for the covariance keeping the rest of the model fixed
model_covar = keras.Model(inputs=model_input, outputs=[out_mean, chol_precision_weights, log_diag_chol_precision])

# The covariance part is a harder problem, so lower the learning rate and increase the number of epochs
opt.lr = 0.001
model_covar.compile(optimizer=opt,
                    loss={'out_mean': log_prob_loss(chol_precision_weights, log_diag_chol_precision)},
                    )

model_covar.fit(train_images,
                {'out_mean': train_images},
                epochs=10, batch_size=b,
                )
#
############

############
# Evaluate for some test images
eval_images = test_images[0:b]

predictions_mean, predictions_weights, predictions_log_diag = model_covar.predict(eval_images[0:b])

tf_predictions_mean = tf.convert_to_tensor(predictions_mean)
predictions_mean_flat = keras.layers.Flatten(input_shape=(w, h))(tf_predictions_mean)
mvg = mvg_dist.MultivariateNormalPrecCholFilters(loc=predictions_mean_flat, weights_precision=predictions_weights,
                                                 filters_precision=None,
                                                 log_diag_chol_precision=predictions_log_diag,
                                                 sample_shape=img_shape)

# Note that sampling is a slow and memory hungry operation as it builds the dense matrix
# representation of the Cholesky of the precision matrix (chol_precision_weights)
tf_prediction_sample = mvg.sample()
tf_prediction_sample = tf.reshape(tf_prediction_sample, (b, w, h))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

predictions_sample = sess.run(tf_prediction_sample)

# The samples could be out of range, clip to image space
predictions_sample = np.clip(predictions_sample, 0, 1)

# Number of images to plot
num_plot = 5
plt.figure()

e_str = '\u03B5'
mu_str = '\u03BC'

j = 1
for i in range(num_plot):
    plt.subplot(num_plot, 5, j)
    plt.imshow(eval_images[i], vmin=0, vmax=1, cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.title('input')
    j += 1

    plt.subplot(num_plot, 5, j)
    plt.imshow(eval_images[i] - predictions_mean[i], vmin=-1, vmax=1, cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.title('input - ' + mu_str)  # This is the real epsilon

    j += 1

    plt.subplot(num_plot, 5, j)
    plt.imshow(predictions_mean[i], vmin=0, vmax=1, cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.title(mu_str)
    j += 1

    plt.subplot(num_plot, 5, j)
    plt.imshow(predictions_sample[i] - predictions_mean[i], vmin=-1, vmax=1, cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.title(e_str)

    j += 1

    plt.subplot(num_plot, 5, j)
    plt.imshow(predictions_sample[i], vmin=0, vmax=1, cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.title(mu_str + '+' + e_str)
    j += 1

plt.show()
