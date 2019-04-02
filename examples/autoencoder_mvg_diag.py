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
    # Decoder for the diagonal covariance of the Gaussian distribution
    # from low dim (n_z) to number of pixels (w * h)
    h0 = keras.layers.Dense(n_h, activation=tf.nn.relu)(z)
    log_diag_precision = keras.layers.Dense((w * h), activation=None, name='log_diag_chol_precision')(h0)
    return log_diag_precision


# Build the network, one encoder, and two decoders
model_input = keras.layers.Input(shape=(w, h))
z = encoder(model_input)
out_mean = decoder_mean(z)
log_diag_precision = decoder_covar(z)

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
def log_prob_loss(log_diag_precision):
    def loss(y_true, y_pred):
        out_mean = tf.layers.flatten(y_pred)
        mvg = mvg_dist.MultivariateNormalDiag(loc=out_mean, log_diag_precision=log_diag_precision)

        # The probability of the input data under the model
        log_prob = mvg.log_prob(tf.layers.flatten(y_true))
        neg_log_prob = - tf.reduce_mean(log_prob)

        return neg_log_prob

    return loss


############
# Train the decoder for the covariance keeping the rest of the model fixed
model_covar = keras.Model(inputs=model_input, outputs=[out_mean, log_diag_precision])

# Use same learning rate and number of epochs as in learn_mvg_sparse_matrix.py example
opt.lr = 0.001
model_covar.compile(optimizer=opt,
                    loss={'out_mean': log_prob_loss(log_diag_precision)},
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

predictions_mean, predictions_log_diag = model_covar.predict(eval_images[0:b])

tf_predictions_mean = tf.convert_to_tensor(predictions_mean)
predictions_mean_flat = keras.layers.Flatten(input_shape=(w, h))(tf_predictions_mean)
mvg = mvg_dist.MultivariateNormalDiag(loc=predictions_mean_flat, log_diag_precision=predictions_log_diag)

# Get a sample from the predicted distribution
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
