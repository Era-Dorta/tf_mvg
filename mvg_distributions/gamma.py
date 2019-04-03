import tensorflow as tf
import tensorflow_probability

tfd = tensorflow_probability.distributions


class Gamma(tfd.Gamma):
    """ Gamma distribution where the log_prob can be evaluated with a log_x value, avoids doing log(exp(log_x))
       to get the log(x) value needed for the log_prob """

    def _log_prob(self, log_x):
        return self._log_unnormalized_prob(log_x) - self._log_normalization()

    def _log_unnormalized_prob(self, log_x):
        return (self.concentration - 1.) * log_x - self.rate * tf.exp(log_x)

    def _sample_n(self, n, seed=None):
        # Sample a log(value)
        x = super()._sample_n(n, seed=seed)
        return tf.log(x)
