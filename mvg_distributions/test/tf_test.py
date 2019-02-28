import os
import unittest
import numpy as np
import tensorflow as tf
import random


def set_rand_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def use_only_gpu(gpu_number=0):
    """ Hide all GPUS except gpu_number to tensorflow """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)


def get_default_session_config():
    """ Create a config for not preallocating all the GPU memory and allowing GPU-CPU operation placement """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    return config


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class TFTest(unittest.TestCase):
    # Common class for tensorflow unit testing
    def setUp(self):
        set_rand_seeds()
        use_only_gpu(0)
        self.rtol = 5.e-6  # Increase default tolerances, otherwise the tests fail
        self.atol = 5.e-6
        self.dtype = tf.float32

        self.tf_feed = None

    def tearDown(self):
        if self.__class__ == TFTest:
            return
        self.sess.close()
        tf.reset_default_graph()

    def _launch_session(self):
        if self.__class__ == TFTest:
            return
        # Launch the session
        config = get_default_session_config()
        self.sess = tf.Session(config=config)

    def _assert_allclose_tf_np(self, tf_input, np_output):
        tf_result = self.sess.run(tf_input, feed_dict=self.tf_feed)
        np.testing.assert_allclose(tf_result, np_output, rtol=self.rtol, atol=self.atol)

    def _asset_allclose_tf_tf(self, tf_input1, tf_input2):
        tf_result1, tf_result2 = self.sess.run([tf_input1, tf_input2], feed_dict=self.tf_feed)
        np.testing.assert_allclose(tf_result1, tf_result2, rtol=self.rtol, atol=self.atol)

    def _asset_allclose_tf_feed(self, tf_input, desired_val):
        if isinstance(desired_val, tf.Tensor):
            self._asset_allclose_tf_tf(tf_input, desired_val)
        else:
            self._assert_allclose_tf_np(tf_input, desired_val)
