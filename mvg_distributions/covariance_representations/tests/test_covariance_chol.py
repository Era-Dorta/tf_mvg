import os
import unittest

import tensorflow as tf

import mvg_distributions.covariance_representations as cov_rep
from mvg_distributions.covariance_representations.tests.test_covariance_matrix import CovarianceTest, \
    declare_inv_method_test_classes


class CovarianceCholTest(CovarianceTest):
    def setUp(self):
        super(CovarianceCholTest, self).setUp()
        self._create_tf_placeholders()

        self._create_covariance_instance()

        self.tf_feed = {self.tf_input: self.np_input}

        self._launch_session()

    def _create_tf_placeholders(self):
        self.tf_input = tf.placeholder(dtype=self.dtype, shape=self.matrix_shape)

    def _create_covariance_instance(self):
        self.cov_object = cov_rep.CovarianceCholesky(chol_covariance=self.tf_input,
                                                     inversion_method=self.inversion_method)

        self.np_covariance = self.create_random_covariance(self.batch_size, self.features_size, self.dtype, False)

        self._create_np_precision_cholesky()

        self.np_input = self.np_chol_covariance

    def test_sampling_covariance_sqrt(self):
        self.rtol, self.atol = 5e-5, 5e-5
        super(CovarianceCholTest, self).test_sampling_covariance_sqrt()

    def test_x_precision_x_batch(self):
        super(CovarianceCholTest, self).test_x_precision_x_batch()


class PrecisionCholTest(CovarianceCholTest):
    def _create_covariance_instance(self):
        self.inverse_chol_covariance = False
        super(PrecisionCholTest, self)._create_covariance_instance()
        self.cov_object = cov_rep.PrecisionCholesky(chol_precision=self.tf_input,
                                                    inversion_method=self.inversion_method)

        self.np_input = self.np_chol_precision


def _add_all_inv_methods():
    class_list = list()
    class_list.append(CovarianceCholTest)
    class_list.append(PrecisionCholTest)

    declare_inv_method_test_classes(class_list, globals())


if __name__ == os.path.splitext(os.path.basename(__file__))[0]:
    _add_all_inv_methods()

if __name__ == '__main__':
    _add_all_inv_methods()
    unittest.main()
