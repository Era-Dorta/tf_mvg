import os
import unittest

import numpy as np
import tensorflow as tf

import mvg_distributions.covariance_representations as cov_rep
from mvg_distributions.covariance_representations.tests.test_covariance_matrix import CovarianceTest


class CovarianceDiagTest(CovarianceTest):
    def setUp(self):
        super(CovarianceDiagTest, self).setUp()
        self.log_det_decomp = [cov_rep.DecompMethod.CUSTOM]

        self._create_tf_placeholders()

        self._create_covariance_instance()

        self.tf_feed = {self.diag: self.np_input}
        self._launch_session()

    def _create_tf_placeholders(self):
        self.diag = tf.placeholder(dtype=self.dtype, shape=self.input_shape)

    def _create_covariance_instance(self):
        # Create a diagonal covariance matrix with a random positive values
        self.cov_object = cov_rep.CovarianceDiag(log_diag_covariance=self.diag)

        self.np_input = np.random.normal(size=self.input_shape).astype(self.dtype.as_numpy_dtype)
        self.np_covariance = np.zeros(self.matrix_shape, dtype=self.dtype.as_numpy_dtype)
        for i in range(self.batch_size):
            self.np_covariance[i][np.diag_indices_from(self.np_covariance[i])] = np.exp(self.np_input[i])

        self._create_np_precision_cholesky()


class PrecisionDiagTest(CovarianceDiagTest):
    def _create_covariance_instance(self):
        self.inverse_chol_covariance = False
        # Create a diagonal precision matrix with a random positive values
        self.cov_object = cov_rep.PrecisionDiag(log_diag_precision=self.diag)

        self.np_input = np.random.normal(size=self.input_shape).astype(self.dtype.as_numpy_dtype)
        self.np_precision = np.zeros(self.matrix_shape, dtype=self.dtype.as_numpy_dtype)
        for i in range(self.batch_size):
            self.np_precision[i][np.diag_indices_from(self.np_precision[i])] = np.exp(self.np_input[i])

        self._create_np_precision_cholesky()


def _add_all_inv_methods():
    pass


if __name__ == os.path.splitext(os.path.basename(__file__))[0]:
    _add_all_inv_methods()

if __name__ == '__main__':
    _add_all_inv_methods()
    unittest.main()
