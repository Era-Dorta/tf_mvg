import os
import unittest

import numpy as np
import tensorflow as tf

import mvg_distributions.covariance_representations as cov_rep
from mvg_distributions.covariance_representations.tests.test_covariance_matrix import CovarianceTest, \
    declare_inv_method_test_classes


class CovarianceEigTest(CovarianceTest):
    reduced_eig_vec = False

    def setUp(self):
        super(CovarianceEigTest, self).setUp()
        if self.reduced_eig_vec:
            self.n_eig = self.features_size // 2
        else:
            self.n_eig = self.features_size
        self._create_tf_placeholders()

        self._create_covariance_instance()

        self.tf_feed = {self.eig_vec: self.np_eig_vec, self.log_eig_val: self.np_log_eig_val}
        self._launch_session()

    def _create_tf_placeholders(self):
        # Testing with less than a full reconstruction doesn't work due to numerical instabilities
        self.eig_vec = tf.placeholder(dtype=self.dtype, shape=(self.batch_size, self.features_size, self.n_eig))
        self.log_eig_val = tf.placeholder(dtype=self.dtype, shape=(self.batch_size, self.n_eig))

    def _create_covariance_instance(self):
        # Create a random set of orthogonal eigen vectors and positive eigen values
        self.cov_object = cov_rep.CovarianceEig(log_eig_val_covar=self.log_eig_val, eig_vec=self.eig_vec,
                                                inversion_method=self.inversion_method)

        outputs = self.create_random_covariance(self.batch_size, self.features_size, self.dtype, True, self.n_eig)
        self.np_covariance, self.np_eig_vec, self.np_log_eig_val, self.np_eig_val_matrix = outputs

        self._create_np_precision_cholesky()

    def test_log_det_covariance(self):
        if self.reduced_eig_vec:
            _, np_log_det_covariance = np.linalg.slogdet(self.np_eig_val_matrix)
            self._asset_allclose_tf_feed(self.cov_object.log_det_covariance(), np_log_det_covariance)
        else:
            super(CovarianceEigTest, self).test_log_det_covariance()

    def test_chol_covariance(self):
        if self.reduced_eig_vec:
            return
        else:
            super(CovarianceEigTest, self).test_chol_covariance()

    def test_chol_precision(self):
        if self.reduced_eig_vec:
            return
        else:
            super(CovarianceEigTest, self).test_chol_precision()


class PrecisionEigTest(CovarianceEigTest):
    def _create_covariance_instance(self):
        super(PrecisionEigTest, self)._create_covariance_instance()
        self.cov_object = cov_rep.PrecisionEig(log_eig_val_precision=self.log_eig_val, eig_vec=self.eig_vec,
                                               inversion_method=self.inversion_method)

        self.np_covariance, self.np_precision = self.np_precision, self.np_covariance
        self.np_eig_val_matrix = np.linalg.inv(self.np_eig_val_matrix)

        self.np_sqrt_precision, self.np_sqrt_covariance = self.np_sqrt_covariance, self.np_sqrt_precision
        self.np_covariance_sqrt_sample_matrix, self.np_precision_sqrt_sample_matrix = self.np_precision_sqrt_sample_matrix, \
                                                                                      self.np_covariance_sqrt_sample_matrix

        if self.reduced_eig_vec is True:
            return

        self.np_chol_covariance, self.np_chol_precision = self.np_chol_precision, self.np_chol_covariance
        self.np_covariance_chol_sample_matrix, self.np_precision_chol_sample_matrix = self.np_precision_chol_sample_matrix, \
                                                                                      self.np_covariance_chol_sample_matrix


class CovarianceEigDiagTest(CovarianceEigTest):
    def setUp(self):
        super(CovarianceEigDiagTest, self).setUp()

        self.tf_feed = {self.eig_vec: self.np_eig_vec, self.log_eig_val: self.np_log_eig_val,
                        self.diag_a: self.np_diag_a}

    def _create_tf_placeholders(self):
        super(CovarianceEigDiagTest, self)._create_tf_placeholders()
        self.diag_a = tf.placeholder(dtype=self.dtype, shape=self.input_shape)

    def _create_covariance_instance(self):
        super(CovarianceEigDiagTest, self)._create_covariance_instance()
        self.cov_object = cov_rep.CovarianceEigDiag(log_eig_val_covar=self.log_eig_val, eig_vec=self.eig_vec,
                                                    diag_a=self.diag_a, inversion_method=self.inversion_method)

        self.np_diag_a = np.exp(np.random.normal(size=self.batch_size).astype(self.dtype.as_numpy_dtype))
        self.np_diag_a = np.tile(np.expand_dims(self.np_diag_a, axis=1), [1, self.features_size])

        np_matrix = np.zeros(self.matrix_shape, dtype=self.dtype.as_numpy_dtype)
        for i in range(self.batch_size):
            np_matrix[i][np.diag_indices_from(np_matrix[i])] = self.np_diag_a[i]

        ori_eig_vec = self.np_eig_vec.copy()
        ori_eig_val = self.np_eig_val_matrix.copy()

        self.np_covariance_no_diag = self.np_covariance.copy()
        self.np_precision_no_diag = np.linalg.inv(self.np_covariance)

        self.np_covariance += np_matrix
        self.np_precision = None
        self.np_chol_covariance = None
        self.np_chol_precision = None
        self.np_sqrt_covariance = None
        self.np_sqrt_precision = None
        self.np_eig_val_matrix = None
        self.np_eig_vec = None

        self._create_np_precision_cholesky()

        self.np_eig_vec = ori_eig_vec
        self.np_eig_val_matrix = ori_eig_val

    def test_covariance_no_diag(self):
        self._asset_allclose_tf_feed(self.cov_object.covariance_no_diag, self.np_covariance_no_diag)

    def test_precision_no_diag(self):
        self._asset_allclose_tf_feed(self.cov_object.precision_no_diag, self.np_precision_no_diag)


class PrecisionEigDiagTest(CovarianceEigDiagTest):
    def _create_covariance_instance(self):
        super(PrecisionEigDiagTest, self)._create_covariance_instance()
        self.cov_object = cov_rep.PrecisionEigDiag(log_eig_val_precision=self.log_eig_val, eig_vec=self.eig_vec,
                                                   diag_a=self.diag_a, inversion_method=self.inversion_method)

        self.np_covariance, self.np_precision = self.np_precision, self.np_covariance
        self.np_covariance_no_diag, self.np_precision_no_diag = self.np_precision_no_diag, self.np_covariance_no_diag
        self.np_chol_covariance, self.np_chol_precision = self.np_chol_precision, self.np_chol_covariance
        self.np_sqrt_covariance, self.np_sqrt_precision = self.np_sqrt_precision, self.np_sqrt_covariance
        self.np_covariance_sqrt_sample_matrix, self.np_precision_sqrt_sample_matrix = self.np_precision_sqrt_sample_matrix, \
                                                                                      self.np_covariance_sqrt_sample_matrix
        self.np_covariance_chol_sample_matrix, self.np_precision_chol_sample_matrix = self.np_precision_chol_sample_matrix, \
                                                                                      self.np_covariance_chol_sample_matrix


def declare_reduced_eig_method_test_classes(class_list, current_module):
    for test_class in class_list:
        # Create the new class with the changed
        class_name = "{}_{}_reduced_eig".format(test_class.__name__, cov_rep.DecompMethod.EIGEN)
        current_module[class_name] = type(class_name, (test_class,), dict(
            inversion_method=cov_rep.DecompMethod.EIGEN, reduced_eig_vec=True))


def _add_all_inv_methods():
    class_list = list()
    class_list.append(CovarianceEigTest)
    class_list.append(PrecisionEigTest)
    declare_inv_method_test_classes(class_list, globals())
    declare_reduced_eig_method_test_classes(class_list, globals())

    class_list = list()
    class_list.append(CovarianceEigDiagTest)
    class_list.append(PrecisionEigDiagTest)
    inv_methods = (cov_rep.DecompMethod.LU, cov_rep.DecompMethod.EIGEN, cov_rep.DecompMethod.CUSTOM)
    declare_inv_method_test_classes(class_list, globals(), inv_methods=inv_methods)


if __name__ == os.path.splitext(os.path.basename(__file__))[0]:
    _add_all_inv_methods()

if __name__ == '__main__':
    _add_all_inv_methods()
    unittest.main()
