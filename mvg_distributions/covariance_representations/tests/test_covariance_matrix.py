import os
import unittest

import numpy as np
import tensorflow as tf
from scipy.stats import ortho_group

import mvg_distributions.covariance_representations as cov_rep
from mvg_distributions.test.tf_test import TFTest


class CovarianceTest(TFTest):
    inversion_method = cov_rep.DecompMethod.CHOLESKY

    def setUp(self):
        super(CovarianceTest, self).setUp()

        self.batch_size = 10
        self.features_size = 25
        self.input_shape = (self.batch_size, self.features_size)
        self.matrix_shape = (self.batch_size, self.features_size, self.features_size)

        self.tf_feed = None

        self.cov_object = None

        self.np_covariance = None
        self.np_precision = None
        self.np_eig_vec = None
        self.np_eig_val_matrix = None
        self.np_sqrt_covariance = None
        self.np_sqrt_precision = None
        self.log_det_decomp = [cov_rep.DecompMethod.CHOLESKY, cov_rep.DecompMethod.EIGEN, cov_rep.DecompMethod.LU]
        self.sampling_methods = [cov_rep.SampleMethod.SQRT, cov_rep.SampleMethod.CHOLESKY]
        self.inverse_chol_covariance = True

    def tearDown(self):
        if self.__class__ == CovarianceTest:
            return
        super(CovarianceTest, self).tearDown()

    def _launch_session(self):
        if self.__class__ == CovarianceTest:
            return
        super(CovarianceTest, self)._launch_session()

    def _create_np_precision_cholesky(self):
        if self.np_covariance is None and self.np_precision is None:
            self.fail("Should define precision or covariance")

        if self.np_precision is None:
            self.np_precision = np.linalg.inv(self.np_covariance)
        if self.np_covariance is None:
            self.np_covariance = np.linalg.inv(self.np_precision)

        if self.np_eig_vec is None or self.np_eig_val_matrix is None:
            eig_val, self.np_eig_vec = np.linalg.eigh(self.np_covariance)

            self.np_eig_val_matrix = np.zeros(self.matrix_shape, dtype=self.dtype.as_numpy_dtype)
            for i in range(self.batch_size):
                self.np_eig_val_matrix[i][np.diag_indices_from(self.np_eig_val_matrix[i])] = eig_val[i]

        # Square root are symmetric L * L
        if self.np_sqrt_covariance is None:
            self.np_sqrt_covariance = np.matmul(self.np_eig_vec, np.sqrt(self.np_eig_val_matrix))
            self.np_sqrt_covariance = np.matmul(self.np_sqrt_covariance, self.np_eig_vec.transpose([0, 2, 1]))

        if self.np_sqrt_precision is None:
            np_eig_val_sq_inv = self.np_eig_val_matrix.copy()
            np_eig_val_sq_inv[np_eig_val_sq_inv != 0] = 1.0 / np.sqrt(np_eig_val_sq_inv[np_eig_val_sq_inv != 0])
            self.np_sqrt_precision = np.matmul(self.np_eig_vec, np_eig_val_sq_inv)
            self.np_sqrt_precision = np.matmul(self.np_sqrt_precision, self.np_eig_vec.transpose([0, 2, 1]))

        self.np_covariance_sqrt_sample_matrix = self.np_sqrt_covariance
        self.np_precision_sqrt_sample_matrix = self.np_sqrt_precision

        try:
            self.np_chol_covariance = np.linalg.cholesky(self.np_covariance)
            if self.inverse_chol_covariance:
                self.np_covariance_chol_sample_matrix = self.np_chol_covariance
                self.np_precision_chol_sample_matrix = np.linalg.inv(self.np_chol_covariance).transpose([0, 2, 1])
        except np.linalg.LinAlgError:
            pass

        try:
            self.np_chol_precision = np.linalg.cholesky(self.np_precision)
            if not self.inverse_chol_covariance:
                self.np_covariance_chol_sample_matrix = np.linalg.inv(self.np_chol_precision).transpose([0, 2, 1])
                self.np_precision_chol_sample_matrix = self.np_chol_precision
        except np.linalg.LinAlgError:
            pass

        self.np_covariance_net_sample_matrix = None
        self.np_precision_net_sample_matrix = None

    @staticmethod
    def create_random_covariance(batch_size, features_size, dtype, return_eig, n_eig=None):
        if n_eig is None:
            n_eig = features_size
        matrix_shape = [batch_size, features_size, n_eig]

        # Build a random covariance matrix from an eigen decomposition
        eig_vec = ortho_group.rvs(features_size, batch_size).astype(dtype.as_numpy_dtype)
        eig_vec = eig_vec.reshape((batch_size, features_size, features_size))[:, :, :n_eig]

        log_eig_val = np.random.rand(batch_size, n_eig).astype(dtype.as_numpy_dtype)
        eig_val_matrix = np.zeros((batch_size, n_eig, n_eig), dtype=dtype.as_numpy_dtype)
        for i in range(batch_size):
            eig_val_matrix[i][np.diag_indices_from(eig_val_matrix[i])] = np.exp(log_eig_val[i])
        covariance = np.matmul(eig_vec, eig_val_matrix)
        covariance = np.matmul(covariance, eig_vec.transpose([0, 2, 1]))

        if return_eig:
            return covariance, eig_vec, log_eig_val, eig_val_matrix
        else:
            return covariance

    def test_covariance(self):
        if self.__class__ == CovarianceTest:
            return  # Don'r run the test for the CovarianceTest class
        self._asset_allclose_tf_feed(self.cov_object.covariance, self.np_covariance)

    def test_precision(self):
        if self.__class__ == CovarianceTest:
            return
        self._asset_allclose_tf_feed(self.cov_object.precision, self.np_precision)

    def test_chol_covariance(self):
        if self.__class__ == CovarianceTest:
            return
        self._asset_allclose_tf_feed(self.cov_object.chol_covariance, self.np_chol_covariance)

    def test_chol_precision(self):
        if self.__class__ == CovarianceTest:
            return
        self._asset_allclose_tf_feed(self.cov_object.chol_precision, self.np_chol_precision)

    def test_sqrt_covariance(self):
        if self.__class__ == CovarianceTest:
            return
        self._asset_allclose_tf_feed(self.cov_object.sqrt_covariance, self.np_sqrt_covariance)

    def test_sqrt_precision(self):
        if self.__class__ == CovarianceTest:
            return
        self._asset_allclose_tf_feed(self.cov_object.sqrt_precision, self.np_sqrt_precision)

    def _test_diag(self, tf_diag, np_matrix, test_log=False):
        np_diag = np_matrix.diagonal(axis1=1, axis2=2)
        if test_log:
            np_diag = np.log(np_diag)
        self._asset_allclose_tf_feed(tf_diag, np_diag)

    def test_diag_covariance(self):
        if self.__class__ == CovarianceTest:
            return
        self._test_diag(self.cov_object.covariance_diag_part, self.np_covariance)

    def test_diag_precision(self):
        if self.__class__ == CovarianceTest:
            return
        self._test_diag(self.cov_object.precision_diag_part, self.np_precision)

    def test_log_det_covariance(self):
        if self.__class__ == CovarianceTest:
            return

        _, np_log_det_covariance = np.linalg.slogdet(self.np_covariance)
        for decomp in self.log_det_decomp:
            self.cov_object._log_det_covariance = None
            self._asset_allclose_tf_feed(self.cov_object.log_det_covariance(decomp), np_log_det_covariance)

    def _x_precision_x(self, do_mean):
        if self.__class__ == CovarianceTest:
            return

        for num_samples in range(1, 4, 2):
            # Test x_precision_x, with and without mean batch
            np_x = np.random.normal(size=(self.batch_size, num_samples, self.features_size)).astype(
                self.dtype.as_numpy_dtype)
            np_x_p_x = np.zeros(shape=(self.batch_size, num_samples), dtype=self.dtype.as_numpy_dtype)
            for i in range(num_samples):
                n_p_x_r = np.expand_dims(np_x[:, i, :], axis=1)
                np_x_p_x[:, i] = np.matmul(np.matmul(n_p_x_r, self.np_precision),
                                           n_p_x_r.transpose([0, 2, 1])).squeeze()
            np_x_p_x = np_x_p_x.squeeze()
            tf_x_p_x = self.cov_object.x_precision_x(x=tf.constant(np_x), mean_batch=do_mean)
            if do_mean:
                self._asset_allclose_tf_feed(tf.reduce_mean(tf_x_p_x), np_x_p_x.mean())
            else:
                self._asset_allclose_tf_feed(tf_x_p_x, np_x_p_x)

            if num_samples == 1:
                # For one sample also test with input of shape [batch dim, num features]
                tf_x_p_x = self.cov_object.x_precision_x(x=tf.constant(np_x.squeeze(axis=1)), mean_batch=do_mean)
                if do_mean:
                    self._asset_allclose_tf_feed(tf.reduce_mean(tf_x_p_x), np_x_p_x.mean())
                else:
                    self._asset_allclose_tf_feed(tf_x_p_x, np_x_p_x)

    def test_x_precision_x_batch(self):
        if self.__class__ == CovarianceTest:
            return
        self._x_precision_x(do_mean=False)

    def test_x_precision_x_mean(self):
        if self.__class__ == CovarianceTest:
            return
        self._x_precision_x(do_mean=True)

    def _test_sample(self, sample_method, test_covariance):
        for num_samples in range(1, 4, 2):
            # Test with 1 sample and with 3 samples
            np_x = np.random.normal(size=(self.batch_size, self.features_size, num_samples))
            np_x = np_x.astype(self.dtype.as_numpy_dtype)
            tf_np_x = tf.constant(np_x.transpose([0, 2, 1]))

            # Test sample Precision
            if test_covariance:
                # Test sample Sigma
                if sample_method == cov_rep.SampleMethod.SQRT:
                    np_x_c = np.matmul(self.np_covariance_sqrt_sample_matrix, np_x)
                elif sample_method == cov_rep.SampleMethod.CHOLESKY:
                    np_x_c = np.matmul(self.np_covariance_chol_sample_matrix, np_x)
                elif sample_method == cov_rep.SampleMethod.NET:
                    np_x_c = np.matmul(self.np_covariance_net_sample_matrix, np_x)
                else:
                    raise ValueError("Unkown sampling method type {}".format(sample_method))
                tf_x_c = self.cov_object.sample_covariance(epsilon=tf_np_x, sample_method=sample_method,
                                                           flatten_output=True)
            else:
                if sample_method == cov_rep.SampleMethod.SQRT:
                    matrix = self.np_precision_sqrt_sample_matrix
                elif sample_method == cov_rep.SampleMethod.CHOLESKY:
                    matrix = self.np_precision_chol_sample_matrix
                elif sample_method == cov_rep.SampleMethod.NET:
                    matrix = self.np_precision_net_sample_matrix
                else:
                    raise ValueError("Unkown sampling method type {}".format(sample_method))

                np_x_c = np.matmul(np_x.transpose([0, 2, 1]), matrix).transpose([0, 2, 1])

                tf_x_c = self.cov_object.whiten_x(x=tf_np_x, sample_method=sample_method,
                                                  flatten_output=True)

            np_x_c = np_x_c.transpose([0, 2, 1]).squeeze()
            tf_x_c = tf.squeeze(tf_x_c)
            self._asset_allclose_tf_feed(tf_x_c, np_x_c)

    def test_sampling_covariance_sqrt(self):
        if self.__class__ == CovarianceTest:
            return
        if cov_rep.SampleMethod.SQRT in self.sampling_methods:
            self._test_sample(sample_method=cov_rep.SampleMethod.SQRT, test_covariance=True)
        else:
            return

    def test_sampling_covariance_cholesky(self):
        if self.__class__ == CovarianceTest:
            return
        if cov_rep.SampleMethod.CHOLESKY in self.sampling_methods:
            self._test_sample(sample_method=cov_rep.SampleMethod.CHOLESKY, test_covariance=True)
        else:
            return

    def test_sampling_covariance_net(self):
        if self.__class__ == CovarianceTest:
            return
        if cov_rep.SampleMethod.NET in self.sampling_methods:
            self._test_sample(sample_method=cov_rep.SampleMethod.NET, test_covariance=True)
        else:
            return

    def test_whiten_x_sqrt(self):
        if self.__class__ == CovarianceTest:
            return
        if cov_rep.SampleMethod.SQRT in self.sampling_methods:
            self._test_sample(sample_method=cov_rep.SampleMethod.SQRT, test_covariance=False)
        else:
            return

    def test_whiten_x_cholesky(self):
        if self.__class__ == CovarianceTest:
            return
        if cov_rep.SampleMethod.CHOLESKY in self.sampling_methods:
            self._test_sample(sample_method=cov_rep.SampleMethod.CHOLESKY, test_covariance=False)
        else:
            return

    def test_whiten_x_net(self):
        if self.__class__ == CovarianceTest:
            return
        if cov_rep.SampleMethod.NET in self.sampling_methods:
            self._test_sample(sample_method=cov_rep.SampleMethod.NET, test_covariance=False)
        else:
            return

    def _get_eye(self):
        eye = np.expand_dims(np.eye(self.matrix_shape[1]), axis=0)
        return np.tile(eye, [self.batch_size, 1, 1])

    def test_sqrt_identity_matrices(self):
        if self.__class__ == CovarianceTest:
            return
        eye = self._get_eye()

        # sqrt(Covariance) * sqrt(Precision) = Identity
        tf_cov_prec = tf.matmul(self.cov_object.sqrt_covariance, self.cov_object.sqrt_precision)
        self._asset_allclose_tf_feed(tf_cov_prec, eye)


class CovarianceFullTest(CovarianceTest):
    def setUp(self):
        super(CovarianceFullTest, self).setUp()
        self._create_tf_placeholders()
        self._create_covariance_instance()
        self.tf_feed = {self.tf_input: self.np_input}
        self._launch_session()

    def _create_tf_placeholders(self):
        self.tf_input = tf.placeholder(dtype=self.dtype, shape=self.matrix_shape)

    def _create_covariance_instance(self):
        self.np_covariance = self.create_random_covariance(self.batch_size, self.features_size, self.dtype, False)
        self.cov_object = cov_rep.CovarianceFull(covariance=self.tf_input, inversion_method=self.inversion_method)
        self._create_np_precision_cholesky()
        self.np_input = self.np_covariance

    def test_sampling_covariance_sqrt(self):
        self.rtol, self.atol = 5e-5, 5e-5
        super(CovarianceFullTest, self).test_sampling_covariance_sqrt()


class PrecisionFullTest(CovarianceFullTest):
    def _create_covariance_instance(self):
        self.inverse_chol_covariance = False
        super(PrecisionFullTest, self)._create_covariance_instance()
        self.cov_object = cov_rep.PrecisionFull(precision=self.tf_input, inversion_method=self.inversion_method)
        self._create_np_precision_cholesky()
        self.np_input = self.np_precision


# Declare new classes dynamically to also test with additional inversion methods
def declare_inv_method_test_classes(class_list, current_module,
                                    inv_methods=(cov_rep.DecompMethod.LU, cov_rep.DecompMethod.EIGEN)):
    for test_class in class_list:
        for inv_method in inv_methods:
            # Create the new class with the changed inverse method and add it to the current module
            class_name = "{}_{}".format(test_class.__name__, inv_method)
            current_module[class_name] = type(class_name, (test_class,), dict(inversion_method=inv_method))


def _add_all_inv_methods():
    class_list = list()
    class_list.append(CovarianceFullTest)
    class_list.append(PrecisionFullTest)

    declare_inv_method_test_classes(class_list, globals())


if __name__ == os.path.splitext(os.path.basename(__file__))[0]:
    _add_all_inv_methods()

if __name__ == '__main__':
    _add_all_inv_methods()
    unittest.main()
