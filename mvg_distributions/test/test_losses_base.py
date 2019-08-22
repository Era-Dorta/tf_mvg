import numpy as np
import tensorflow as tf

import mvg_distributions.covariance_representations as cov_rep
from mvg_distributions.test.tf_test import TFTest


class LossesTestBase(TFTest):
    def setUp(self):
        super().setUp()

        self.batch_size = 10
        self.features_size = 25
        self._launch_session()

    def _random_normal_params(self, cov_type):
        x = np.random.normal(size=(self.batch_size, self.features_size)).astype(np.float32)
        mu = np.random.normal(size=(self.batch_size, self.features_size)).astype(np.float32)

        if cov_type is cov_rep.CovarianceDiag:
            diag_covariance = np.random.normal(size=(self.batch_size, self.features_size)).astype(np.float32)
            diag_covariance = np.abs(diag_covariance)

            return x, mu, diag_covariance
        if cov_type is cov_rep.PrecisionConvCholFilters:
            from mvg_distributions.covariance_representations.tests.test_covariance_conv import \
                PrecisionConvCholFiltersTest

            test_obj = PrecisionConvCholFiltersTest()
            test_obj.setUp()

            return x, mu, test_obj.np_covariance, test_obj.np_weights, test_obj.np_basis, \
                   test_obj.np_log_diag_chol_precision

        if cov_type is cov_rep.PrecisionDilatedConvCholFilters:
            from mvg_distributions.covariance_representations.tests.test_covariance_conv import \
                PrecisionDilatedConvCholFiltersTest

            test_obj = PrecisionDilatedConvCholFiltersTest()
            test_obj.setUp()

            assert len(test_obj.cov_object.dilation_rates) == 2, "Assumes only two dilation rates"
            assert test_obj.num_basis == 4, "Assumes num basis is 4"

            np_weights = [test_obj.np_weights[..., :2], test_obj.np_weights[..., 2:]]
            np_basis = [test_obj.np_basis[:2, ...], test_obj.np_basis[2:, ...]]

            return x, mu, test_obj.np_covariance, np_weights, np_basis, test_obj.cov_object.dilation_rates, \
                   test_obj.np_log_diag_chol_precision

        if cov_type is tf.distributions.Bernoulli:
            x = np.random.uniform(low=0, high=1, size=(self.batch_size, self.features_size)).astype(np.float32)
            mu = np.random.uniform(low=0, high=1, size=(self.batch_size, self.features_size)).astype(np.float32)
            x = np.round(x)
            return x, mu

        from mvg_distributions.covariance_representations.tests.test_covariance_matrix import \
            CovarianceTest

        covariance = CovarianceTest.create_random_covariance(batch_size=self.batch_size,
                                                             features_size=self.features_size,
                                                             dtype=self.dtype, return_eig=False)

        return x, mu, covariance
