import sys
import torch
import unittest

import numpy as np


###############################################################################

sys.path[0] += "/../src/TranscriptDecomPy/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import inla_TD
import distributions_TD as dist_TD


###############################################################################


class TestQMatrix(unittest.TestCase):

    # ------------------------------------------------- #
    #  tests for the function build_gmrf_precision_mat  #
    # ------------------------------------------------- #

    def test_build_gmrf_Q_size_5(self):
        """Tests whether the precision matrix constructed has the shape that it
        should have. Here, we use 5 features.
        """

        theta = inla_TD.gen_tensor(1)
        Q = dist_TD.build_gmrf_Q(5, theta, theta, theta)
        expected_result = (11, 11)

        return self.assertEqual(Q.shape, expected_result)
    

    def test_build_gmrf_Q_size_12(self):
        """Tests whether the precision matrix constructed has the shape that it
        should have. Here, we use 12 features.
        """

        theta = inla_TD.gen_tensor(1)
        Q = dist_TD.build_gmrf_Q(12, theta, theta, theta)
        expected_result = (25, 25)

        return self.assertEqual(Q.shape, expected_result)
    

    def test_build_gmrf_Q_diag_ones(self):
        """Tests whether the precision matrix constructed has the main diagonal
        that it should have, when all the parameters are 1.
        """

        n = 5
        theta_intercept = inla_TD.gen_tensor(1)
        theta_PD = inla_TD.gen_tensor(1)
        theta_PI = inla_TD.gen_tensor(1)

        Q = dist_TD.build_gmrf_Q(n, theta_intercept, theta_PD, theta_PI)
        result = torch.diag(Q)

        Q_00 = (theta_intercept + n * theta_PI)[None]
        Q_11 = (theta_PD + theta_PI)[None]
        Q_22 = 2 * theta_PD + theta_PI
        expected_result = torch.hstack((Q_00, Q_11, 
                                        inla_TD.gen_tensor(Q_22, n - 2), Q_11, 
                                        inla_TD.gen_tensor(theta_PI, n)))
        
        return torch.testing.assert_close(result, expected_result)
    

    def test_build_gmrf_Q_diag_diff(self):
        """Tests whether the precision matrix constructed has the main diagonal
        that it should have. Each parameter is chosen so that we can account if
        the constribution of each parameter to the values of the main diagonal.
        """

        n = 7
        theta_intercept = inla_TD.gen_tensor(1)
        theta_PD = inla_TD.gen_tensor(10)
        theta_PI = inla_TD.gen_tensor(100)

        Q = dist_TD.build_gmrf_Q(n, theta_intercept, theta_PD, theta_PI)
        result = torch.diag(Q)

        Q_00 = (theta_intercept + n * theta_PI)[None]
        Q_11 = (theta_PD + theta_PI)[None]
        Q_22 = 2 * theta_PD + theta_PI
        expected_result = torch.hstack((Q_00, Q_11, 
                                        inla_TD.gen_tensor(Q_22, n - 2), Q_11, 
                                        inla_TD.gen_tensor(theta_PI, n)))
        
        return torch.testing.assert_close(result, expected_result)
    

    def test_build_gmrf_Q_offdiag_below(self):
        """Tests whether the precision matrix constructed has the first off-
        -diagonal that it should have. Here the first off-diagonal refers to
        the diagonal immediately below the main diagonal.
        """
        
        n = 7
        theta_intercept = inla_TD.gen_tensor(1)
        theta_PD = inla_TD.gen_tensor(10)
        theta_PI = inla_TD.gen_tensor(100)

        Q = dist_TD.build_gmrf_Q(n, theta_intercept, theta_PD, theta_PI)
        result = torch.diag(Q, -1)

        expected_result = torch.hstack((theta_PI[None], 
                                        inla_TD.gen_tensor(-theta_PD, n - 1), 
                                        inla_TD.gen_tensor(0, n)))
        
        return torch.testing.assert_close(result, expected_result)
    

    def test_build_gmrf_Q_offdiag_above(self):
        """Tests whether the precision matrix constructed has the first off-
        -diagonal that it should have. Here the first off-diagonal refers to
        the diagonal immediately above the main diagonal.
        """

        n = 7
        theta_intercept = inla_TD.gen_tensor(1)
        theta_PD = inla_TD.gen_tensor(10)
        theta_PI = inla_TD.gen_tensor(100)

        Q = dist_TD.build_gmrf_Q(n, theta_intercept, theta_PD, theta_PI)
        result = torch.diag(Q, 1)

        expected_result = torch.hstack((theta_PI[None], 
                                        inla_TD.gen_tensor(-theta_PD, n - 1), 
                                        inla_TD.gen_tensor(0, n)))
        
        return torch.testing.assert_close(result, expected_result)
    

    def test_build_gmrf_Q_far_offdiag_below(self):
        """Tests whether the precision matrix constructed has the upper 
        off-diagonal that is far from the main diagonal that it should have.
        """

        n = 7
        theta_intercept = inla_TD.gen_tensor(1)
        theta_PD = inla_TD.gen_tensor(10)
        theta_PI = inla_TD.gen_tensor(100)

        Q = dist_TD.build_gmrf_Q(n, theta_intercept, theta_PD, theta_PI)
        result = torch.diag(Q, -n)

        expected_result = torch.hstack((theta_PI[None], 
                                        inla_TD.gen_tensor(-theta_PI, n)))
        
        return torch.testing.assert_close(result, expected_result)


    def test_build_gmrf_Q_far_offdiag_above(self):
        """Tests whether the precision matrix constructed has the lower 
        off-diagonal that is far from the main diagonal that it should have.
        """

        n = 7
        theta_intercept = inla_TD.gen_tensor(1)
        theta_PD = inla_TD.gen_tensor(10)
        theta_PI = inla_TD.gen_tensor(100)

        Q = dist_TD.build_gmrf_Q(n, theta_intercept, theta_PD, theta_PI)
        result = torch.diag(Q, n)

        expected_result = torch.hstack((theta_PI[None], 
                                        inla_TD.gen_tensor(-theta_PI, n)))
        
        return torch.testing.assert_close(result, expected_result)


###############################################################################

unittest.main()