import sys
sys.path[0] += "/../src"

from TranscriptDecomPy import inla_TD as tdp

###############################################################################

import unittest
import functools

import numpy as np

from scipy.stats import nbinom, multivariate_normal


# ###############################################################################

class TestLikelihood(unittest.TestCase):

    # ----------------------------------------------------- #
    #  tests for the function Negative Binomial Likelihood  #
    # ----------------------------------------------------- #


    def test_likelihood_shape_return_2_dim(self):
        """Test to check if the likelihood returns an array of the appropriate
        dimensions if the combination of the observations and the means 
        produces a 2D array.
        """

        dataset = np.array([nbinom.rvs(5, 0.7, size=100) for i in range(30)])
        result = tdp.log_likelihood_neg_binom(dataset, np.full(100, 5), 0.7)

        self.assertEqual(result.shape, (30,))


    def test_likelihood_shape_return_3_dim(self):
        """Test to check if the likelihood returns an array of the appropriate
        dimensions if the combination of the observations and the means 
        produces a 3D array.
        """

        dataset = [[nbinom.rvs(5, 0.7, size=100) for _ in range(30)],
                   [nbinom.rvs(6, 0.3, size=100) for _ in range(30)]]
        dataset = np.array(dataset)
        result = tdp.log_likelihood_neg_binom(dataset, 
                                                np.full(100, 5, dtype=float), 
                                                0.7)

        self.assertEqual(result.shape, (2,))


    def test_likelihood_with_different_dist(self):
        """Tests if the log-likelihood function can successfully give higher 
        probability to the means that actually generated data. Since this is
        based on sampling, it might give the wrong result sum of the times.
        """

        theta = 0.3
        obs = np.array([nbinom.rvs(i * theta / (1-theta) , theta, 200) 
                for i in range(1,31)]).T
        xs = np.repeat(np.arange(1, 15)[:, np.newaxis], 30, 1)[:, np.newaxis, :]
        xs = np.concatenate((xs, np.arange(1,31)[np.newaxis, np.newaxis]), 0)
        
        results = tdp.log_likelihood_neg_binom(obs, xs, theta)
        max_value_idx = np.argmax(results)
        self.assertEqual(max_value_idx, 14)



class TestDerivativesApprox(unittest.TestCase):

    # ---------------------------------------------- #
    #  tests for the function expand_increment_axis  #
    # ---------------------------------------------- #

    def test_expand_increment_axis(self):

        result = tdp.expand_increment_axis(3, 3)
        expected_result = [[[3, 0, 0]],
                            [[0, 3, 0]],
                            [[0, 0, 3]]]
        
        return np.testing.assert_array_equal(result, expected_result)
    

    # ------------------------------------------------------ #
    #  tests for the function fst_order_central_differences  #
    # ------------------------------------------------------ #
    

    def test_fst_order_cdf_1_rep_mult_out(self):
        """Tests the function giving it only one observation and making the
        objective function return an array."""

        n_feat = 4
        x = np.arange(n_feat)
        h = 0.2
        increment = tdp.expand_increment_axis(n_feat, h)
        test_f = lambda z: z

        result = tdp.fst_order_central_differences(test_f, x, increment, h)
        expected_result = np.eye(n_feat)[:, np.newaxis, :]

        return np.testing.assert_array_almost_equal(result, expected_result)
    

    def test_fst_order_cdf_5_rep_mult_out(self):
        """Tests the function giving it 5 observations and making the objective 
        function return an array."""

        n_feat = 2
        x = np.array([6,2])
        h = 0.1
        increment = tdp.expand_increment_axis(n_feat, h)
        obs = np.ones((5,2))
        test_f = lambda z: obs + z

        result = tdp.fst_order_central_differences(test_f, x, increment, h)
        expected_result = np.repeat(np.eye(2)[:, np.newaxis, :], 5, 1)

        return np.testing.assert_almost_equal(result, expected_result)
    

    def test_fst_order_cdf_1_rep_1_out(self):
        """Tests the function giving it only 1 observation and making the 
        objective function return a single value based on an array."""

        n_feat = 4
        x = np.arange(n_feat)
        h = 0.2
        increment = tdp.expand_increment_axis(n_feat, h)
        test_f = lambda z: np.sum(z, 2)

        result = tdp.fst_order_central_differences(test_f, x, increment, h)
        expected_result = np.ones(n_feat)[:, np.newaxis]

        return np.testing.assert_array_almost_equal(result, expected_result)


    def test_fst_order_cdf_5_rep_1_out(self):
        """Tests the function giving it 5 observations and making the objective 
        function return a single value based on an array."""
        
        n_feat = 2
        x = np.array([6,2])
        h = 0.1
        increment = tdp.expand_increment_axis(n_feat, h)
        obs = np.array([np.arange(5), np.ones(5)]).T
        test_f = lambda z: np.sum(obs + z, 2)

        result = tdp.fst_order_central_differences(test_f, x, increment, h)
        expected_result = np.ones((n_feat, 5))

        return np.testing.assert_almost_equal(result, expected_result)
    

    # ------------------------------------------------------ #
    #  tests for the function snd_order_central_differences  #
    # ------------------------------------------------------ #

    def test_snd_order_cdf_1_rep_mult_out(self):
        """Tests the function giving it only one observation and making the
        objective function return an array."""

        n_feat = 4
        x = np.arange(n_feat)
        h = 1
        increment = tdp.expand_increment_axis(n_feat, h)
        test_f = lambda z: z**2

        result = tdp.snd_order_central_differences(test_f, x, increment, h)
        expected_result = 2 * np.eye(n_feat)[:, np.newaxis, :]

        return np.testing.assert_array_almost_equal(result, expected_result)
    

    def test_snd_order_cdf_3_rep_mult_out(self):
        """Tests the function giving it 3 observations and making the objective 
        function return an array."""

        n_feat = 4
        x = np.arange(4)
        h = 1
        increment = tdp.expand_increment_axis(n_feat, h)
        obs = np.arange(12).reshape((3, n_feat))
        test_f = lambda z: (obs + z)**2

        result = tdp.snd_order_central_differences(test_f, x, increment, h)
        expected_result = 2 * np.eye(n_feat, dtype=float)[:, np.newaxis, :]
        expected_result = np.repeat(expected_result, 3, 1)

        return np.testing.assert_almost_equal(result, expected_result)
    

    def test_snd_order_cdf_1_rep_1_out(self):
        """Tests the function giving it only 1 observation and making the 
        objective function return a single value based on an array."""
        
        n_feat = 4
        x = np.arange(n_feat)
        h = 1
        increment = tdp.expand_increment_axis(n_feat, h)
        test_f = lambda z: np.sum(z**2, len(z.shape)-1)

        result = tdp.snd_order_central_differences(test_f, x, increment, h)
        expected_result = np.full((n_feat, 1), 2)

        return np.testing.assert_array_almost_equal(result, expected_result)



###############################################################################

unittest.main()