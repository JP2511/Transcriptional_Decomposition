import sys
sys.path[0] += "/../src"

from TranscriptDecomPy import inla_TD as tdp

###############################################################################

import unittest
import functools

import numpy as np

from scipy import sparse, linalg
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

        self.assertEqual(result.shape, ())


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

def create_prec_and_cov(n_feat: int, diag_val: float, offset: float) -> tuple:
    """Creates a precision matrix (and its inverse) with only nonzero elements
    in the main diagonal and the first offset diagonal on each side of the 
    matrix.

    Args:
        n_feat (int): number of rows or columns of the precision matrix.
        diag_val (float): value in the main diagonal.
        offset (float): value on each of the first offset diagonals.

    Returns:
        Q: precision matrix.
        cov: covariance matrix (inverse of the precision matrix). 
    """

    Q = sparse.diags(np.full(n_feat, diag_val))
    Q.setdiag(offset, k=-1)
    Q.setdiag(offset, k=1)
    
    dense_Q = Q.todense()
    cov = linalg.inv(dense_Q)
    Q = sparse.dia_matrix(dense_Q)

    return (Q, cov)


def create_synthetic_data(theta: float, alpha: np.ndarray, cov: np.ndarray,
                            n_reps: int) -> np.tuple:
    """Creates a synthetic dataset where a specified multivariate normal 
    distribution generates a latent vector, which is then used in a negative
    binomial distribution to generate the observations.

    Args:
        theta (float): probability of success in the negative binomial 
            distribution.
        alpha (np.ndarray): vector of means of the multivariate normal 
            distribution.
        cov (np.ndarray): covariance matrix of the multivariate normal 
            distribution.
        n_reps (int): number of observations to be sampled from the negative
            binomial distribution.

    Returns:
        gmrf (np.ndarray): latent vector sampled by the multivariate normal
            distribution that is used to sample the observations.
        obs (np.ndarray): sampled observations.
    """

    gmrf = multivariate_normal(alpha, cov).rvs()
    param_n_NB = gmrf * theta / (1 - theta)
    obs = nbinom(param_n_NB, theta).rvs(size=(n_reps, alpha.shape[0]))
    return (gmrf, obs)


class TestConditionalDensities(unittest.TestCase):

    # ---------------------------------------------- #
    #  tests for the function newton_raphson_method  #
    # ---------------------------------------------- #


    def test_newton_raphson_method(self):
        """Tests the Newton-Raphson method by testing if it correctly 
        approximates the latent vector that both was generated and is used to
        generate the observations."""
        n_feat = 10
        theta = 0.35
        alpha = np.full(n_feat, 3)
        
        Q, cov = create_prec_and_cov(n_feat, 2, -1)
        
        gmrf, obs = create_synthetic_data(theta, alpha, cov, 200)
        objective = functools.partial(tdp.log_likelihood_neg_binom, theta=theta, 
                                        obs=obs)

        mode_x, _, = tdp.newton_raphson_method(objective=objective, 
                                                Q=Q, mu=alpha, h=1e-4, 
                                                threshold=1e-6, max_iter=100, 
                                                init_v=np.ones(n_feat))
        return np.testing.assert_array_almost_equal(mode_x, gmrf, 0)



###############################################################################

unittest.main()