import os
import sys

sys.path[0] += "/../src"

from TranscriptDecomPy import inla_TD as tdp

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###############################################################################

import unittest
import functools

import numpy as np

from scipy import sparse, linalg
from scipy.stats import nbinom, multivariate_normal, norm
from itertools import product


###############################################################################

# class TestLikelihood(unittest.TestCase):

#     # ----------------------------------------------------- #
#     #  tests for the function Negative Binomial Likelihood  #
#     # ----------------------------------------------------- #


#     def test_likelihood_shape_return_2_dim(self):
#         """Test to check if the likelihood returns an array of the appropriate
#         dimensions if the combination of the observations and the means 
#         produces a 2D array.
#         """

#         dataset = np.array([nbinom.rvs(5, 0.7, size=100) for i in range(30)])
#         result = tdp.log_likelihood_neg_binom(dataset, np.full(100, 5), 0.7)

#         self.assertEqual(result.shape, ())


#     def test_likelihood_shape_return_3_dim(self):
#         """Test to check if the likelihood returns an array of the appropriate
#         dimensions if the combination of the observations and the means 
#         produces a 3D array.
#         """

#         dataset = [[nbinom.rvs(5, 0.7, size=100) for _ in range(30)],
#                    [nbinom.rvs(6, 0.3, size=100) for _ in range(30)]]
#         dataset = np.array(dataset)
#         result = tdp.log_likelihood_neg_binom(dataset, 
#                                                 np.full(100, 5, dtype=float), 
#                                                 0.7)

#         self.assertEqual(result.shape, (2,))


#     def test_likelihood_with_different_dist(self):
#         """Tests if the log-likelihood function can successfully give higher 
#         probability to the means that actually generated data.
#             Since this is based on sampling, it might give the wrong result 
#         sometimes. However, increasing the number of samples should reduce the
#         odds of this happening."
#         """

#         theta = 0.3
#         obs = np.array([nbinom.rvs(i * theta / (1-theta) , theta, 200) 
#                 for i in range(1,31)]).T
#         xs = np.repeat(np.arange(1, 15)[:, np.newaxis], 30, 1)[:, np.newaxis, :]
#         xs = np.concatenate((xs, np.arange(1,31)[np.newaxis, np.newaxis]), 0)
        
#         results = tdp.log_likelihood_neg_binom(obs, xs, theta)
#         max_value_idx = np.argmax(results)
#         self.assertEqual(max_value_idx, 14)



class TestDerivativesApprox(unittest.TestCase):

    # ---------------------------------------------- #
    #  tests for the function expand_increment_axis  #
    # ---------------------------------------------- #

    def test_expand_increment_axis(self):

        result = tdp.expand_increment_axis(3, 3).cpu()
        result = np.array(result)

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
        x = torch.arange(n_feat).to(device)
        h = 0.2
        increment = tdp.expand_increment_axis(n_feat, h)
        test_f = lambda z: z

        result = tdp.fst_order_central_differences(test_f, x, increment, h)
        result = np.array(result.cpu())
        expected_result = np.eye(n_feat)[:, np.newaxis, :]

        return np.testing.assert_array_almost_equal(result, expected_result)
    

    def test_fst_order_cdf_5_rep_mult_out(self):
        """Tests the function giving it 5 observations and making the objective 
        function return an array."""

        n_feat = 2
        x = torch.tensor([6,2]).to(device)
        h = 0.1
        increment = tdp.expand_increment_axis(n_feat, h)
        obs = torch.ones((5,2)).to(device)
        test_f = lambda z: obs + z

        result = tdp.fst_order_central_differences(test_f, x, increment, h)
        result = np.array(result.cpu())
        expected_result = np.repeat(np.eye(2)[:, np.newaxis, :], 5, 1)

        return np.testing.assert_almost_equal(result, expected_result)
    

    def test_fst_order_cdf_1_rep_1_out(self):
        """Tests the function giving it only 1 observation and making the 
        objective function return a single value based on an array."""

        n_feat = 4
        x = torch.arange(n_feat).to(device)
        h = 0.2
        increment = tdp.expand_increment_axis(n_feat, h)
        test_f = lambda z: torch.sum(z, 2)

        result = tdp.fst_order_central_differences(test_f, x, increment, h)
        result = np.array(result.cpu())
        expected_result = np.ones(n_feat)[:, np.newaxis]

        return np.testing.assert_array_almost_equal(result, expected_result)


    def test_fst_order_cdf_5_rep_1_out(self):
        """Tests the function giving it 5 observations and making the objective 
        function return a single value based on an array."""
        
        n_feat = 2
        x = torch.tensor([6,2]).to(device)
        h = 0.1
        increment = tdp.expand_increment_axis(n_feat, h)

        obs = np.array([np.arange(5), np.ones(5)]).T
        obs = torch.tensor(obs).to(device)
        test_f = lambda z: torch.sum(obs + z, 2)

        result = tdp.fst_order_central_differences(test_f, x, increment, h)
        result = np.array(result.cpu())
        expected_result = np.ones((n_feat, 5))

        return np.testing.assert_almost_equal(result, expected_result)
    

    # ------------------------------------------------------ #
    #  tests for the function snd_order_central_differences  #
    # ------------------------------------------------------ #

    def test_snd_order_cdf_1_rep_mult_out(self):
        """Tests the function giving it only one observation and making the
        objective function return an array."""

        n_feat = 4
        x = torch.arange(n_feat).to(device)
        h = 1
        increment = tdp.expand_increment_axis(n_feat, h)
        test_f = lambda z: z**2

        result = tdp.snd_order_central_differences(test_f, x, increment, h)
        result = np.array(result.cpu())

        expected_result = 2 * np.eye(n_feat)[:, np.newaxis, :]

        return np.testing.assert_array_almost_equal(result, expected_result)
    

    def test_snd_order_cdf_3_rep_mult_out(self):
        """Tests the function giving it 3 observations and making the objective 
        function return an array."""

        n_feat = 4
        x = torch.arange(4).to(device)
        h = 1
        increment = tdp.expand_increment_axis(n_feat, h)
        obs = torch.arange(12).reshape((3, n_feat)).to(device)
        test_f = lambda z: (obs + z)**2

        result = tdp.snd_order_central_differences(test_f, x, increment, h)
        result = np.array(result.cpu())

        expected_result = 2 * np.eye(n_feat, dtype=float)[:, np.newaxis, :]
        expected_result = np.repeat(expected_result, 3, 1)

        return np.testing.assert_almost_equal(result, expected_result)
    

    def test_snd_order_cdf_1_rep_1_out(self):
        """Tests the function giving it only 1 observation and making the 
        objective function return a single value based on an array."""
        
        n_feat = 4
        x = torch.arange(n_feat).to(device)
        h = 1
        increment = tdp.expand_increment_axis(n_feat, h)
        test_f = lambda z: torch.sum(z**2, len(z.shape)-1)

        result = tdp.snd_order_central_differences(test_f, x, increment, h)
        result = np.array(result.cpu())

        expected_result = np.full((n_feat, 1), 2)

        return np.testing.assert_array_almost_equal(result, expected_result)



###############################################################################

def creating_IGMRF_Q(n_dim: int) -> np.ndarray:
    """Generates the precision matrix of a first-order Random-Walk. This 
    corresponds to an intrinsic GMRF.

    Args:
        n_dim (int): number of features of the Random-Walk. Side of the 
            precision matrix.

    Returns:
        np.ndarray: precision matrix.
    """
    
    diagonal = np.hstack(([1], np.full(n_dim -2, 2), [1]))
    Q = sparse.diags(diagonal)
    Q.setdiag(-1, k=1)
    Q.setdiag(-1, k=-1)

    return Q.todense()


def sampling_IGMRF(Q: np.ndarray, theta_PD: float, 
                    n_samples: int) -> np.ndarray:
    """Samples from the intrinsic GMRF. 
    
        Given that an intrinsic GMRF is characterized by a rank deficient
    matrix (which means the precision matrix has no inverse and a 0 
    determinant), a different method has to be used to sample from it. In this
    case, we use the eigenvectors with non-zero eigenvalues associated.
    

    Args:
        Q (np.ndarray): graph relationships of the IGMRF in the form of a 
            matrix.
        theta_PD (float): constant multiplied to Q to form the precision matrix. 
        n_samples (int): number of samples to be generated of the IGMRF.

    Returns:
        np.ndarray: Samples of the IGMRF.
    """
    
    v, e = linalg.eigh(Q)
    v, e = v[1:], e[:, 1:]
    
    cov = sparse.diags((v*theta_PD)**(-1)).toarray()
    y = multivariate_normal(np.zeros(v.shape[0]), cov).rvs(size=n_samples)    
    y = y[:, np.newaxis] if n_samples == 1 else y.T

    x = e @ y
    return x.T


def sampling_global_gmrf(n_dim: int, theta_intercept: float, theta_PD: float, 
                            theta_PI: float, n_samples: int) -> tuple:
    """Samples from the GMRF obtained in the following model,
    
        eta = intercept + PD + PI
        x = (intercept, PD^T, eta^T)^T

        where PD is a first-order Random Walk, PI is a fixed effects model and
    x is the global GMRF which we are sampling.

    Args:
        n_dim (int): Number of features of the PD and PI GMRFs.
        theta_intercept (float): inverse variance of the intercept.
        theta_PD (float): constant that multiplies with the structure of the
            graph of the first-order Random-Walk, forming the precision matrix
            of the Random-Walk.
        theta_PI (float): inverse variance of each of the features of the fixed-
            -effects model.
        n_samples (int): Number of samples to obtain from the global GMRF.

    Returns:
        (tuple): samples of the global GMRF.
            intercept (np.ndarray): samples from the intercept
            PD (np.ndarray): samples from the first-order Random-Walk IGMRF.
            eta (np.ndarray): samples of the combined GMRF.
    """
    
    intercept = norm(0, 1/theta_intercept).rvs(n_samples)[:, np.newaxis]

    Q_PD = creating_IGMRF_Q(n_dim)
    PD = sampling_IGMRF(Q_PD, theta_PD, n_samples)

    Q_PI = sparse.diags(np.full(n_dim, (1/theta_PI))).todense()
    PI = multivariate_normal(np.zeros(n_dim), Q_PI).rvs(size=n_samples)
    PI = PI[np.newaxis, :] if n_samples == 1 else PI

    eta = intercept + PD + PI
    return (intercept, PD, eta)



def sampling_synthetic_data(gmrf: np.ndarray, theta_y: float, 
                                n_samples: int) -> np.ndarray:
    """Samples the observations given a GMRF sample using a Negative Binomial
    distribution.

    Args:
        gmrf (np.ndarray): sample of the GMRF that generates the observations.
        theta_y (float): dispersion of the data in the Negative Binomial 
            distribution.
        n_samples (int): number of samples of the observations

    Returns:
        (np.ndarray): observations sampled.
    """

    eta = gmrf[2][0]
    eta_length = eta.shape[0]
    
    n = np.exp(eta) * (theta_y/ (1 - theta_y))
    return nbinom.rvs(n=n, p=theta_y, size=(n_samples, eta_length))


class TestConditionalDensities(unittest.TestCase):

    ####################################################
    # Defining some class variables for data synthesis #
    ####################################################

    n_feat = 20
    n_obs = 700

    theta_y = 0.40
    theta_intercept = 2
    theta_PD = 3
    theta_PI = 4

    gmrf = sampling_global_gmrf(n_feat, theta_intercept, theta_PD, theta_PI, 1)
    obs = sampling_synthetic_data(gmrf, theta_y, n_obs)
    intercept, pd, eta = gmrf

    # converting for GPU
    theta_y = torch.tensor([theta_y], dtype=torch.float64).to(device)

    gmrf = torch.tensor(np.concatenate((intercept[0], pd[0], eta[0])),
                        dtype=torch.float64)
    gmrf = gmrf.to(device)

    obs = torch.tensor(obs, dtype=torch.float64).to(device)

    #######################################
    #  end of class variable definitions  #
    #######################################


    # ---------------------------------------------- #
    #  tests for the function newton_raphson_method  #
    # ---------------------------------------------- #


    def test_newton_raphson_method(self):
        """Tests the Newton-Raphson method by testing if it correctly 
        approximates the latent vector that both was generated and is used to
        generate the observations."""
        
        objective = functools.partial(tdp.log_likelihood_neg_binom, 
                                        theta_y=self.theta_y,
                                        obs=self.obs)
        
        Q = tdp.build_gmrf_precision_mat(self.n_feat, self.theta_intercept,
                                            self.theta_PD, self.theta_PI)
        
        init_val = torch.ones(self.n_feat * 2 + 1, dtype=torch.float64)
        init_val = init_val.to(device)
        
        mode_x, _, = tdp.newton_raphson_method(objective=objective, Q=Q, 
                                                h=1e-4, threshold=1e-3,
                                                max_iter=30, init_v=init_val)

        return np.testing.assert_array_almost_equal(mode_x.cpu(), 
                                                    self.gmrf.cpu(), 0)


    # -------------------------------------------------- #
    #  tests for the function approx_marg_post_of_theta  #
    # -------------------------------------------------- #

    def neg_p_theta_given_y(self, curr_theta: np.ndarray) -> float:
        """Creates the function that calculates the -log p(theta | y) for the
        generated data.

        Args:
            curr_theta (int): parameters used in the calculation of the 
                probability.

        Returns:
            float: - log p(theta | y)
        """
        theta_y, theta_intercept, theta_PD, theta_PI = curr_theta
        theta_y = torch.tensor(theta_y, dtype=torch.float64).to(device)

        new_Q = tdp.build_gmrf_precision_mat(self.n_feat, theta_intercept, 
                                                theta_PD, theta_PI)

        objective = functools.partial(tdp.log_likelihood_neg_binom, 
                                            theta_y=theta_y, 
                                            obs=self.obs)
        
        init_v = torch.ones(self.n_feat*2 + 1, dtype=torch.float64).to(device)
            
        mode_x, ga_det = tdp.newton_raphson_method(objective, new_Q, 
                                                1e-4, 1e-3, 30,
                                                init_v)
        
        gmrf_prior = tdp.create_gmrf_density_func(new_Q)
        p_theta_y = tdp.approx_marg_post_of_theta(data_likelihood=objective,
                                                    theta=curr_theta,
                                                    gmrf_likelihood=gmrf_prior,
                                                    theta_dist=lambda _: 1,
                                                    gaus_approx_mean=mode_x,
                                                    gaus_approx_det=ga_det)
        
        return -p_theta_y


    # def test_approx_marg_post_of_theta(self):
    #     """Tests the approximation of the marginal posterior of theta by 
    #     checking that the function can successfully give higher probability to 
    #     the parameter that is actually used in generating the data. 
    #         Since this is based on sampling, it might give the wrong result 
    #     sometimes. However, increasing the number of samples should reduce the
    #     odds of this happening."""

    #     correct_result = (self.theta_y, self.theta_intercept, self.theta_PD,
    #                         self.theta_PI)
        
    #     results = []
    #     range_common=np.arange(2, 5)
    #     y_range = np.arange(0.2, 0.7, 0.1)
        
    #     poss = []
    #     poss_gen = product(y_range, range_common, range_common, range_common)
    #     for curr_theta in poss_gen:
    #         theta_y, *theta = curr_theta
    #         curr_theta = (np.round(theta_y, 1), *theta)
            
    #         poss.append(curr_theta)            
    #         res = self.neg_p_theta_given_y(curr_theta)
    #         print(f"{curr_theta} -> {res[0]}")
    #         results.append(-res[0])
        
    #     final_choice = np.argmax(results)
    #     return self.assertTupleEqual(poss[final_choice], correct_result)
    

    # def test_lbfgs(self):
    #     """Tests the lbfgs function, to see if it can find the right mode of 
    #     the density function."""
        
    #     result = tdp.lbfgs(p_theta_given_y=self.neg_p_theta_given_y, 
    #                         init_guess=0.8,
    #                         bounds=[(0.03, 0.97)],
    #                         n_hist_updates=30)
        
    #     print("Theta Approximated:")
    #     print(result[0])
    #     return self.assertAlmostEqual(result[0], self.theta, 1)


###############################################################################

unittest.main()


###############################################################################
# def read_data(name: str) -> np.ndarray:

#     with open(name, 'r', encoding='utf8') as datafile:
#         data = []
#         for line in datafile.read().splitlines()[1:]:
#             data.append(line.split(",")[1:])
#         return np.array(data, dtype=int)
    

# # data = read_data("cage_data_test.csv")
# # x = np.full(109, 10000)[np.newaxis, :]

# # import time
# # start = time.time()
# # print(np.sum(individual_log_neg_binom_likelihood(data, x, np.array([0.2])), 1))
# # print(f"It took {time.time() - start}")