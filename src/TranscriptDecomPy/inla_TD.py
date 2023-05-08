import os
os.environ["CHOLMOD_USE_GPU"] = "1"

from sksparse.cholmod import cholesky

###############################################################################


import numpy as np

from functools import partial

from scipy import linalg, sparse
from scipy.stats import nbinom
from scipy.optimize import minimize

import torch
from torch.distributions import NegativeBinomial

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from typing import Callable


###############################################################################


def log_likelihood_neg_binom(x: torch.tensor, obs: torch.tensor, 
                                theta_y: torch.tensor) -> torch.tensor:
    """Calculates the log likelihood of the negative binomial distribution. It 
    is assumed that a point in this distribution is multivariate. As such, the
    log-likelihood of a such a point corresponds to the sum of the 
    log-likelihood its features. In case, there are multiple observations, the
    log-likelihood is calculated for all the observations, meaning that the
    value obtained corresponds to the sum of the log-likelihood of each point. 

    Args:
        x (torch.tensor): Gaussian Markov Random Field sample. If it contains n
            elements, only the last (n-1)//2 elements will be used.
        obs (torch.tensor): observations of which to calculate the probability 
            mass (pmf).
        theta_y (torch.tensor): success probability in experiment in the 
            negative binomial distribution.

    Returns:
        pm_for_single_obs (torch.tensor): log likelihood of the point given the
            specified negative binomial distribution.

    Requires:
        obs_i > 0
        x_i > 0
        0 <= theta_y <= 1
    """
    
    n_feat = (x.shape[-1] -1) // 2
    x = x[:, :, -n_feat:] if len(x.shape) > 1 else x[-n_feat:]
    
    n = torch.exp(x) * (theta_y/ (1 - theta_y))

    p_NB = NegativeBinomial(n, 1-theta_y).log_prob(obs)
    pm_for_single_obs = torch.sum(p_NB, -1)
    
    if len(pm_for_single_obs.shape) > 0:
        return torch.sum(pm_for_single_obs, -1)
    
    return pm_for_single_obs


def build_gmrf_precision_mat(n_dim: int, theta_intercept: float, 
                                theta_PD: float,
                                theta_PI: float) -> sparse.csc_matrix:
    """Constructs a precision matrix (2 * n_dim + 1) x (2 * n_dim + 1) which 
    models a gmrf x where:
        intercept ~ N(0, 1/theta_intercept)
        PD        ~ N(0, (theta_PD * R)^-1 )
        PI        ~ N(0, (theta_PI * I)^-1 )
        
        eta = intercept + PD + PI
        x   = (intercept, PD^T, eta^T)^T

        Here, I is the identity matrix of dimentions n_dim x n_dim and R is a
    first-order Random Walk graph in the form of a matrix n_dim x n_dim.

    Args:
        n_dim (int): number of dimensions of the observed variable to model.
        theta_intercept (float): parameter that controls the variance of the
            intercept of the model.
        theta_PD (float): parameter that controls the variance/influence of the
            Random Walk component of the model.
        theta_PI (float): parameter that controls the variance/influence of the 
            fixed effects component of the model.

    Returns:
        sparse.csc_matrix: precision matrix.
    """

    Q_11 = [ theta_intercept + n_dim * theta_PI ]
    k_v_1T = np.full(n_dim, theta_PI)
    k_v_1 = k_v_1T[:, np.newaxis]
    k_v_I = sparse.diags(np.full(n_dim, theta_PI), format='csr')

    R = sparse.diags(np.full(n_dim, theta_PD*2))
    R.setdiag(-theta_PD, k=-1)
    R.setdiag(-theta_PD, k=1)
    R = sparse.csr_matrix(R)

    fst_block_row = sparse.hstack((Q_11,  k_v_1T    , -k_v_1T), format='csc')
    snd_block_row = sparse.hstack((k_v_1 , R + k_v_I, -k_v_I ), format='csc')
    trd_block_row = sparse.hstack((-k_v_1, -k_v_I   ,  k_v_I ), format='csc')

    return sparse.vstack((fst_block_row, snd_block_row, trd_block_row), 
                            format='csc')



def gmrf_density(x: np.ndarray, det: float, Q: sparse.csc_matrix) -> float:
    """Calculates the density of a value using the probability density function
    of a multivariate normal distribution.

    Args:
        x (np.ndarray): point used to calculate the density.
        det (float): determinant of the precision matrix of the multivariate
            normal distribution used.
        Q (sparse.csc_matrix): precision matrix of the multivariate normal 
            distribution used.

    Returns:
        density (float): density of the point given in the distribution given.
    """
    
    n = x.shape[0]
    exp_value = -0.5 * (x @ Q @ x[:, np.newaxis])
    density = ((2 * np.pi)**(-n/2)) * np.sqrt(det) * np.exp(exp_value)
    return density


def create_gmrf_density_func(Q: sparse.csc_matrix) -> Callable:
    """Generates a function that calculates the density of the GMRF assuming the
    the precision matrix given.

    Args:
        Q (sparse.csc_matrix): precision matrix of the GMRF.

    Returns:
        Callable: function that calculates the density of the GMRF.
    """
    
    det = np.exp(cholesky(Q).logdet())

    return partial(gmrf_density, det=det, Q=Q)


###############################################################################

def expand_increment_axis(n_feat: int, h: float) -> torch.tensor:
    """Creates an expanded matrix that contains all the vectors necessary to
    individually increment each feature while not changing the other features.

    Args:
        n_feat (int): number of features that need to be incremented.
        h (float): increment value.

    Returns:
        np.ndarray: multi-dimensional array that contains the individual 
            increments at each feature without altering the other features.

    Ensures:
        The function returns a multi-dimensional array of the shape 
            (n_feat, 1, n_feat). In the context of the problem, the first axis
            corresponds to the increment at a particular feature; the second
            axis corresponds to the observations and the third axis corresponds
            to the actual value of the features after the addition of the 
            increment.
    """

    return (h * torch.eye(n_feat, dtype=torch.float64)[:, None, :]).to(device)


def fst_order_central_differences(objective: Callable, x: np.ndarray,
                                    increment: np.ndarray, 
                                    h: float) -> np.ndarray:
    """Approximates the first derivative of a function using finite differences,
    in particular using first order central differences.

    Args:
        objective (Callable): function whose derivative we want to approximate.
        x (np.ndarray): value around which we want the approximation to the 
            derivative.
        increment (np.ndarray): 3-dimensional array containing the vectors that
            increment each feature of x individually based on h.
        h (float): step of the approximation. Ideally, it should be as close to
            zero as possible.

    Returns:
        np.ndarray: approximate derivative value at the x point. The shape of 
            the returned array depends on the objective function.

    Requires:
        increment should be a 3-dimensional array, where the first dimension
            corresponds to each of the increments to the individuals features;
            the second dimension should correspond to the observations, so it
            should have shape one in this array; and the third dimension 
            corresponds to the features of x.
        the step actually used in the increment array should be h.
    """
    
    return (objective(x + increment) - objective(x - increment)) / (2*h)


def snd_order_central_differences(objective: Callable, x: torch.tensor,
                                    increment: torch.tensor, 
                                    h: float) -> float:
    """Approximates the second derivative of a function using finite 
    differences, in particular using second order central differences. This 
    function considers only derivatives where the objective function is twice
    partially derived by the same variable.

    Args:
        objective (Callable): function whose derivative we want to approximate.
        x (torch.tensor): value around which we want the approximation to the 
            derivative.
        increment (torch.tensor): 3-dimensional array containing the vectors 
            that increment each feature of x individually based on h.
        h (float): step of the approximation. Ideally, it should be as close to
            zero as possible.

    Returns:
        np.ndarray: approximate derivative value at the x point. The shape of 
            the returned array depends on the objective function.

    Requires:
        increment should be a 3-dimensional array, where the first dimension
            corresponds to each of the increments to the individuals features;
            the second dimension should correspond to the observations, so it
            should have shape one in this array; and the third dimension 
            corresponds to the features of x.
        the step actually used in the increment array should be h.
    """

    diffs = objective(x + increment) - 2*objective(x) + objective(x - increment)
    return diffs / (h**2)



###############################################################################
# Newton-Raphson method

def approx_taylor_expansion(objective: Callable, x: torch.tensor, 
                                h: torch.tensor) -> tuple:
    """Approximates the second and third terms of a quadratic Taylor expansion.

    Args:
        objective (Callable): function we are expanding.
        x (torch.tensor): value around which we are performing the Taylor 
            expansion.
        h (torch.tensor): step of the approximation. Ideally, it should be as 
            close to zero as possible.

    Returns:
        b (torch.tensor): approximate second term of the quadratic Taylor
            expansion.
        c (torch.tensor): approximate third term of the quadratic Taylor
            expansion.
    """

    increment = expand_increment_axis(x.shape[0], h)
    c = - snd_order_central_differences(objective, x, increment, h)

    b = fst_order_central_differences(objective, x, increment, h)
    b += x * c
    return (b, c)


def newton_raphson_method(objective: Callable, 
                            Q: sparse.csc_matrix, 
                            h: float, 
                            threshold: float, 
                            max_iter: int, 
                            init_v: np.ndarray) -> tuple:
    """Calculates the Gaussian approximation to the full conditional of the 
    GMRF, p(x|y,theta), using a second-order Taylor expansion.

        The Gaussian approximation is made by specifically matching the modal 
    configuration and the curvature at the mode.
        The calculations make use of the Cholesky decomposition. We assume that
    the precision matrix is symmetric positive-definite and that even after 
    summing diag(c) it maintains this property. Since this matrix is sparse,
    the Cholesky decomposition will also be sparse, leading to speed-ups. The
    Cholesky decomposition is faster than LU decomposition. The Cholesky 
    decomposition is accelerated using the GPU.

    Args:
        objective (Callable): log-likelihood function to use in the 
            approximation.
        Q (sparse.dia_matrix): precision matrix of the GMRF.
        h (float): step used in the finite differences approximations of the
            derivatives.
        threshold (float): maximum Euclidean-distance required to consider that
            the Newton-Raphson method has converged.
        max_iter (int): maximum number of iterations before stopping the 
            Newton-Raphson method and announcing that the method has not 
            converged.
        init_v (np.ndarray): initial point around which to perform the Taylor
            expansion on the log-likelihood function.

    Raises:
        Exception: The iteration surpasses the maximum number iterations 
            specified in the arguments.

    Returns:
        new_x (np.ndarray): mean of the Gaussian approximation to the full 
            conditional of the GMRF.
        matrix_A (float): determinant of the precision matrix of the 
            Gaussian approximation to the full conditional of the GMRF.
    """
    
    current_x = init_v
    for _ in range(max_iter):
        b, c = approx_taylor_expansion(objective, current_x, h)
        b = b.to('cpu')
        c = np.array(c.to('cpu'))

        ## Calculates the mean solving an equation of the type: 
        ##    matrix_A @ x = b
        matrix_A = Q + sparse.diags(c, format='csc')
        chol_factor = cholesky(matrix_A)
        new_x = torch.tensor(chol_factor(b), dtype=torch.float64).to(device)

        if linalg.norm(b) < threshold:
            return (new_x, np.exp(chol_factor.logdet()))
        
        current_x = new_x
    
    return (new_x, np.exp(chol_factor.logdet()))



###############################################################################
# calculating p(theta | y)

def general_determinant(A: sparse.dia_matrix, zero: float=1e-5) -> float:
    """Calculates a general determinant that applies to singular matrices, i.e.,
    matrices that are not of full-rank. The determinant here corresponds to the
    product of all the nonzero eigenvalues of the matrix.

        Here, we make use of the assumptions of A being a diagonal matrix, so
    that we can use a banded solver as these are much faster than general sparse
    solvers.

    Args:
        A (sparse.dia_matrix): real symmetric matrix.
        zero (float): let e be an eigenvalue, if 0 <= e < zero, then e is
            considered to be zero, otherwise e has its own value.


    Returns:
        float: general determinant of A.
    
    Requires:
        A should only have nonzero values in the main diagonal and the first
            offset diagonals of the matrix.
        A.data should be ordered such that the main diagonal is the middle 
            element of the array, and that the the top offset diagonal should be
            the first element of the array.
        zero >= 0
        zero should be a very small number.
    """

    eigvs = linalg.eigvals_banded(A.data[:2])
    return np.prod(eigvs[eigvs > zero])


def approx_marg_post_of_theta(data_likelihood: Callable,
                                theta: np.ndarray,
                                gmrf_likelihood: Callable,
                                theta_dist: Callable,
                                gaus_approx_mean: np.ndarray,
                                gaus_approx_det: float) -> float:
    """Approximates the marginal posterior of theta, p(theta|y).

    Args:
        data_likelihood (Callable): function that calculates the likelihood of
            the data, given the provided means
        theta (float): second parameter of the likelihood of the data
        gmrf_likelihood (Callable): function that calculates the density of the
            GMRF (that corresponds to a Multivariate Normal Distribution)
        theta_dist (Callable): probability density function of the distribution
            of theta
        gaus_approx_mean (np.ndarray): mean of the Gaussian approximation to the
            full conditional of the GMRF.
        gaus_approx_det (float): determinant of the precision matrix of the 
            Gaussian approximation to the full conditional of the GMRF.

    Returns:
        float: approximation of the (natural) log probability of the marginal
            posterior of theta. 
    """
    
    x = gaus_approx_mean
    dim = x.shape[0]

    # ln p(y|x, theta)
    likelihood = data_likelihood(x).cpu()

    # ln p(x|theta)
    gmrf_prior = np.log(gmrf_likelihood(np.array(x.cpu())))
    
    # ln p(theta)
    theta_prior = np.log(theta_dist(theta))

    # ln p_G(x|y, theta)
    ga_full_conditional_x = np.sqrt(gaus_approx_det * ((2*np.pi)**(-dim)))
    ga_full_conditional_x = np.log(ga_full_conditional_x)

    # ln p(theta | y)
    return likelihood + gmrf_prior + theta_prior - ga_full_conditional_x



###############################################################################
# exploring p(theta | y)


def lbfgs(p_theta_given_y: Callable, init_guess: float, bounds: list,
            n_hist_updates: int) -> np.ndarray:
    """Calculates the mode of the objective function using the l-BFGS-B 
    quasi-Newtonian method. This method uses a limitd amount of computer memory,
    because it uses only a few vectors that represent the approximation to the
    Hessian implicitly.

        In the context of this problem, the objective function corresponds to
    the density function, p(theta | y). Here, use the negative logarithmic 
    density function. We use the negative, because our original goal is to 
    maximize p(theta | y), so if we negate it, our goal is now to minimize
    -p(theta | y). We use the log probability, because it is more numerically
    stable and since it is a monotone function, the mode for the minimization
    of the log is the same as the mode for the original problem.

    Args:
        p_theta_given_y (Callable): objective function to obtain the value that
            minimizes the function.
        init_guess (float): initial value to try in the minimization.
        bounds (list): constraints on the possible values taken by the function.
        n_hist_updates (int): number of recorded history updates used in the
            l-BFGS-B method.

    Returns:
        np.ndarray: input that minimizes the objective function.
    """
    

    result = minimize(p_theta_given_y,
                        init_guess, 
                        method='L-BFGS-B', 
                        jac='3-point', 
                        bounds=bounds,
                        options={'maxcor': n_hist_updates, 
                                    'finite_diff_rel_step':1e-3})

    return result.x