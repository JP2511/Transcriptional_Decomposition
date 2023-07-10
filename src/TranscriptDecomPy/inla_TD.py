import torch
import torchmin

import numpy as np

from typing import Callable
from functools import partial
from itertools import product
from scipy.optimize import root
from torch.distributions.normal import Normal

###############################################################################
# set up

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)

###############################################################################
# utils

def general_determinant(A: torch.tensor, zero_dec: int) -> float:
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
        log_det (float): logarithm of the general determinant of A.
    
    Requires:
        A should only have nonzero values in the main diagonal and the first
            offset diagonals of the matrix.
        A.data should be ordered such that the main diagonal is the middle 
            element of the array, and that the the top offset diagonal should be
            the first element of the array.
        zero >= 0
        zero should be a very small number.
    """

    evs = torch.linalg.eigvalsh(A)
    non_zero_evs = evs[torch.round(evs, decimals=zero_dec) != 0]
    log_det = torch.sum(torch.log(non_zero_evs))

    return log_det


def gen_tensor(value: float or list, shape: int=None) -> torch.tensor:
    """Creates a float64 tensor on the GPU.

    Args:
        value (float or list): value which we want to convert one instance or 
            multiple instance of into a tensor.
        shape (int, optional): in case the value is a single float that we would
            like to repeat to form a 1-D vector, additionally to the value, the
            length of the tensor should also be provided. Defaults to None.

    Returns:
        torch.tensor: tensor created.
    """

    if shape:
        return torch.full((shape,), value, dtype=torch.float64, device=device)
    return torch.tensor(value, dtype=torch.float64, device=device)


###############################################################################
# GMRF pdf

def gmrf_density(x: torch.tensor, log_det: float, Q: torch.tensor) -> float:
    """Calculates the log density of a value using the probability density
    function of a multivariate normal distribution.

    Args:
        x (torch.tensor): point used to calculate the density.
        log_det (float): determinant of the precision matrix of the multivariate
            normal distribution used.
        Q (torch.tensor): precision matrix of the multivariate normal 
            distribution used.

    Returns:
        density (float): log density of the point given in the distribution
            given.
    """
    
    n = x.shape[0]
    exp_value = -(1/2) * (x[None, :] @ Q @ x[:, None])[0]
    double_pi = gen_tensor(2*torch.pi)
    log_density = ( -n * torch.log(double_pi) + log_det ) / 2 + exp_value
    return log_density


def create_gmrf_density_func(Q: torch.tensor) -> Callable:
    """Generates a function that calculates the density of the GMRF assuming the
    the precision matrix given.

    Args:
        Q (torch.tensor): precision matrix of the GMRF.

    Returns:
        Callable: function that calculates the density of the GMRF.
    """
    
    log_det = general_determinant(Q, 5)

    return partial(gmrf_density, log_det=log_det, Q=Q)


###############################################################################
# Newton-Raphson method

def constrain_NR(x: torch.tensor, 
                    bounds: tuple,
                    L: torch.tensor,
                    A: torch.tensor, 
                    e: torch.tensor) -> torch.tensor:
    """Restructures a given tensor such that it obeys the given constraints and
    is bounded by the given interval.

    Args:
        x (torch.tensor): tensor to constrain.
        bounds (tuple): pair of lower and upper bounds that create an interval
            where each element of x must fall within. If an element x is higher
            than the upper bound, then the value is replaced with the upper
            bound. The same happens in the other direction for the lower bound.
        L (torch.tensor): lower tringular matrix that is created from the
            Cholesky decomposition of the precision matrix associated with the
            x tensor.
        A (torch.tensor): matrix defining the linear relationships between the
            values such that they obey a certain constraint. A @ x = e
        e (torch.tensor): tensor defining the result of the linear relationships
            defined in A and applied to x. A @ x = e

    Returns:
        torch.tensor: retructured x such that it obeys the linear constraints
            defined and that each value of x is bounded by the interval given.
    """
    
    
    lower_bound, upper_bound = bounds
    x = torch.where(x > upper_bound, upper_bound, x)
    x = torch.where(x < lower_bound, lower_bound, x)
    
    # Ax - e
    constraint = A @ x - e

    # Q^{-1} A^T
    invQ_At = torch.cholesky_solve(A.T, L)

    # ( A Q^{-1} A^T )^{-1}
    inv_A_invQ_At = torch.linalg.inv(A @ invQ_At)

    # x - Q^{-1} A^T ( A Q^{-1} A^T )^{-1} ( Ax - e )
    new_x = x - invQ_At @ inv_A_invQ_At @ constraint

    return new_x


def p_x_given_y_theta(data_likelihood: Callable, 
                        Q: torch.tensor, 
                        init_v: torch.tensor) -> tuple:
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
        data_likelihood (Callable): log-likelihood function to use in the 
            approximation.
        Q (sparse.dia_matrix): precision matrix of the GMRF.
        init_v (np.ndarray): initial point around which to perform the Taylor
            expansion on the log-likelihood function.

    Returns:
        new_x (torch.tensor): mean of the Gaussian approximation to the full 
            conditional of the GMRF.
        L (torch.tensor): lower triangular matrix obtained by the Cholesky
            decomposition of the precision matrix of the Gaussian approximation,
            which corresponds to the sum of the precision matrix of the GMRF and
            a diagonal matrix with the second order derivatives of the 
            data_likelihood.
    """
    
    obj_f = lambda x: -data_likelihood(x) + 0.5 * (x[None] @ Q @ x[:, None])[0]
    current_x = torchmin.minimize(obj_f, init_v, method='newton-cg').x

    Hess = torch.autograd.functional.hessian(data_likelihood, current_x)
    c = -torch.diag(Hess)

    n_feat = (current_x.shape[0] - 1) // 2

    # constraining the values of x
    L = torch.linalg.cholesky(Q + torch.diag(c))
    A = torch.hstack((gen_tensor(0), gen_tensor(1, n_feat), 
                        gen_tensor(0, n_feat)))[None, :]
    
    current_x = constrain_NR(current_x, (-100, 100), L, A, gen_tensor(0))

    return (current_x, L)


###############################################################################
# calculating p(theta | y)

def approx_marg_post_of_theta(data_likelihood: Callable,
                                theta: np.ndarray,
                                gmrf_likelihood: Callable,
                                theta_dist: Callable,
                                gaus_approx_mean: torch.tensor,
                                gaus_approx_L: torch.tensor) -> float:
    """Approximates the marginal posterior of theta, p(theta|y).

    Args:
        data_likelihood (Callable): function that calculates the likelihood of
            the data, given the provided means
        theta (float): hyperparameters of the model.
        gmrf_likelihood (Callable): function that calculates the density of the
            GMRF (that corresponds to a Multivariate Normal Distribution)
        theta_dist (Callable): probability density function of the distribution
            of theta
        gaus_approx_mean (torch.tensor): mean of the Gaussian approximation to
            the full conditional of the GMRF.
        gaus_approx_L (torch.tensor): lower triangular matrix obtained by the 
            Cholesky decomposition of the precision matrix of the Gaussian
            approximation.

    Returns:
        float: approximation of the (natural) log probability of the marginal
            posterior of theta. 
    """
    
    x = gaus_approx_mean
    dim = x.shape[0]

    # ln p(y|x, theta)
    likelihood = data_likelihood(x)

    # ln p(x|theta)
    gmrf_prior = gmrf_likelihood(x)
    
    # ln p(theta)
    theta_prior = theta_dist(theta)

    # ln p_G(x|y, theta)
    det = torch.sum(torch.log(torch.diag(gaus_approx_L)))
    ga_full_conditional_x = det - (dim/2) * np.log(2*np.pi)

    # ln p(theta | y)
    return likelihood + gmrf_prior + theta_prior - ga_full_conditional_x



###############################################################################
# exploring p(theta | y)

def calc_mode_of_marg_post_theta(p_theta_given_y: Callable, 
                                    bounding_func: Callable,
                                    init_guess: torch.tensor) -> torch.tensor:
    """Calculates the mode of the objective function using the l-BFGS-B 
    quasi-Newtonian method. This method uses a few vectors that represent the 
    approximation to the Hessian implicitly.

        In the context of this problem, the objective function corresponds to
    the marginal posterior of theta, p(theta | y). Here, use the negative 
    logarithmic density function. We use the negative, because our original goal
    is to maximize p(theta | y), so if we negate it, our goal is now to minimize
    -p(theta | y). We use the log probability, because it is more numerically
    stable and since it is a monotone function, the mode for the minimization
    of the log is the same as the mode for the original problem.

    Args:
        p_theta_given_y (Callable): objective function to obtain the value that
            minimizes the function.
        bounding_func (Callable): function that enforces bounds. Since minimize
            function utilized does not allow for box constraints, I use this
            function to convert the input from the minimize function to a set
            of values that is inside of the box constraints.
        init_guess (float): initial value to try in the minimization.

    Returns:
        torch.tensor: input that minimizes the objective function.
    """

    def calc_bounded_theta_posterior(theta: torch.tensor) -> torch.tensor:
        bounded_theta = bounding_func(theta)
        return p_theta_given_y(bounded_theta)[-1]

    result = torchmin.minimize(calc_bounded_theta_posterior, init_guess, 
                                method='l-bfgs').x
    
    return bounding_func(result)


def gen_f_theta_explorer(neg_ln_p_theta_given_y: Callable,
                            bounding_func: Callable,
                            init_guess: torch.tensor) -> Callable:
    """Creates a function that allows for the reparameterized exploration of
    theta values.

    Args:
        neg_ln_p_theta_given_y (Callable): function that given a vector of
            thetas returns the posterior marginal of theta.
        bounding_func (Callable): function that constraints the inputs fed to
            the posterior marginal of theta that were given by the minimizer.
        init_guess (torch.tensor): initial guess for the values of theta (the
            parameters).

    Raises:
        Exception: in the explorer function returned, if the vector given makes
            one of the thetas become negative, it stops the execution as it will
            not run either way.
    
    Returns:
        explorer (Callable): function that allows the reparameterized exporation
            of the posterior marginal of theta.
    """
    
    mode = calc_mode_of_marg_post_theta(neg_ln_p_theta_given_y, bounding_func,
                                        init_guess)
    print(f"\nShowing mode found: {mode}")
    
    correct_neg_post_marg_theta = lambda x: neg_ln_p_theta_given_y(x)[-1]
    theta_Hess = torch.autograd.functional.hessian(correct_neg_post_marg_theta, 
                                                    mode)
    eigVals, eigVec = torch.linalg.eigh(theta_Hess)
    Lmbda = torch.diag(1/torch.sqrt(eigVals))

    def explorer(z: torch.tensor) -> tuple:
        theta = mode[:, None] + eigVec @ Lmbda @ z[:, None]
        theta = theta.T[0]
        if torch.any(theta < 0):
            raise Exception("Failed becaused one of the thetas in the " +
                            "explorer has a negative value. Consider using a" +
                            "smaller step")
        
        ga_mode, ga_L, neg_p_theta = neg_ln_p_theta_given_y(theta)
        vars = torch.diag(torch.cholesky_inverse(ga_L))
        return torch.hstack((z, theta, ga_mode, vars, -neg_p_theta))
    
    return explorer


def obtain_theta_axis_points(explorer: Callable, 
                                theta_dim: int,
                                step: float,
                                stop: float) -> torch.tensor:
    """Performs the initial exploration of the posterior marginal theta along
    the axis (in both directions) of the reparameterization

    Args:
        explorer (Callable): function that allows the reparameterized exporation
            of the posterior marginal of theta.
        theta_dim (int): number of thetas used/that are going to be explored.
        step (float): distance between the thetas fed to the explorer.
        stop (float): threshold to stop the exploration of the thetas in the
            current direction.

    Returns:
        points (torch.tensor): the points explored.
    
    Requires:
        points should have the following structure:
            -each row corresponds to a single point
            -the first `theta_dim` columns should be the parameterized \\thetas
            -the second `theta_dim`columns should be the \\thetas in the
            original space
            -the next `n_feat` columns should be the mode of the Gaussian
            approximation for p_G(x | y, \\theta)
            -the next `n_feat` columns should be the variance of the Gaussian
            approximation for p_G(x | y, \\theta)
            -the last column correspond to p(\\theta | y).
    """
    
    init_z = gen_tensor(0, theta_dim)
    theta_0_results = explorer(init_z)
    p_theta_0 = theta_0_results[-1]
    print(f"\n Initial probability is {p_theta_0}")
    print(f"With original theta {theta_0_results[theta_dim:2*theta_dim]}")
    points = theta_0_results
    for i in torch.arange(theta_dim):
        for direction in (1, -1):
            iteration = 1
            while True:
                pos = torch.eye(theta_dim, device=device, 
                                dtype=torch.float64)[i] 
                pos = step * iteration * pos * direction
                new_point = explorer(pos)
                
                print("\nNew iteration:")
                print(pos)
                print(f"Current diff {p_theta_0 - new_point[-1]}")
                if p_theta_0 - new_point[-1] > stop:
                    break
                points = torch.vstack((points, new_point))
                iteration += 1
    return points


def obtain_combination_points(explorer: Callable,
                                points: torch.tensor,
                                step: float,
                                theta_dim: int) -> torch.tensor:
    """Performs the exploration of the posterior marginal of theta along the
    combinations of the explored reparameterized thetas (which were explored
    along the axis).

    Args:
        explorer (Callable): function that allows the reparameterized exporation
            of the posterior marginal of theta.
        points (torch.tensor): the points previously explored along the axis.
        step (float): distance between the thetas along the axis fed to the 
            explorer.
        theta_dim (int): number of thetas used/that are going to be explored.

    Returns:
        torch.tensor: explored points corresponding to the combination of values
            along the axis already explored. The structure is equal to the
            structure of the argument `points`.
    
    Requires:
        points should have the following structure:
            -each row corresponds to a single point
            -the first `theta_dim` columns should be the parameterized \\thetas
            -the second `theta_dim`columns should be the \\thetas in the
            original space
            -the next `n_feat` columns should be the mode of the Gaussian
            approximation for p_G(x | y, \\theta)
            -the next `n_feat` columns should be the variance of the Gaussian
            approximation for p_G(x | y, \\theta)
            -the last column correspond to p(\\theta | y).
    """

    z = points[:, :theta_dim]
    min_z = torch.min(z, 0)[0]
    max_z = torch.max(z, 0)[0]
    combinations = []
    for start, end in zip(min_z, max_z):
        point = torch.arange(start, end, step=step,  device=device, 
                                dtype=torch.float64)
        point = torch.round(point, decimals=2)
        combinations.append(point)
    
    structured_results = []
    for z in product(*combinations):
        z = torch.hstack(z)
        if torch.sum(z == 0) < theta_dim - 1:
            print("\nCombined iteration")
            print(z)
            structured_results.append(explorer(z))
    
    return torch.vstack(structured_results)


def explore_p_theta_given_y(neg_p_theta_given_y: Callable,
                            functional_bound: Callable,
                            init_guess: torch.tensor,
                            step: float,
                            stop: float) -> torch.tensor:
    """Performs the complete exploration of the posterior marginal of theta.

    Args:
        neg_p_theta_given_y (Callable): function that calculates the negative
            log posterior marginal of theta.
        bounding_func (Callable): function that constraints the inputs fed to
            the posterior marginal of theta that were given by the minimizer.
        init_guess (torch.tensor): initial guess for the values of theta (the
            parameters).
        step (float): distance between the thetas fed to the explorer.
        stop (float): threshold to stop the exploration of the thetas in the
            current direction.

    Returns:
        torch.tensor: all explored points.
    
    Ensures:
        points will have the following structure:
            -each row corresponds to a single point
            -the first `theta_dim` columns should be the parameterized \\thetas
            -the second `theta_dim`columns should be the \\thetas in the
            original space
            -the next `n_feat` columns should be the mode of the Gaussian
            approximation for p_G(x | y, \\theta)
            -the next `n_feat` columns should be the variance of the Gaussian
            approximation for p_G(x | y, \\theta)
            -the last column correspond to p(\\theta | y).
    """
    
    explorer = gen_f_theta_explorer(neg_p_theta_given_y, functional_bound,
                                    init_guess)
    theta_dim = init_guess.shape[-1]
    axis_points = obtain_theta_axis_points(explorer, theta_dim, step, stop)
    comb_points = obtain_combination_points(explorer, axis_points, step,
                                            theta_dim)
    return torch.vstack((axis_points, comb_points))