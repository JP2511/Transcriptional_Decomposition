import numpy as np

from scipy import linalg, sparse
from scipy.stats import nbinom

from typing import Callable


###############################################################################


def log_likelihood_neg_binom(obs: int, x: int, theta: float) -> float:
    """Calculates the log likelihood of the negative binomial distribution.

    Args:
        obs (int): observation of which to calculate the likelihood.
        x (int): number of successes until the experiment is stopped in the 
            negative binomial distribution.
        theta (float): success probability in experiment in the negative 
            binomial distribution.

    Returns:
        float: log likelihood of the point given the specified negative binomial 
            distribution.

    Requires:
        obs > 0
        x > 0
        0 <= theta <= 1
    """

    return np.sum(nbinom.logpmf(obs, x, theta), 1)


def central_differences(objective: Callable, x: float, h: float) -> float:
    """Approximates the derivative of a function using finite differences, in
    particular using the central differences.

    Args:
        objective (Callable): function whose derivative we want to approximate.
        x (float): value around which we want the approximation to the 
            derivative.
        h (float): step of the approximation. Ideally, it should be as close to
            zero as possible.

    Returns:
        float: approximate derivative value at the x point.
    """

    return (objective(x + h) - objective(x - h)) / (2*h)


###############################################################################
# Newton-Raphson method

def approx_taylor_expansion(objective: Callable, x: np.ndarray, 
                                h: np.ndarray) -> tuple:
    """Approximates the second and third terms of a quadratic Taylor expansion.

    Args:
        objective (Callable): function we are expanding.
        x (np.ndarray): value around which we are performing the Taylor 
            expansion.
        h (np.ndarray): step of the approximation. Ideally, it should be as 
            close to zero as possible.

    Returns:
        b: approximate second term of the quadratic Taylor expansion.
        c: approximate third term of the quadratic Taylor expansion.
    """
    
    c = - ((objective(x + h) - 2*objective(x) + objective(x - h)) / (h**2))
    b = central_differences(objective, x, h) - x * c
    return (b, c)


def newton_raphson_method(objective: Callable, Q: sparse.dia_matrix, 
                            mu: np.ndarray, h: float, threshold: float, 
                            max_iter: int, init_v: np.ndarray) -> tuple:
    """Calculates the Gaussian approximation to the full conditional of the 
    GMRF, p(x|y,theta), using a second-order Taylor expansion.

        The Gaussian approximation is made by specifically matching the modal 
    configuration and the curvature at the mode.
        The calculations make use of sparse diagonal matrix linear algebra. 
    We use this because we make the assumption that the precision matrix of the
    GMRF is not only sparse but diagonal as well. Since we are only looking at
    implementing the model for a combination of a first-order Random-Walk and
    a mixed-effects iid model, the assumption should be valid.

    Args:
        objective (Callable): log-likelihood function to use in the 
            approximation.
        Q (sparse.dia_matrix): precision matrix of the GMRF.
        mu (np.ndarray): mean of the GMRF.
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
        matrix_A (np.ndarray): precision matrix of the Gaussian approximation
            to the full conditional of the GMRF.
    """
    
    current_x = init_v
    for _ in range(max_iter):
        b, c = approx_taylor_expansion(objective, current_x, h)

        ## Calculates the mean solving an equation of the type: 
        ##    matrix_A @ x = result_b
        matrix_A = Q + sparse.diags(c)
        result_b = Q @ mu + b
        new_x = linalg.solve_banded(matrix_A, result_b)

        if linalg.norm(current_x - new_x) < threshold:
            return (new_x, matrix_A)
        
        current_x = new_x
    
    raise Exception("Max iteration achieved and Newton_Raphson method " + 
                    "did not converge")