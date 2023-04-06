import numpy as np

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


# def newton_raphson_method()