import numpy as np

from scipy import linalg, sparse
from scipy.stats import nbinom

from typing import Callable


###############################################################################


def log_likelihood_neg_binom(x: np.ndarray, obs: np.ndarray, 
                                theta: float) -> np.ndarray:
    """Calculates the log likelihood of the negative binomial distribution. It 
    is assumed that a point in this distribution is multivariate. As such, the
    log-likelihood of a such a point corresponds to the sum of the 
    log-likelihood its features. In case, there are multiple observations, the
    log-likelihood is calculated for all the observations, meaning that the
    value obtained corresponds to the sum of the log-likelihood of each point. 

    Args:
        x (np.ndarray): mean of the negative binomial distribution.
        obs (np.ndarray): observations of which to calculate the probability 
            mass (pm).
        theta (float): success probability in experiment in the negative 
            binomial distribution.

    Returns:
        np.ndarray: log likelihood of the point given the specified negative 
            binomial distribution.

    Requires:
        obs_i > 0
        x_i > 0
        0 <= theta <= 1
    """

    n = x * (theta/(1-theta))
    pm_for_single_obs = np.sum(nbinom.logpmf(obs, n, theta), -1)
    
    if len(pm_for_single_obs.shape) > 1:
        return np.sum(pm_for_single_obs, -1)
    
    return pm_for_single_obs


###############################################################################

def expand_increment_axis(n_feat: int, h: float) -> np.ndarray:
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

    return h * np.eye(n_feat)[:, np.newaxis, :]


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


def snd_order_central_differences(objective: Callable, x: np.ndarray,
                                    increment: np.ndarray, 
                                    h: float) -> float:
    """Approximates the second derivative of a function using finite 
    differences, in particular using second order central differences. This 
    function considers only derivatives where the objective function is twice
    partially derived by the same variable.

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
    
    diffs = objective(x + increment) - 2*objective(x) + objective(x - increment)
    return diffs / (h**2)



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

    increment = expand_increment_axis(x.shape[0], h)
    c = - snd_order_central_differences(objective, x, increment, h)

    b = fst_order_central_differences(objective, x, increment, h)
    b += x * c # verify
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
            # log_det = cholesky(matrix_A).logdet()
            # return np.exp(log_det)
            return (new_x, matrix_A)
        
        current_x = new_x
    
    raise Exception("Max iteration achieved and Newton_Raphson method " + 
                    "did not converge")