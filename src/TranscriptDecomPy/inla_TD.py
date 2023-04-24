import numpy as np

from scipy import linalg, sparse
from scipy.stats import nbinom
from scipy.optimize import minimize

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
    
    if len(pm_for_single_obs.shape) > 0:
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
    a mixed-effects iid model, the assumption should be valid. It should also be
    true that precision matrix should only have the main diagonal and the first
    offset diagonals with nonzero elements.

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
        matrix_A = Q + sparse.diags(c, format='dia')
        result_b = Q @ mu + b

        new_x = linalg.solve_banded((1,1), matrix_A.data, result_b)

        if linalg.norm(current_x - new_x) < threshold:
            return (new_x, matrix_A)
        
        current_x = new_x
    
    raise Exception("Max iteration achieved and Newton_Raphson method " + 
                    "did not converge")



# ###############################################################################
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
                                theta: float,
                                alpha: np.ndarray,
                                Q: sparse.dia_matrix,
                                Q_det: float,
                                theta_dist: Callable,
                                gaus_approx_mean: np.ndarray,
                                gaus_approx_Q: sparse.dia_matrix) -> float:
    """Approximates the marginal posterior of theta, p(theta|y).

        Since we assume that the precision matrix Q of the GRMF or the matrix 
    Q + diag(c) are singular (not of full-rank), we calculate the general 
    determinant instead of the common determinant. These calculations assume
    that these matrices are diagonal (main diagonal and the first offset 
    diagonals) to use in a banded_solver making calculations quite fast.

    Args:
        data_likelihood (Callable): function that calculates the likelihood of
            the data, given the provided means
        theta (float): second parameter of the likelihood of the data
        alpha (np.ndarray): means of the GMRF
        Q (sparse.dia_matrix): precision matrix of the GMRF
        Q_det (float): determinant of the precision matrix of the GRMF
        theta_dist (Callable): probability density function of the distribution
            of theta
        gaus_approx_mean (np.ndarray): mean of the Gaussian approximation to the
            full conditional of the GMRF.
        gaus_approx_Q (sparse.dia_matrix): precision matrix of the Gaussian 
            approximation to the full conditional of the GMRF.

    Returns:
        float: approximation of the (natural) log probability of the marginal
            posterior of theta. 
    """
    
    dim = alpha.shape[0]
    x = gaus_approx_mean

    # ln p(y|x, theta)
    likelihood = data_likelihood(x)

    # ln p(x|theta)
    exponent = -(1/2) * (x - alpha) @ Q @ (x - alpha)[:, np.newaxis]
    gmrf_prior = np.sqrt(Q_det * ((2*np.pi)**(-dim))) * (np.exp(exponent))
    gmrf_prior = np.log(gmrf_prior)[0]
    
    # ln p(theta)
    theta_prior = np.log(theta_dist(theta))

    # ln p_G(x|y, theta)
    ga_det = general_determinant(gaus_approx_Q)
    ga_full_conditional_x = np.log(np.sqrt(ga_det * ((2*np.pi)**(-dim))))

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