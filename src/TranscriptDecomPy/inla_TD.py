import numpy as np

from scipy.stats import nbinom


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
