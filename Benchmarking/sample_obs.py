import numpy as np

from scipy.stats import nbinom

###############################################################################

def gen_obs_csv(theta_y: float, key: str, npz_array):
    """Generates a dataset with a observations sampled from a Negative Binomial
    distribution. The mean of the Negative Binomial distribution is connected
    to the latent variable by a link function that selects only the last n
    elements of the latent variable and exponentiates them.

    Args:
        theta_y (float): probability of success in the negative binomial
            distribution.
        key (str): latent variable in use. Also tells information about the
            number of features and number of observations that the synthetic
            dataset should have.
        npz_array: npz file that stores the arrays that contain the latent
            variables for the datasets.
    """

    print(key)
    n_feat, obs = key.split("_")
    n_feat, obs = int(n_feat), int(obs)
    
    eta = npz_array[key][-n_feat:]
    n = np.exp(eta) * (theta_y/ (1 - theta_y))
    if obs <= 500:
        data = nbinom.rvs(n, theta_y, size=(obs, n_feat))
        np.savetxt(f"datasets/data_{key}.csv", data, delimiter=',')
    else:
        with open(f"datasets/data_{key}.csv",'a') as csvfile:
            for _ in range(obs // 500):
                data = nbinom.rvs(n, theta_y, size=(500, n_feat))
                np.savetxt(csvfile, data, delimiter=',')


def gen_all_ds(theta_y: float, npz_array):
    """Creates all the synthetic datasets.

    Args:
        theta_y (float): probability of success in the negative binomial
            distribution.
        npz_array: npz file that stores the arrays that contain the latent
            variables for the datasets.
    """
    
    for key in npz_array.keys():
        gen_obs_csv(theta_y, key, npz_array)


###############################################################################

if __name__ == '__main__':
    npz_array = np.load('gmrf_samples.npz')

    gen_all_ds(0.357, npz_array)