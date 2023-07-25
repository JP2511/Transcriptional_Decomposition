import sys
sys.path[0] += "/../src/TranscriptDecomPy/"

import torch
import numpy as np

import utils
import distributions_TD as dist_TD

###############################################################################


def sample_gmrf(theta_intercept: torch.tensor,
                theta_PD: torch.tensor,
                theta_PI: torch.tensor,
                n_feat: int) -> torch.tensor:
    """Creates the latent variable for a synthetic dataset. The latent variable
    corresponds to the sum a first-order Random Walk component, a Random Effects
    component and an intercept.

    Args:
        theta_intercept (torch.tensor): parameter that controls the precision of
            the intercept.
        theta_PD (torch.tensor): parameter that controls the precision of the
            Random Walk.
        theta_PI (torch.tensor): parameter that controls the precision of the
            Random Effects.
        n_feat (int): number of features of the synthetic dataset.

    Returns:
        torch.tensor: tensor that contains each component and then the latent
            variable.
    """
    
    intercept = dist_TD.sample_intercept(theta_intercept, 1)[0]
    RW1 = dist_TD.creating_RW1_Q(theta_PD, n_feat)
    PD = dist_TD.sample_IGMRF(RW1, 1)[0]
    PI = dist_TD.sample_random_effects(n_feat, theta_PI, 1)[0]
    
    eta = intercept + PD + PI
    return torch.concatenate((intercept, PD, PI, eta))


def create_GMRF_samples_file(theta_intercept: torch.tensor,
                                theta_PD: torch.tensor,
                                theta_PI: torch.tensor,
                                obs_nums: list, 
                                feat_nums: list):
    """Creates the latent variables for synthetic datasets. It creates the
    latent variables for each combination of number of observations and number
    of features.

    Args:
        theta_intercept (torch.tensor): parameter that controls the precision of
            the intercept.
        theta_PD (torch.tensor): parameter that controls the precision of the
            Random Walk.
        theta_PI (torch.tensor): parameter that controls the precision of the
            Random Effects.
        obs_nums (list): numbers of observations used to create synthetic
            datasets.
        feat_nums (list): numbers of features used to create synthetic datasets.
    """

    data = {}
    for n_feat in feat_nums:
        for obs in obs_nums:
            sample = sample_gmrf(theta_intercept, theta_PD, theta_PI, n_feat)
            print((n_feat, obs))
            data[f"{n_feat}_{obs}"] = sample.cpu().numpy()

    np.savez_compressed('gmrf_samples', **data)


##############################################################################

if __name__ == '__main__':
    import time

    poss_obs   = [2, 5, 10, 50, 100, 500, 1_000, 5_000, 10_000, 20_000]
    poss_feats = [25, 30, 50, 100, 200, 500, 1_000, 4_000]

    start = time.time()
    create_GMRF_samples_file(theta_intercept=utils.gen_tensor(0.6),
                                theta_PD=utils.gen_tensor(30),
                                theta_PI=utils.gen_tensor(1),
                                obs_nums=poss_obs,
                                feat_nums=poss_feats)
    print(f"\nDemorei {time.time() - start}")