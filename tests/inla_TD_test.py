import sys
sys.path[0] += "/../src"

from TranscriptDecomPy import inla_TD as tdp

###############################################################################

import unittest

import numpy as np

from scipy.stats import halfnorm, nbinom

###############################################################################


class TestLikelihood(unittest.TestCase):

    # --------------------------------------- #
    #  tests for the function getting_bags    #
    # --------------------------------------- #

    def test_likelihood_with_different_dist(self):
        
        list_of_tests = [halfnorm.rvs(10, 4, 100) for i in range(31)]
        list_of_tests.append(np.full(100, 21))
        for _ in range(40):
            list_of_tests.append(halfnorm.rvs(10, 4, 100))
        list_of_tests = np.array(list_of_tests)
        print(list_of_tests.shape)

        results = tdp.log_likelihood_neg_binom(list_of_tests, 
                                                np.full(100, 10), [0.3])
        max_value_idx = np.argmax(results)
        self.assertEqual(max_value_idx, 31)



unittest.main()



def read_data(name: str) -> np.ndarray:

    with open(name, 'r', encoding='utf8') as datafile:
        data = []
        for line in datafile.read().splitlines()[1:]:
            data.append(line.split(",")[1:])
        return np.array(data, dtype=int)
    

# data = read_data("cage_data_test.csv")
# x = np.full(109, 10000)[np.newaxis, :]

# import time
# start = time.time()
# print(np.sum(individual_log_neg_binom_likelihood(data, x, np.array([0.2])), 1))
# print(f"It took {time.time() - start}")