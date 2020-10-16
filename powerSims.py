'''

File: powerSims.py
Author: Steve Goslen
Date: May 2018
Purpose: This will determine the statisical "power" that is needed by doing simulations
'''


import numpy as np

import scipy.stats

n_per_group = 10 

beta = 0.05

mean=1712660071.1431818
mean2=mean*0.95

sigma=38004515.57840848

# effect size = 0.8
group_means = [mean, mean2]
group_sigmas = [sigma, sigma]

n_groups = len(group_means)

# number of simulations
n_sims = 500

for samples in range(2,40):
    # store the p value for each simulation
    sim_p = np.empty(n_sims)
    sim_p.fill(np.nan)

    for i_sim in range(n_sims):

        data = np.empty([samples, n_groups])
        data.fill(np.nan)

        # simulate the data for this 'experiment'
        for i_group in range(n_groups):

            data[:, i_group] = np.random.normal(
                loc=group_means[i_group],
                scale=group_sigmas[i_group],
                size=samples
            )

        result = scipy.stats.ttest_ind(data[:, 0], data[:, 1])
        #Really needs to be just a 1 sided test so need to divide the
        #p-value that is returned by 2 so need to modify code below

        sim_p[i_sim] = result[1]

    # number of simulations where the null was rejected
    n_rej = np.sum(sim_p < 0.05)

    prop_rej = n_rej / float(n_sims)

    print("Power: ", prop_rej)
    if prop_rej > (1-beta):
        break

print("Final Power: ", prop_rej, "Samples: ", samples)
