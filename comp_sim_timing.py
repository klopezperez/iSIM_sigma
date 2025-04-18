from isim_sigma.isim import calculate_comp_sim
import numpy as np
import pandas as pd
from time import time

# Load the complete chembl_33 natural product dataset
fps = np.load("chembl_33_np.npy", mmap_mode="r")

# Do three trials of complementary similarity
n_trials = 3
times = {}
for i in range(n_trials):
    trial_time = []
    for n in range(100, len(fps), 100):
        sigma = calculate_comp_sim(fps[:n], n_ary="JT")

        start = time()
        sigma_sorted = np.argsort(sigma)
        trial_time.append(time() - start)

    times[f'Trial_{i}'] = trial_time


# Save the data
dataframe = pd.DataFrame(
    times,
    index=np.arange(100, len(fps), 100)
)

dataframe.to_csv("isim_sigma_results/arg_sort_times.csv", index_label="n")
