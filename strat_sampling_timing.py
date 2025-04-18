from isim_sigma.sigma import stratified_sigma
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
        start = time()
        sigma = stratified_sigma(fps[:n], 50, "JT")
        trial_time.append(time() - start)

    times[f'Trial_{i}'] = trial_time


# Save the data
dataframe = pd.DataFrame(
    times,
    index=np.arange(100, len(fps), 100)
)

dataframe.to_csv("isim_sigma_results/strat_sigma_times.csv", index_label="n")
