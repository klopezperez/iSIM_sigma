from isim_sigma.sigma import stratified_sigma, random_sigma
from isim_sigma.utils import rdkit_pairwise_sim, npy_to_rdkit
import numpy as np
import pandas as pd

# Load the complete chembl_33 natural product dataset
fps = np.load("chembl_33_fps.npy", mmap_mode="r")
fps_rdkit = npy_to_rdkit(fps)

# Calculate the iSIM-sigma to monitor the error
strat_sigma = stratified_sigma(fps, n=50, n_ary='JT')

# Calculate the true sigma
true_sigma = rdkit_pairwise_sim(fps_rdkit)[1]

sigma_error = np.abs(strat_sigma - true_sigma)
print(sigma_error)

# Define random sigma
def random_sigma(fps_rdkit, n):
    indexes_rand = np.random.choice(len(fps_rdkit), n, replace = False)
    fps_rand = fps_rdkit[indexes_rand]

    average, std = rdkit_pairwise_sim(fps_rand)

    return std


# Do three trials of random sigma
n_trials = 3
data = {}
for i in range(n_trials):
    errors = []
    for n in range(10, len(fps_rdkit), 10):
        sigma = random_sigma(fps_rdkit, n)
        errors.append(np.abs(sigma - true_sigma))

    data[f'Trial_{i}'] = errors

# Save the data
dataframe = pd.DataFrame(
    data,
    index=np.arange(10, len(fps_rdkit), 10)
)

dataframe.to_csv("random_sigma_errors.csv", index_label="n")

better_errors = []
for row in dataframe.iterrows():
    better_errors.append(row['Trial_0'] < sigma_error and row['Trial_1'] < sigma_error and row['Trial_2'] < sigma_error)

dataframe['Better'] = better_errors
dataframe.to_csv("random_sigma_errors.csv", index_label="n")
    