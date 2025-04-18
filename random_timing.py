from isim_sigma.sigma import stratified_sigma, random_sigma
from isim_sigma.utils import rdkit_pairwise_sim, npy_to_rdkit
import numpy as np
import pandas as pd
from time import time

# Load the complete chembl_33 natural product dataset
fps = np.load("chembl_33_np.npy", mmap_mode="r")
fps_rdkit = npy_to_rdkit(fps)

start = time()
# Calculate the iSIM-sigma to monitor the error
strat_sigma = stratified_sigma(fps, n=50, n_ary='JT')
print(f"strat_sigma:{strat_sigma}")
print(time()-start)

# Calculate the random_sigma
# Define random sigma
def random_sigma(fps_rdkit, n):
    indexes_rand = np.random.choice(len(fps_rdkit), n, replace = False)
    fps_rand = [fps_rdkit[i] for i in indexes_rand]

    average, std = rdkit_pairwise_sim(fps_rand)

    return std

start = time()
std = random_sigma(fps_rdkit, 1000)
print(time() - start)

start = time()
std = random_sigma(fps_rdkit, 32300)
print(time() - start)
 
