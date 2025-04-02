from isim_sigma.utils import pairwise_average, npy_to_rdkit, rdkit_pairwise_sim
import numpy as np
from isim_sigma.sigma import get_stdev_tanimoto_fast, get_stdev_russell_fast, get_stdev_sokal_fast, random_sigma, stratified_sigma
import pandas as pd
import time

# Load the fingerprints
fps = np.load('chembl_33.npy', mmap_mode = 'r')

data = []
times = []
for n_ary in ['JT', 'RR', 'SM']:
    for i in range(500):
        entry = []
        time_entry = []
        entry.append(n_ary)
        time_entry.append(n_ary)

        n_subset = np.random.randint(1000, 5000)
        n_subset_2 = int(n_subset/2)
        entry.append(n_subset)
        time_entry.append(n_subset)

        # Choose two chunks of fingperprints from the dataset
        # We selected two random chunks of consecutive molecules (as they were ordered in the dataset) because
        # taking them randomly will result always in almost the same std
        chunk_1 = np.random.choice(len(fps), 1, replace = False)[0]
        while chunk_1 + n_subset/2 > len(fps):
            chunk_1 = np.random.choice(len(fps), 1, replace = False)[0]

        chunk_2 = np.random.choice(len(fps), 1, replace = False)[0]
        while chunk_2 + n_subset/2 > len(fps) or (chunk_2 > chunk_1 and chunk_2 < chunk_1 + n_subset/2):
            chunk_2 = np.random.choice(len(fps), 1, replace = False)[0]

        fps_subset = fps[chunk_1:chunk_1+n_subset_2]
        fps_subset = np.concatenate((fps_subset, fps[chunk_2:chunk_2+n_subset_2]))


        # Do stratisfied sampling with multiple samples
        for n_strat in [10, 25, 50]:
            start = time.time()
            # Calculate the pairwise average of the sampled indexes
            std = stratified_sigma(fps_subset, n_strat, n_ary = n_ary)
            
            time_entry.append(time.time() - start)
            entry.append(std)

        # Do random sampling three times
        for i in range(3):
            start = time.time()
            std = random_sigma(fps_subset, 50, n_ary = n_ary)

            time_entry.append(time.time() - start)
            entry.append(std)

        # Do the pairwise average
        if n_ary == 'JT':
            fps_rdkit = npy_to_rdkit(fps_subset)

            # Calculate the pairwise similarity of the sampled indexes
            start = time.time()
            average_rdkit, std_rdkit = rdkit_pairwise_sim(fps_rdkit)

            time_entry.append(time.time() - start)
            entry.append(std_rdkit)
        else:
            start = time.time()
            average, std =  pairwise_average(fps_subset, n_ary = n_ary)

            time_entry.append(time.time() - start)
            entry.append(std)

        # Do the fast exact method
        if n_ary == 'JT':
            start = time.time()
            stdev_tanimoto = get_stdev_tanimoto_fast(fps_subset)
            time_entry.append(time.time() - start)
            entry.append(stdev_tanimoto)
        elif n_ary == 'RR':
            start = time.time()
            stdev_russell = get_stdev_russell_fast(fps_subset)
            time_entry.append(time.time() - start)
            entry.append(stdev_russell)
        elif n_ary == 'SM':
            start = time.time()
            stdev_sokal = get_stdev_sokal_fast(fps_subset)
            time_entry.append(time.time() - start)
            entry.append(stdev_sokal)

        data.append(entry)
        times.append(time_entry)

        # Create a dataframe with the results
        df = pd.DataFrame(data, columns = ['n_ary', 'n_subset', 'strat_10', 'strat_25', 'strat_50', 'random_1', 'random_2', 'random_3', 'rdkit', 'fast_exact'])

        # Create a dataframe with the times
        df_times = pd.DataFrame(times, columns = ['n_ary', 'n_subset', 'strat_10', 'strat_25', 'strat_50', 'random_1', 'random_2', 'random_3', 'pairwise', 'fast_exact'])

        # Save the dataframe
        df.to_csv('std_stat_chunks.csv', index = False)

        # Save the times
        df_times.to_csv('std_stat_chunks_times.csv', index = False)

        # Reset chunks for the next iteration
        chunk_1, chunk_2 = None, None