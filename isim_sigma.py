import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from isim_sampling import stratified_sampling
from isim_utils import pairwise_average

def get_stdev_russell_fast(arr):
    sums = np.sum(arr, axis=0)
    total = len(arr)*(len(arr)-1)/2
    probs = sums*(sums-1)/2/total

    #Covariance Step

    def get_covariance(i):
        output = []
        for j in range(i+1, len(arr[0])):
            counter = 0
            counter += arr[:, i] @ arr[:, j]
            prob = counter*(counter - 1)/2/total
            output.append(prob - probs[i]*probs[j])
        return np.sum(output)
    
    with parallel_backend('loky', n_jobs=10):
        covariances = Parallel()(delayed(get_covariance)(i) for i in range(len(arr[0])))
    
    covariance_sum = np.sum(covariances)

    return np.sqrt(np.sum(probs*(1-probs)) + 2*covariance_sum)/len(arr[0])

def get_stdev_tanimoto_fast(arr):
    sums = np.sum(arr, axis=0)
    total = len(arr)*(len(arr)-1)/2
    probs = sums*(sums-1)/2/total

    #Covariance Step

    def get_covariance(i):
        output = []
        for j in range(i+1, len(arr[0])):
            counter = 0
            counter += arr[:, i] @ arr[:, j]
            prob = counter*(counter - 1)/2/total
            output.append(prob - probs[i]*probs[j])
        return np.sum(output)
    
    with parallel_backend('loky', n_jobs=10):
        covariances = Parallel()(delayed(get_covariance)(i) for i in range(len(arr[0])))
    
    covariance_sum = np.sum(covariances)

    ### Getting Denominator
    #Crude approximation
    sums_zeros = len(arr) - sums
    denom = np.sum(total - sums_zeros*(sums_zeros-1)/2)/total
    return np.sqrt(np.sum(probs*(1-probs)) + 2*covariance_sum)/denom

def get_stdev_sokal_fast(arr):
    sums = np.sum(arr, axis=0)
    total = len(arr)*(len(arr)-1)/2
    probs = sums*(sums-1)/2/total

    sums_zeros = len(arr) - sums
    probs += sums_zeros*(sums_zeros-1)/2/total
    #Covariance Step

    def get_covariance(i):
        output = []
        for j in range(i+1, len(arr[0])):
            counter = 0
            counter += arr[:, i] @ arr[:, j]
            prob = counter*(counter - 1)/2/total
            counter_zeros = 0
            counter_zeros += (1-arr[:, i]) @ (1-arr[:, j])
            prob += counter_zeros*(counter_zeros-1)/2/total

            counter_pair_1 = arr[:, i] @ (1 - arr[:, j])
            counter_pair_2 = (1-arr[:,i]) @ arr[:, j]
            prob += counter_pair_1*(counter_pair_1 - 1)/2/total
            prob += counter_pair_2*(counter_pair_2 - 1)/2/total
            output.append(prob - probs[i]*probs[j])
        return np.sum(output)
    
    with parallel_backend('loky', n_jobs=10):
        covariances = Parallel()(delayed(get_covariance)(i) for i in range(len(arr[0])))
    covariance_sum = np.sum(covariances)
    #print(covariance_sum)

    return np.sqrt(np.sum(probs*(1-probs)) + 2*covariance_sum)/len(arr[0])

def stratified_sigma(fps, n, n_ary):
    # Sample the 
    indexes_strat = stratified_sampling(fps, n_ary = n_ary, percentage = n*100/len(fps))
    fps_strat = fps[indexes_strat]

    # Calculate the pairwise average of the sampled indexes
    average, std = pairwise_average(fps_strat, n_ary = n_ary)

    return std

def random_sigma(fps, n, n_ary):
    indexes_rand = np.random.choice(len(fps), n, replace = False)
    fps_rand = fps[indexes_rand]

    average, std = pairwise_average(fps_rand, n_ary = n_ary)

    return std