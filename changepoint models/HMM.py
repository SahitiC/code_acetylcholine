<<<<<<< HEAD

=======
#importing required packages
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm

#%%
#generating trial - using uniform and geometric processes
n = 10 #trial length
p_signal = 0.5 #prob of signal trial
mu_0 = 0; mu_1 = 1; mu_2 = 0 #means of gaussian for observations in states 0,1,2
sigma = 1 #standard deviation of Gaussian
q = 0.2 #constant probability of leaving

start = time.perf_counter()
nTrials = 2000
arr = np.full((nTrials,3),0) #signal trial or not, start point of signal, signal len
for k in range(nTrials):
    trial_signal = np.random.binomial(1,0.5) #trial is signal/non signal with prob 0.5
    start_signal = np.random.randint(0, n) #index of sample when signal begins (on signal trial)
    end_signal = np.random.geometric(q, size=1) #time points for which signal will stay on
    
    trial = np.full((int(round(n)),1),0);#state values within a trial
    if trial_signal == 1: 
        r = min(end_signal, n-start_signal)
        trial[start_signal:start_signal+int(r)] = np.full((int(r),1),1)
        trial[start_signal+int(r):n] = np.full((n-start_signal-int(r),1),2) 
    
    observations = np.full((int(round(n)),1),np.nan)
    for i in range(len(observations)):
        if trial[i] == 0:
            observations[i] = np.random.normal(mu_0,sigma)
        elif trial[i] == 1:
            observations[i] = np.random.normal(mu_1,sigma)
        elif trial[i] == 2:
            observations[i] = np.random.normal(mu_2,sigma)
        
    if trial_signal == 1:
        arr[k,0] = 1; 
        arr[k,1] = start_signal
        arr[k,2] = len(np.where(trial == 1)[0])
print(time.perf_counter()-start)  
        
#%%
#generating trial - using HMM logic
n = 10 #trial length
p_signal = 0.5 #prob of signal trial
mu_0 = 0; mu_1 = 1; mu_2 = 0 #means of gaussian for observations in states 0,1,2
sigma = 1 #standard deviation of Gaussian
q = 0.2 #constant probability of leaving
transition_matrix = np.array([[1,0,0],[0,1-q,q],[0,0,1]])
emission = np.array([mu_0,mu_1,mu_2])

start = time.perf_counter()
nTrials = 2000
arrHMM = np.full((nTrials,3),0) #signal trial or not, start point of signal, signal len
for k in range(nTrials):        
    trial = np.full((int(round(n))+1,1),0);#state values within a trial
    observation = np.full((int(round(n)),1),np.nan)
    trial_signal = np.random.binomial(1,0.5) #trial is signal/non signal with prob 0.5
    
    for i in range(n):
        probStart = 1/(n-i)
        transition_matrix = np.array([[1-probStart,probStart,0],[0,1-q,q],[0,0,1]])
        
        if trial_signal == 1:
            trial[i+1] = np.random.choice([0,1,2], p=transition_matrix[trial[i],:][0])
        
        observation[i] = np.random.normal(emission[trial[i+1]],sigma)
        
    if trial_signal == 1:
        arrHMM[k,0] = 1; 
        arrHMM[k,1] = np.intersect1d(np.where(trial[1:] == 1)[0], np.where(trial[:-1] == 0)[0])
        arrHMM[k,2] = len(np.where(trial == 1)[0])
print(time.perf_counter()-start)    
>>>>>>> 2084695b0cb3e10d8f14adf4e392e9a22ce79dbf
