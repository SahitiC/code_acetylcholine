#importing required packages
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import special 
from scipy.stats import norm

 #%%
def lognormalCDF(x,mu,sigma):
    a = (np.log(x) - mu)/(np.sqrt(2)*sigma)
    cdf = 0.5 + 0.5*(special.erf(a))
    return cdf

def lognormalX(cdf, mu, sigma):
    c = np.sqrt(2)*sigma*special.erfinv(2*cdf - 1) + mu
    X = np.exp(c)
    return X

#%%
#generative model + overall likelihood inference
hit = 0; fa = 0; miss = 0; cr = 0
n = 10 #no. of samples
r = 1; #length of signal
mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution

trial_signal = np.random.binomial(1,0.5) #trial is signal/non signal with prob 0.5
start_signal = np.random.randint(0, n-r+1) #index of sample when signal begins (on signal trial)

trial = np.full((int(round(n)),1),0);#signal values within a trial
if trial_signal == 1: 
    trial[start_signal:start_signal+r] = np.full((int((r)),1),1)

X = np.full((int(round(n)),1),np.nan)
for i in range(len(X)):
    if trial[i] == 0:
        X[i] = np.random.normal(mu_n,sigma_n)
    elif trial[i] == 1:
        X[i] = np.random.normal(mu_s,sigma_s)
        
#inference and response
likelihood = sum(np.exp(mu_s*X/(sigma_s**2)))*np.exp(-mu_s**2/(2*sigma_s**2))*(1/len(X))

response = 2; 
if likelihood > 1:
    response = 1; 
    if trial_signal == 1: hit = hit+1
    elif trial_signal == 0: fa = fa+1
elif likelihood < 1: 
    response = 0
    if trial_signal == 1: miss = miss+1
    elif trial_signal == 0: cr = cr+1    

#%%   
#generative model & overall likelihood inference - multiple trials

nArr = np.arange(1,1000,10) #array of trial lengths 
nTrials = 2000 #no. of trials for each ITU
trialTypesSimulated = np.full((len(nArr),5),0.0) #counting rates of h/m/f/c for each trial length
mu_s = 0.2; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution


start = time.perf_counter()
for y in range(len(nArr)):
    hit = 0; miss = 0; cr = 0; fa = 0
    for z in range(nTrials):
        n = nArr[y]; #no. of samples (trial length)
        r = 1; #length of signal
        trial_signal = np.random.binomial(1,0.5) #trial is signal/non signal with prob 0.5
        start_signal = np.random.randint(0, n-r+1) #index of sample when signal begins (on signal trial)
        
        trial = np.full((int(round(n)),1),0);#signal values within a trial
        if trial_signal == 1: 
            trial[start_signal:start_signal+r] = np.full((int((r)),1),1)
        
        X = np.full((int(round(n)),1),np.nan)
        for i in range(len(X)):
            if trial[i] == 0:
                X[i] = np.random.normal(mu_n,sigma_n)
            elif trial[i] == 1:
                X[i] = np.random.normal(mu_s,sigma_s)
                
        #inference and response
        likelihood = sum(np.exp(mu_s*X/(sigma_s**2)))*np.exp(-mu_s**2/(2*sigma_s**2))*(1/len(X))

        response = 2; 
        if likelihood > 1:
            response = 1; 
            if trial_signal == 1: hit = hit+1
            elif trial_signal == 0: fa = fa+1
        elif likelihood < 1: 
            response = 0
            if trial_signal == 1: miss = miss+1
            elif trial_signal == 0: cr = cr+1  
        
    trialTypesSimulated[y,0] = hit; trialTypesSimulated[y,1] = miss;
    trialTypesSimulated[y,2] = cr; trialTypesSimulated[y,3] = fa;
    
trialTypesSimulated[:,4] = norm.ppf(trialTypesSimulated[:,0]/(
    trialTypesSimulated[:,0]+trialTypesSimulated[:,1]))-norm.ppf(
        trialTypesSimulated[:,3]/(trialTypesSimulated[:,2]+trialTypesSimulated[:,3]))
print(time.perf_counter()-start)

#%%
#calculating critical value for the sequential case:
nTrials = 2000 #no. of trials for each ITU length
#nArr = np.arange(1,1000,10)
nArr = [1000] #length of interval (ITU)
r = 1; #length of signal
mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution
vHist = np.full((nTrials,1),0.0) #array of v's to calculate histogram
thresholdArr = np.full((len(nArr),1),0.0) #array of thresholds for each value of n

start = time.perf_counter()
for na in range(len(nArr)):
    n = nArr[na]; #no. of samples
    
    #under H0:
    for nt in range(nTrials):
        trial_signal = 0 #non-signal trial for H0
        start_signal = np.random.randint(0, n-r+1) #index of sample when signal begins (on signal trial)
        
        trial = np.full((int(round(n)),1),0);#signal values within a trial
        if trial_signal == 1: 
            trial[start_signal:start_signal+r] = np.full((int((r)),1),1)
                    
        X = np.full((int(round(n)),1),np.nan) #observations
        for i in range(len(X)):
            if trial[i] == 0:
                X[i] = np.random.normal(mu_n,sigma_n)
            elif trial[i] == 1:
                X[i] = np.random.normal(mu_s,sigma_s)
                
        #inference and response
        s = np.full((len(X)+1,1),0.0)
        h = np.full((len(X)+1,1),0.0)
        v = np.full((len(X)+1,1),0.0)
        v[0] = 1
        for j in range(len(X)):
            c = np.exp(mu_s**2/(2*sigma_s**2))
            s[j+1] = s[j] + np.exp((mu_s*X[j])/(sigma_s**2))
            h[j+1] = (s[j]+(n-j+1)*c)/(s[j+1] + (n-j)*c)
            v[j+1] = min(h[j+1], v[j])
        
        vHist[nt,0] = v[len(X)]
    
    thresholdArr[na,0] = np.quantile(vHist,0.05)

print(time.perf_counter()-start)

#%%
#calculating critical value for the sequential case:
nTrials = 2000 #no. of trials for each ITU length
#nArr = np.arange(1,1000,10)
nArr = [1000] #length of interval (ITU)
r = 1; #length of signal
mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution
vHist = np.full((nTrials,1),0.0) #array of v's to calculate histogram
thresholdArr = np.full((len(nArr),1),0.0) #array of thresholds for each value of n

start = time.perf_counter()
for na in range(len(nArr)):
    n = nArr[na]; #no. of samples
    
    #under H0:
    for nt in range(nTrials):
        trial_signal = 0 #non-signal trial for H0
        start_signal = np.random.randint(0, n-r+1) #index of sample when signal begins (on signal trial)
        
        trial = np.full((int(round(n)),1),0);#signal values within a trial
        if trial_signal == 1: 
            trial[start_signal:start_signal+r] = np.full((int((r)),1),1)
                    
        X = np.full((int(round(n)),1),np.nan) #observations
        for i in range(len(X)):
            if trial[i] == 0:
                X[i] = np.random.normal(mu_n,sigma_n)
            elif trial[i] == 1:
                X[i] = np.random.normal(mu_s,sigma_s)
                
        #inference and response
        s = np.full((len(X)+1,1),0.0)
        h = np.full((len(X)+1,1),0.0)
        v = np.full((len(X)+1,1),0.0)
        v[0] = 1
        for j in range(len(X)):
            c = np.exp(mu_s**2/(2*sigma_s**2))
            s[j+1] = s[j] + np.exp((mu_s*X[j])/(sigma_s**2))
            h[j+1] = (s[j]+(n-j+1)*c)/(s[j+1] + (n-j)*c)
            v[j+1] = min(h[j+1], v[j])
        
        vHist[nt,0] = v[len(X)]
    
    thresholdArr[na,0] = np.quantile(vHist,0.05)

print(time.perf_counter()-start)
 

#%%

#generative model - sequential inference

nArr = np.arange(1,1000,10) #array of trial lengths 
#nArr = [10]
nTrials = 2000 #no. of trials for each ITU
trialTypesSimulated = np.full((len(nArr),5),0.0) #counting rates of h/m/f/c for each trial length
mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution


start = time.perf_counter()
for y in range(len(nArr)):
    threshold_v = thresholdArr[y]
    hit = 0; miss = 0; cr = 0; fa = 0
    for z in range(nTrials):    
        n = nArr[y]; #no. of samples (trial length)
        r = 1; #length of signal
        
        trial_signal = np.random.binomial(1,0.5) #trial is signal/non signal with prob 0.5
        start_signal = np.random.randint(0, n-r+1) #index of sample when signal begins (on signal trial)
        
        trial = np.full((int(round(n)),1),0);#signal values within a trial
        if trial_signal == 1: 
            trial[start_signal:start_signal+r] = np.full((int((r)),1),1)
        
        X = np.full((int(round(n)),1),np.nan)
        for i in range(len(X)):
            if trial[i] == 0:
                X[i] = np.random.normal(mu_n,sigma_n)
            elif trial[i] == 1:
                X[i] = np.random.normal(mu_s,sigma_s)
                
        #inference and response
        s = np.full((len(X)+1,1),0.0)
        h = np.full((len(X)+1,1),0.0)
        v = np.full((len(X)+1,1),0.0)
        v[0] = 1
        for j in range(len(X)):
            c = np.exp(mu_s**2/(2*sigma_s**2))
            s[j+1] = s[j] + np.exp(mu_s*X[j]/(sigma_s**2))
            h[j+1] = (s[j]+(n-j+1)*c)/(s[j+1] + (n-j)*c)
            v[j+1] = min(h[j+1], v[j])
            
        
        response = 2; 
        if v[len(X)] < threshold_v:
            response = 1; 
            if trial_signal == 1: hit = hit+1
            elif trial_signal == 0: fa = fa+1
        elif v[len(X)] > threshold_v: 
            response = 0
            if trial_signal == 1: miss = miss+1
            elif trial_signal == 0: cr = cr+1  
            
    trialTypesSimulated[y,0] = hit; trialTypesSimulated[y,1] = miss;
    trialTypesSimulated[y,2] = cr; trialTypesSimulated[y,3] = fa;
    
trialTypesSimulated[:,4] = norm.ppf(trialTypesSimulated[:,0]/(
    trialTypesSimulated[:,0]+trialTypesSimulated[:,1]))-norm.ppf(
        trialTypesSimulated[:,3]/(trialTypesSimulated[:,2]+trialTypesSimulated[:,3]))

print(time.perf_counter()-start)
#%%
#expected hit and fa rates -- approximation

#nArrExp = np.arange(1,1000,1) #array of trial lengths 
nArrExp = [1]
trialTypesExpected = np.full((len(nArrExp),5),0.0)
mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution

mu_sp = (mu_s**2)/(sigma_s**2); sigma_sp = (mu_s)/(sigma_s) 
mu_np = (mu_s*mu_n)/(sigma_n**2); sigma_np = (mu_s)/(sigma_n) 

for t in range(len(nArrExp)):
    n = nArrExp[t]
    r = 1 #length of signal
    
    exp_i0 = np.exp(0.5*sigma_np**2); #expected value of Y_i under H0
    var_i0 = np.exp(2*sigma_np**2) - np.exp(sigma_np**2); #variance of Y_i under H0
    exp_i1 = (np.exp(mu_sp + 0.5*sigma_sp**2) + (n-1)*np.exp(0.5*sigma_np**2))/n #expected value of Y_i under H1
    var_i1 = (1/n)*(np.exp(2*mu_sp+2*sigma_sp**2)+(n-1)*np.exp(2*sigma_np**2)) - exp_i1**2 #variance of Y_i under H1
    
    #finding params = mu_i and var_i : for each Yi
    mu_i0 = np.log((exp_i0**2)/(((exp_i0**2)+var_i0)**0.5)) #mu param under h0
    mu_i1 = np.log((exp_i1**2)/(((exp_i1**2)+var_i1)**0.5)) #mu param under h1
    sig2_i0 = np.log(1 + (var_i0/(exp_i0**2))) #sigma param under h0
    sig2_i1 = np.log(1 + (var_i1/(exp_i1**2))) #sigma param under h1
    
    #mu_z and sig_z for Z = sum of Yi
    z0 = (np.exp(2*mu_i0+sig2_i0)*(np.exp(sig2_i0)-1))/(n*np.exp(mu_i0+0.5*sig2_i0)**2)
    sig2_z0 = np.log(z0+1)
    z1 = (np.exp(2*mu_i1+sig2_i1)*(np.exp(sig2_i1)-1))/(n*np.exp(mu_i1+0.5*sig2_i1)**2)
    sig2_z1 = np.log(z1+1)    
    mu_z0 = np.log(n*np.exp(mu_i0+0.5*sig2_i0)) - sig2_z0/2
    mu_z1 = np.log(n*np.exp(mu_i1+0.5*sig2_i1)) - sig2_z1/2
    
    threshold = (n)*np.exp((mu_s**2)/(2*sigma_s**2)) #to compare sumY'is with
    trialTypesExpected[t,0] = 1-lognormalCDF(threshold,mu_z1,np.sqrt(sig2_z1)) #h
    trialTypesExpected[t,1] = lognormalCDF(threshold,mu_z1,np.sqrt(sig2_z1)) #m
    trialTypesExpected[t,2] = 1-lognormalCDF(threshold,mu_z0,np.sqrt(sig2_z0)) #fa
    trialTypesExpected[t,3] = lognormalCDF(threshold,mu_z0,np.sqrt(sig2_z0)) #cr
    trialTypesExpected[t,4] = norm.ppf(trialTypesExpected[t,0])-norm.ppf(trialTypesExpected[t,2])

#%%
plt.plot(nArr, trialTypesSimulated[:,0]/(trialTypesSimulated[:,0]+trialTypesSimulated[:,1]), 
         label ='hits Simulated')
plt.plot(nArr, trialTypesSimulated[:,3]/(trialTypesSimulated[:,2]+trialTypesSimulated[:,3]), 
         label ='fa Simulated')
#plt.plot(nArrExp, trialTypesExpected[:,0], label ='hits approx')
#plt.plot(nArrExp, trialTypesExpected[:,2], label ='fa approx')
plt.legend(); plt.xlabel('number of samples'); plt.title('mu_signal=%1.2f'%mu_s)

#%%
plt.plot(nArr, trialTypesSimulated[:,4], label ='simulated')
plt.plot(nArrExp, trialTypesExpected[:,4], label ='expected')
plt.legend(); plt.xlabel('number of samples'); plt.ylabel('d-prime'); 
plt.title('mu_signal=%1.2f'%mu_s)

#%%
#hit and fa rates for range of mu's

nArrExp = np.arange(1,1000,1) #array of trial lengths 
#nArrExp = [10]
mu_sArr = [0.1, 0.5,1,2,3,4,6]
trialTypesExpected = np.full((len(nArrExp),5),0.0)
mu_s = 4; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution
fig1, ax1 = plt.subplots(); fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

for m in range(len(mu_sArr)):
    mu_s = mu_sArr[m]
    for t in range(len(nArrExp)):
        n = nArrExp[t]
        r = 1 #length of signal
        
        exp_i0 = np.exp(0.5); #expected value of Y_i under H0
        var_i0 = np.exp(2) - np.exp(1); #variance of Y_i under H0
        exp_i1 = (np.exp(mu_s + 0.5) + (n-1)*np.exp(0.5))/n #expected value of Y_i under H1
        var_i1 = (1/n)*(np.exp(2*mu_s+2)+(n-1)*np.exp(2)) - exp_i1**2 #variance of Y_i under H1
        
        #finding params = mu_i and var_i
        mu_i0 = np.log((exp_i0**2)/(((exp_i0**2)+var_i0)**0.5)) #mu param under h0
        mu_i1 = np.log((exp_i1**2)/(((exp_i1**2)+var_i1)**0.5)) #mu param under h1
        sig2_i0 = np.log(1 + (var_i0/(exp_i0**2))) #sigma param under h0
        sig2_i1 = np.log(1 + (var_i1/(exp_i1**2))) #sigma param under h1
        
        z0 = (np.exp(2*mu_i0+sig2_i0)*(np.exp(sig2_i0)-1))/(n*np.exp(mu_i0+0.5*sig2_i0)**2)
        sig2_z0 = np.log(z0+1)
        z1 = (np.exp(2*mu_i1+sig2_i1)*(np.exp(sig2_i1)-1))/(n*np.exp(mu_i1+0.5*sig2_i1)**2)
        sig2_z1 = np.log(z1+1)    
        mu_z0 = np.log(n*np.exp(mu_i0+0.5*sig2_i0)) - sig2_z0/2
        mu_z1 = np.log(n*np.exp(mu_i1+0.5*sig2_i1)) - sig2_z1/2
        
        
        threshold = (n-r+1)*np.exp(r/2) #to compare sumY'is with
        trialTypesExpected[t,0] = 1-lognormalCDF(threshold,mu_z1,np.sqrt(sig2_z1)) #h
        trialTypesExpected[t,1] = lognormalCDF(threshold,mu_z1,np.sqrt(sig2_z1)) #m
        trialTypesExpected[t,2] = 1-lognormalCDF(threshold,mu_z0,np.sqrt(sig2_z0)) #fa
        trialTypesExpected[t,3] = lognormalCDF(threshold,mu_z0,np.sqrt(sig2_z0)) #cr
        trialTypesExpected[t,4] = norm.ppf(trialTypesExpected[t,0])-norm.ppf(trialTypesExpected[t,2])
    
    ax1.plot(nArrExp, trialTypesExpected[:,4], label = 'mu=%1.2f'%mu_s); 
    ax1.set_ylabel('d-prime'); ax1.legend()
    ax2.plot(nArrExp, trialTypesExpected[:,0], label = 'mu=%1.2f'%mu_s)
    ax2.set_ylabel('hit rate'); ax2.legend()
    ax3.plot(nArrExp, trialTypesExpected[:,2], label = 'mu=%1.2f'%mu_s)
    ax3.set_ylabel('fa rate'); ax3.legend()


#%%
# function for generating trials, inference and resposne for the case when there
# is a single change point and multiple observations, fixed trial length
def trial_singleChangePoint(n,mu_s, sigma_s, mu_n, sigma_n,p_signal,trial_type, signal_start_type, signal_length):
    """
    Returns the generated trial observations, likelihood measures for the case when the trial length is fixed and a single fixed
    point marks teh shift in distribution: the change point can occur anywhere in the interval
    
    Args:
        n (scalar): number of observations in the trial 
        mu_s (scalar): mean of signal distribution
        sigma_s (scalar): standard deviation of signal distribution
        mu_n (scalar): mean of non-signal distribution
        sigma_n (scalar): standard deviation of non-signal distribution
        p_signal (scalar): probability of the trial being a signal trial
        trial_type (scalar): whether trial is a non-signal (0), signal(1) or one of them with probability p_signal (2)
        signal_start_type (scalar): whether signal start point is fixed (0) or random from a uniform distribution (1)
        signal_length (scalar): length of signal (integer <= n) if signal start point is fixed or if not applicable, is 0
        
    Returns:
        (scalar): trial type (signal/ non-signal)
        (array): array of underlying signal in the trial
        (array): array of observations
        (array): ratio of likelihoods under the two hypotheses at each time point
        (array): most discrepant cumulative Bayes factor of recent past at each time point
        (array): run length at each time point

    """
    
    if trial_type == 0: trial_signal = 0 #only non-signal trials
    elif trial_type == 1: trial_signal = 1 #only signal trials
    elif trial_type == 2: trial_signal = np.random.binomial(1,0.5)  #trial is signal/non signal with prob 0.5 
    
    if signal_start_type == 0: start_signal = np.random.randint(0, n) #index of sample when signal begins (on signal trial) 
    elif signal_start_type == 1: start_signal =  n-signal_length #index of sample when signal begins (on signal trial)
    

    trial = np.full((int(round(n)),1),0) #signal values within a trial
    if trial_signal == 1: 
        trial[start_signal:] = np.full((n-start_signal,1),1)

    observations = np.full((int(round(n)),1),np.nan)
    for x in range(len(observations)):
        if trial[x] == 0:
            observations[x] = np.random.normal(mu_n,sigma_n)
        elif trial[x] == 1:
            observations[x] = np.random.normal(mu_s,sigma_s)

    #inference
    w = np.full((len(observations)+1,1),0.0) #likelihood variable
    v = np.full((len(observations)+1,1),0.0) #max discrepant likelihood in recent past variable
    r = np.full((len(observations)+1,1),0.0) #run-length
    w[0] = 1; v[0] = 1 #initialise
    for j in range(len(observations)):
        c = 0;
        for i in range(j+1):
            f = np.product(np.exp(mu_s*observations[i:j+1]/(sigma_s**2))) * np.exp((-j+i-1)*(mu_s**2)/(2*sigma_s**2))
            c = c+f
        w[j+1] = (c + (n-j-1))/n
        v[j+1] = w[j+1]*np.max([1,v[j]])/w[j]
        if v[j] > 1: r[j+1] = r[j]+1
        elif v[j] <= 1: r[j+1] = 1 

    return trial_signal, trial, observations, w,v,r

def response_singleChangePoint(trial_signal, trial, observations,statistic,threshold):
    """
        trial_signal: trial type (signal/ non-signal)
        trial: array of underlying signal in the trial
        observations: array of observations
        statistic: variable based on which decision should be made
        threshold: threshold on the test statistic
        
        (scalar): choice based on decision rule
        (scalar): whether the trial is a hit
        (scalar): whether the trial is a miss
        (scalar): whether the trial is a cr
        (scalar): whether the trial is a fa
    """
    response = 2;
    hit = 0; miss = 0; cr = 0; fa = 0
    if statistic[len(observations)] > threshold:
        response = 1; 
        if trial_signal == 1: hit = hit+1
        elif trial_signal == 0: fa = fa+1
    elif statistic[len(observations)] < threshold: 
        response = 0
        if trial_signal == 1: miss = miss+1
        elif trial_signal == 0: cr = cr+1
            
    return response, hit, miss, cr, fa

#%%

#finding critical value (threshold) for the statistic : sequential inference 1 change point and multiple observations

nArr = [1,5,10,25,50,75,100] #array of trial lengths
nTrials = 2000 #no. of trials for each trial length
mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution
p_signal = 0.5 # probability of a signal trial

vHist = np.full((nTrials,1),0.0) #histogram of the statistic v
wHist = np.full((nTrials,1),0.0) #histogram of the statistic w
thresholdArr = np.full((len(nArr),1),0.0) #array of thresholds for each value of n
thresholdArrW = np.full((len(nArr),1),0.0) #array of thresholds for each value of n

trial_type = 0 # only non-signal trials to get distribution under H0
signal_start_type = 0 #variable signal lengths
signal_length = 0 #unused variable

start = time.perf_counter()
for y in range(len(nArr)):

    for z in range(nTrials):    
        n = nArr[y]; #no. of samples (trial length)
        
        trial_signal, trial, observations, w,v,r = trial_singleChangePoint(n,mu_s, sigma_s, mu_n, 
                                    sigma_n,p_signal,trial_type, signal_start_type, signal_length)
                
        
        vHist[z,0] = v[len(observations)]
        wHist[z,0] = w[len(observations)]        
    
    thresholdArr[y,0] = np.quantile(vHist,0.95)
    thresholdArrW[y,0] = np.quantile(wHist,0.95)
  
print(time.perf_counter()-start)

#%%
#example run of a trial
mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution
n = 10 #no. of samples
p_signal = 0.5 #prob of signal trial
trial_type = 2; signal_start_type = 0; signal_length = 5
trial_signal, trial, observations, w,v,r = trial_singleChangePoint(n,mu_s, sigma_s, mu_n, 
                                    sigma_n,p_signal,trial_type, signal_start_type, signal_length)
response, hit, miss, cr, fa = response_singleChangePoint(trial_signal,trial, observations,w, thresholdArrW[2,0])

#plot of result
plt.plot(trial, label = 'underlying signal'); plt.plot(observations, label = 'observations')
plt.legend(); plt.figure()
plt.plot(w[1:], label = 'likelihood ratio'); plt.legend()
print('response=%d,hit=%d,miss=%d,cr=%d,fa=%d'%(response,hit,miss,cr,fa))

#%%
#generative model, sequential inference : 1 change point and multiple observations
#looking at hit rates across trial lengths without controlling for length of signal

#nArr = np.arange(1,1000,10) #array of trial lengths 
#[1,5,10,25,50,75,100]
nArr = [1,5,10,25,50,75,100] #array of trial lengths
nTrials = 2000 #no. of trials for each trial lengths

mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution

trial_type = 2 # only non-signal trials to get distribution under H0
signal_start_type = 0 #variable signal lengths
signal_length = 0 #unused variable

trialTypesSimulated = np.full((len(nArr),5),0.0) #counting rates of h/m/f/c for each trial length $!

start = time.perf_counter()
for y in range(len(nArr)): 
    threshold_w = thresholdArrW[y] #!!
    hit = 0; miss = 0; cr = 0; fa = 0
    for z in range(nTrials):    
        n = nArr[y]; #no. of samples (trial length) 
                
        trial_signal, trial, observations, w,v,r = trial_singleChangePoint(n,mu_s, sigma_s, mu_n, 
                                    sigma_n,p_signal,trial_type, signal_start_type, signal_length)
 
        repsonse, hit0, miss0, cr0, fa0 = response_singleChangePoint(trial_signal,trial, observations,w, threshold_w)
    
        hit = hit+hit0; miss = miss+miss0; cr = cr+cr0; fa = fa+fa0
            
    trialTypesSimulated[y,0] = hit; trialTypesSimulated[y,1] = miss;
    trialTypesSimulated[y,2] = cr; trialTypesSimulated[y,3] = fa;
    
trialTypesSimulated[:,4] = norm.ppf(trialTypesSimulated[:,0]/(
    trialTypesSimulated[:,0]+trialTypesSimulated[:,1]))-norm.ppf(
        trialTypesSimulated[:,3]/(trialTypesSimulated[:,2]+trialTypesSimulated[:,3]))

print(time.perf_counter()-start)

#plotting hit rates for different ITU lengths
plt.plot(nArr, trialTypesSimulated[:,0]/(trialTypesSimulated[:,0]+trialTypesSimulated[:,1]), label = 'hit rate', marker = 'o')
plt.plot(nArr, trialTypesSimulated[:,3]/(trialTypesSimulated[:,2]+trialTypesSimulated[:,3]), label = 'fa rate', marker = 'o')
plt.plot(nArr, trialTypesSimulated[:,4], label = 'd-prime', marker = 'o')
plt.xlabel('ITU length'); plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


#%%
#generative model, sequential inference : 1 change point and multiple observations
#looking at hitrates across isgnal durations

signal_lengthArr =[1,3,5,10,15,25,35,50,75] #signal length [1,3,5,10,15,25,35,50,75,100]
nArr = [1,5,10,25,50] #array of trial lengths 1,5,10,25,50,75,100
nTrials = 2000 #no. of trials for each trial lengths

mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution

trial_type = 2 # signal and non-signal trials
signal_start_type = 1 #constant signal lengths

trialTypesSimulatedV = np.full((len(nArr),len(signal_lengthArr),5),0.0) #counting rates of h/m/f/c for each trial length 
trialTypesSimulatedW = np.full((len(nArr),len(signal_lengthArr),5),0.0) #counting rates of h/m/f/c for each trial length 
trialFinalW = np.full((len(nArr),len(signal_lengthArr),4),np.nan) #final value of W for each n, r, for hits, cr, miss, fa

start = time.perf_counter()

for l in range(len(nArr)):
    n = nArr[l] #lenght of trial
    threshold_v = thresholdArr[l] #!
    threshold_w = thresholdArrW[l] #!
    y = 0;
    while signal_lengthArr[y] <= n: 
        signal_length = signal_lengthArr[y]
        hitV = 0; missV = 0; crV = 0; faV = 0
        hitW = 0; missW = 0; crW = 0; faW = 0
        W_n = np.full((1,4),0.0)
        for z in range(nTrials):    

            trial_signal,trial, observations, w,v,r = trial_singleChangePoint(n,mu_s, sigma_s, mu_n, 
                                        sigma_n,p_signal,trial_type, signal_start_type, signal_length)

            responseV, hitv0, missv0, crv0, fav0 = response_singleChangePoint(trial_signal,trial, observations,v,threshold_v)
            responseW, hitw0, missw0, crw0, faw0 = response_singleChangePoint(trial_signal,trial, observations,w,threshold_w)

            hitV = hitV+hitv0; missV = missV+missv0; crV = crV+crv0; faV = faV+fav0
            hitW = hitW+hitw0; missW = missW+missw0; crW = crW+crw0; faW = faW+faw0
            
            if hitw0 == 1: W_n[0,0] = W_n[0,0]+w[n]
            elif missw0 == 1: W_n[0,1] = W_n[0,1]+w[n]
            elif crw0 == 1: W_n[0,2] = W_n[0,2]+w[n]
            elif faw0 == 1: W_n[0,3] = W_n[0,3]+w[n]
            
        trialTypesSimulatedV[l,y,0] = hitV; trialTypesSimulatedV[l,y,1] = missV;
        trialTypesSimulatedV[l,y,2] = crV; trialTypesSimulatedV[l,y,3] = faV;
        trialTypesSimulatedW[l,y,0] = hitW; trialTypesSimulatedW[l,y,1] = missW;
        trialTypesSimulatedW[l,y,2] = crW; trialTypesSimulatedW[l,y,3] = faW;
        trialFinalW[l,y,0] = W_n[0,0]/trialTypesSimulatedW[l,y,0]; trialFinalW[l,y,1] = W_n[0,1]/trialTypesSimulatedW[l,y,1]
        trialFinalW[l,y,2] = W_n[0,2]/trialTypesSimulatedW[l,y,2]; trialFinalW[l,y,3] = W_n[0,3]/trialTypesSimulatedW[l,y,3]
        
        y = y+1

print(time.perf_counter()-start)

#%%
#plotting hit rates vs signal lengths for different ITUs but similar signal durations
for l in range(len(nArr)):

    a = trialTypesSimulatedV[l,:,0]/(trialTypesSimulatedV[l,:,0]+trialTypesSimulatedV[l,:,1])
    plt.plot(signal_lengthArr[:len(trialTypesSimulatedV[l,:,:])],a, marker = 'o', label = 'n=%d'%nArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('signal duration')
plt.ylabel('hit rates'); plt.title('based on V_j')

plt.figure()

#plotting hit rates vs signal lengths for different ITUs but similar signal durations
for l in range(len(nArr)):

    a = trialTypesSimulatedW[l,:,0]/(trialTypesSimulatedW[l,:,0]+trialTypesSimulatedW[l,:,1])
    plt.plot(signal_lengthArr[:len(trialTypesSimulatedW[l,:,:])],a, marker = 'o', label = 'n=%d'%nArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('signal duration')
plt.ylabel('hit rates'); plt.title('based on W_j'); plt.figure()

#plotting avg W's for hits
for s in range(len(signal_lengthArr)-3):

    plt.errorbar(nArr,trialFinalW[:,s,0], marker = 'o',  label = 'signal=%d'%signal_lengthArr[s])
    
plt.plot(nArr,thresholdArrW[:5]); 
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('length of trial')
plt.ylabel('avg W_n on hit trials'); 

#%%
#fixed r - sequential inference
def trial_fixedSignal(trial_length,mu_s, sigma_s, mu_n, sigma_n,p_signal,trial_type, signal_length):
    """
    Returns the generated trial observations for the case when trial length is fixed and so is the size of
    the signal interval: the signal can occur anywhere in the trial
    
    Args:
        trial_length (scalar): number of observations in the trial 
        mu_s (scalar): mean of signal distribution
        sigma_s (scalar): standard deviation of signal distribution
        mu_n (scalar): mean of non-signal distribution
        sigma_n (scalar): standard deviation of non-signal distribution
        p_signal (scalar): probability of the trial being a signal trial
        trial_type (scalar): whether trial is a non-signal (0), signal(1) or one of them with probability p_signal (2)
        signal_length (scalar): length of signal (integer <= n) if signal start point is fixed or if not applicable, is 0
        
    Returns:
        (scalar): trial type (signal/ non-signal)
        (array): array of underlying signal in the trial
        (array): array of observations

    """
    
    if trial_type == 0: trial_signal = 0 #only non-signal trials
    elif trial_type == 1: trial_signal = 1 #only signal trials
    elif trial_type == 2: trial_signal = np.random.binomial(1,0.5)  #trial is signal/non signal with prob 0.5 
    
    start_signal = np.random.randint(0, trial_length-signal_length+1) #index of sample when signal begins (on signal trial) 
    
    trial = np.full((int(round(trial_length)),1),0) #signal values within a trial
    if trial_signal == 1: 
        trial[start_signal:start_signal+signal_length]  = np.full((
            signal_length,1),1)
    
    observations = np.full((int(round(trial_length)),1),np.nan)
    for x in range(len(observations)):
        if trial[x] == 0:
            observations[x] = np.random.normal(mu_n,sigma_n)
        elif trial[x] == 1:
            observations[x] = np.random.normal(mu_s,sigma_s)
            
    return trial_signal, trial, observations


def inference_fixedSignal(trial_length,mu_s, sigma_s, mu_n, sigma_n,
                    p_signal,trial_type,signal_length, trial, observations):
    
    """
    Returns the calculated likelihoods (given the observations and parameters) for the case when trial length is fixed 
    and so is the size of the signal interval: the signal can occur anywhere in the trial
    
    Args:
        trial_length (scalar): number of observations in the trial 
        mu_s (scalar): mean of signal distribution
        sigma_s (scalar): standard deviation of signal distribution
        mu_n (scalar): mean of non-signal distribution
        sigma_n (scalar): standard deviation of non-signal distribution
        p_signal (scalar): probability of the trial being a signal trial
        trial_type (scalar): whether trial is a non-signal (0), signal(1) or one of them with probability p_signal (2)
        signal_length (scalar): length of signal (integer <= n) if signal start point is fixed or if not applicable, is 0
        trial (array): array of underlying signal in the trial
        observations (array): array of observations
        
    Returns:
    
        (array): ratio of likelihoods under the two hypotheses at each time point
        (array): most discrepant cumulative Bayes factor of group of (signal length) observations at each time point
        
     """
                      
    w = np.full((len(observations)+1,1),0.0) #likelihood variable
    v = np.full((len(observations)+1,1),0.0) #max discrepant likelihood in recent past variable
    w[0] = 1; v[0] = 1;
    
    for j in range(len(observations)):
        c = 0
        if j <= (len(observations)-signal_length):
            
            for i in range(j+1):
                f = np.exp(np.sum(mu_s*observations[
                    i:min(i+signal_length-1,j)+1]/sigma_s**2) - ((min(
                        i+signal_length-1,j)-i+1)*(mu_s**2)/(2*sigma_s**2)))
                        
                c = c+f
            
            w[j+1] = (c + (len(observations)-signal_length - j))/(len(observations)-signal_length+1)
            
        elif j > (len(observations)-signal_length):
            
            for i in range(len(observations)-signal_length+1):
                f = np.exp(np.sum(mu_s*observations[
                    i:min(i+signal_length-1,j)+1]/sigma_s**2) - (min(
                        i+signal_length-1,j)-i+1)*(mu_s**2)/(2*sigma_s**2))
    
                c = c+f
                
            w[j+1] = c/(len(observations)-signal_length+1)
        
        c = 0
        for k in range(j+1):
            c = np.max([c, (w[min(k+signal_length,j+1)]/w[k])])
        
        v[j+1] = c
                                     
    return w,v

def onlineInference_fixedSignal(trial_signal, trial, observations):
    w = np.full((len(observations)+1,1),0.0) #likelihood variable
    v = np.full((len(observations)+1,1),0.0) #max discrepant likelihood in recent past variable
    w[0] = 1; v[0] = 1;
    A = np.exp(mu_s*observations/sigma_s**2 - mu_s**2/(2*sigma_s**2))
    B = np.full((len(observations)+1,1),0.0)
    C = np.full((len(observations)+1,1),0.0) 
    D = np.full((len(observations)+1,1),0.0)
    E = np.full((len(observations)+1,1),0.0)
    E[0] = np.exp(np.sum(A[0:signal_length-1]))
    B[0] = C[0]+D[0]
    
    for j in range(len(observations)):
        E[j+1] = E[j]*np.exp(A[max(j+1,signal_length)-1])
        C[j+1] = C[j] + E[j+1]
        
        c = 0
        for k in range(j+1):
            c = np.max([c, (w[min(k+signal_length,j+1)]/w[k])])
        
        v[j+1] = c
        
    return w,v

def response_fixedSignal(trial_signal, trial, observations,statistic,threshold):
    
    """
    Returns the response (given the decision variable and threshold) for the case when trial length is fixed 
    and so is the size of the signal interval (which can occur anywhere in the interval)

    Args:
        trial_signal (scalar): trial type (signal/ non-signal)
        trial (array): array of underlying signal in the trial
        observations (array): array of observations
        statistic (array): statistic/ decision variable calculated at each timepoint
        threshold (array): threshold against which statistic must be compared

    Returns:
        (scalar): response obtained based on the decision rule
        (scalar): whether the trial is a hit (0-no, 1-yes)
        (scalar): whether the trial is a miss (0-no, 1-yes)
        (scalar): whether the trial is a correct rejection (0-no, 1-yes)
        (scalar): whether the trial is a false alarm (0-no, 1-yes)    

    """
    
    response = 2;
    hit = 0; miss = 0; cr = 0; fa = 0
    if statistic[len(observations)] > threshold:
        response = 1; 
        if trial_signal == 1: hit = hit+1
        elif trial_signal == 0: fa = fa+1
    elif statistic[len(observations)] < threshold: 
        response = 0
        if trial_signal == 1: miss = miss+1
        elif trial_signal == 0: cr = cr+1

    return response, hit, miss, cr, fa

#%%
#single trial simulation
trial_length = 5
mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution
p_signal = 0.5; #probability of a signal trial
trial_type = 1; signal_length = 4

trial_signal, trial, observations = trial_fixedSignal(trial_length,mu_s, 
   sigma_s, mu_n, sigma_n,p_signal,trial_type,signal_length)

w,v = inference_fixedSignal(trial_length,mu_s, sigma_s, mu_n, sigma_n,
                    p_signal,trial_type,signal_length, trial, observations)

#plot of result
plt.plot(trial, label = 'underlying signal'); plt.plot(observations, label = 'observations')
plt.legend(); plt.figure()
plt.plot(w[1:], label = 'likelihood ratio'); plt.plot(v[1:], label = 'most discrepant')
plt.legend()

#%%
#finding critical value (threshold) for the statistic : sequential inference,
#fixed signal length
signal_lengthArr =[1,3,5,10,15,25,35,50,75] #signal length [1,3,5,10,15,25,35,50,75,100,101]
nArr = [5,10,25,50] #array of trial lengths  [5,10,25,50,75,100] 
nTrials = 2000 #no. of trials for each trial length
mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution
p_signal = 0.5 # probability of a signal trial
trial_type = 0

vHist = np.full((nTrials,1),0.0) #histogram of the statistic V
wHist = np.full((nTrials,1),0.0) #histogram of the statistic W
thresholdArr = np.full((len(nArr),len(signal_lengthArr)),0.0)
thresholdArrW = np.full((len(nArr),len(signal_lengthArr)),0.0)
#array of thresholds for each value of n

start = time.perf_counter()
for y in range(len(nArr)):
    trial_length = nArr[y]; #no. of samples (trial length)
    s= 0
    while signal_lengthArr[s] <= trial_length: 
        signal_length = signal_lengthArr[s]
        for z in range(nTrials):    
            
            trial_signal, trial, observations = trial_fixedSignal(trial_length,mu_s, 
            sigma_s, mu_n, sigma_n,p_signal,trial_type,signal_length)
            
            w,v = inference_fixedSignal(trial_length,mu_s, sigma_s, mu_n, sigma_n,
                    p_signal,trial_type,signal_length, trial, observations)
                    
            vHist[z,0] = v[len(observations)]
            wHist[z,0] = w[len(observations)]
        
        thresholdArr[y,s] = np.quantile(vHist,0.95)
        thresholdArrW[y,s] = np.quantile(wHist,0.95)
        s = s+1
  
print(time.perf_counter()-start)

#%%
#generative model, sequential inference : fixed signal duration
    
#nArr = np.arange(1,1000,10) #array of trial lengths 
#[1,5,10,25,50,75,100]
signal_lengthArr =[1,3,5,10,15,25,35,50,75] #signal length [1,3,5,10,15,25,35,50,75,100,101]
nArr = [5,10,25,50] #array of trial lengths [5,10,25,50,75]
nTrials = 2000 #no. of trials for each trial length
mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution
p_signal = 0.5 # probability of a signal trial
trial_type = 2

trialTypesSimulated = np.full((len(nArr),len(signal_lengthArr),5),0.0) #counting rates of h/m/f/c for each trial length $!
trialTypesSimulatedW = np.full((len(nArr),len(signal_lengthArr),5),0.0) #counting rates of h/m/f/c for each trial length $!
trialFinalW = np.full((len(nArr),len(signal_lengthArr),4),np.nan) #final value of W for each n, r, for hits, cr, miss, fa

start = time.perf_counter()
for y in range(len(nArr)):
    trial_length = nArr[y]; #no. of samples (trial length) 
    s=0
    while signal_lengthArr[s] <= trial_length: 
        signal_length = signal_lengthArr[s]        
        threshold_v = thresholdArr[y,s] #!
        threshold_w = thresholdArrW[y,s]
        hit = 0; miss = 0; cr = 0; fa = 0
        hitW = 0; missW = 0; crW = 0; faW = 0
        W_n = np.full((1,4),0.0)
        for z in range(nTrials):    
                    
            trial_signal, trial, observations = trial_fixedSignal(trial_length,mu_s, 
            sigma_s, mu_n, sigma_n,p_signal,trial_type, signal_length)
            
            w,v = inference_fixedSignal(trial_length,mu_s, sigma_s, mu_n, sigma_n,
                    p_signal,trial_type,signal_length, trial, observations)
     
            repsonse, hit0, miss0, cr0, fa0 = response_fixedSignal(trial_signal,
                                    trial, observations,v, threshold_v)
        
            repsonseW, hitW0,missW0, crW0, faW0 = response_fixedSignal(trial_signal,
                                    trial, observations,w, threshold_w)
        
            hit = hit+hit0; miss = miss+miss0; cr = cr+cr0; fa = fa+fa0
            hitW = hitW+hitW0; missW = missW+missW0; crW = crW+crW0; faW = faW+faW0
            
            if hitW0 == 1: W_n[0,0] = W_n[0,0]+w[trial_length]
            elif missW0 == 1: W_n[0,1] = W_n[0,1]+w[trial_length]
            elif crW0 == 1: W_n[0,2] = W_n[0,2]+w[trial_length]
            elif faW0 == 1: W_n[0,3] = W_n[0,3]+w[trial_length]
            
                
        trialTypesSimulated[y,s,0] = hit; trialTypesSimulated[y,s,1] = miss;
        trialTypesSimulated[y,s,2] = cr; trialTypesSimulated[y,s,3] = fa;
        
        trialTypesSimulatedW[y,s,0] = hitW; trialTypesSimulatedW[y,s,1] = missW;
        trialTypesSimulatedW[y,s,2] = crW; trialTypesSimulatedW[y,s,3] = faW;

        trialFinalW[y,s,0] = W_n[0,0]/trialTypesSimulatedW[y,s,0]; trialFinalW[y,s,1] = W_n[0,1]/trialTypesSimulatedW[y,s,1]
        trialFinalW[y,s,2] = W_n[0,2]/trialTypesSimulatedW[y,s,2]; trialFinalW[y,s,3] = W_n[0,3]/trialTypesSimulatedW[y,s,3]

        s = s+1
        
    trialTypesSimulated[y,:,4] = norm.ppf(trialTypesSimulated[y,:,0]/(
        trialTypesSimulated[y,:,0]+trialTypesSimulated[y,:,1]))-norm.ppf(
            trialTypesSimulated[y,:,3]/(trialTypesSimulated[y,:,2]+trialTypesSimulated[y,:,3]))

print(time.perf_counter()-start)

#%%
#plots of the results
for l in range(len(nArr)):

    a = trialTypesSimulated[l,:,0]/(trialTypesSimulated[l,:,0]+trialTypesSimulated[l,:,1])
    plt.plot(signal_lengthArr[:len(trialTypesSimulated[l,:,:])],a, marker = 'o', label = 'trial length=%d'%nArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('signal duration')
plt.ylabel('hit rates'); plt.title('using V_j'); plt.figure()

#plots of the results
for l in range(len(nArr)):

    a = trialTypesSimulatedW[l,:,0]/(trialTypesSimulatedW[l,:,0]+trialTypesSimulatedW[l,:,1])
    plt.plot(signal_lengthArr[:len(trialTypesSimulatedW[l,:,:])],a, marker = 'o', label = 'trial length=%d'%nArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('signal duration')
plt.ylabel('hit rates'); plt.title('using W_j'); plt.figure()

#plotting avg W's
for s in range(len(signal_lengthArr[:-5])):

    plt.plot(nArr,trialFinalW[:,s,0], marker = 'o',  label = 'signal=%d'%signal_lengthArr[s])
    plt.plot(nArr,thresholdArrW[:,s], linestyle = 'dashed', label = 'threshold for signal=%d'%signal_lengthArr[s])
    
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('length of trial')
plt.ylabel('avg W_n on hit trials'); 

#%%
#spare/ obsolete code
#generative model - sequential inference
    
hit = 0; fa = 0; miss = 0; cr = 0
n = 10; #no. of samples
r = 1; #length of signal
mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution
threshold_v = lognormalX(0.05,mu_n,sigma_n)

trial_signal = np.random.binomial(1,0.5) #trial is signal/non signal with prob 0.5
start_signal = np.random.randint(0, n-r+1) #index of sample when signal begins (on signal trial)

trial = np.full((int(round(n)),1),0);#signal values within a trial
if trial_signal == 1: 
    trial[start_signal:start_signal+r] = np.full((int((r)),1),1)

X = np.full((int(round(n)),1),np.nan)
for i in range(len(X)):
    if trial[i] == 0:
        X[i] = np.random.normal(mu_n,sigma_n)
    elif trial[i] == 1:
        X[i] = np.random.normal(mu_s,sigma_s)
        
#inference and response
h = np.full((len(X),1),0.0)
v = np.full((len(X)+1,1),0.0)
v_try = np.full((len(X),1),0.0)
v[0] = np.inf; 

for j in range(len(X)):
    c = np.exp(-mu_n**2/(2*sigma_n**2))/np.exp(-mu_s**2/(2*sigma_s**2))
    h[j] = c*np.exp(mu_n*X[j]/sigma_n**2)/np.exp(mu_s*X[j]/sigma_s**2)
    v[j+1] = min(h[j], v[j])
    v_try[j] = min(h[0:j+1])
    

