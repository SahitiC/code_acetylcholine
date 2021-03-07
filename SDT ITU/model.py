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

#generative model, critical value for seq inference : 1 change point and multiple observations

#nArr = np.arange(1,1000,10) #array of trial lengths 
#[1,5,10,25,50,75,100]
nArr = [1,5,10,25,50,75,100]
nTrials = 2000 #no. of trials for each ITU
mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution
vHist = np.full((nTrials,1),0.0)
wHist = np.full((nTrials,1),0.0)
thresholdArr = np.full((len(nArr),1),0.0) #array of thresholds for each value of n
thresholdArrW = np.full((len(nArr),1),0.0) #array of thresholds for each value of n

start = time.perf_counter()
for y in range(len(nArr)):

    for z in range(nTrials):    
        n = nArr[y]; #no. of samples (trial length)
        
        trial_signal = 0 #under H0
        start_signal = np.random.randint(0, n) #index of sample when signal begins (on signal trial)
        
        trial = np.full((int(round(n)),1),0);#signal values within a trial
        if trial_signal == 1: 
            trial[start_signal:] = np.full((n-start_signal,1),1)
        
        X = np.full((int(round(n)),1),np.nan)
        for x in range(len(X)):
            if trial[x] == 0:
                X[x] = np.random.normal(mu_n,sigma_n)
            elif trial[x] == 1:
                X[x] = np.random.normal(mu_s,sigma_s)
                
        #inference and response
        w = np.full((len(X)+1,1),0.0)
        v = np.full((len(X)+1,1),0.0)
        w[0] = 1; v[0] = 1
        for j in range(len(X)):
            c = 0;
            for i in range(j+1):
                f = np.product(np.exp(mu_s*X[i:j+1]/(sigma_s**2))) * np.exp((-j+i-1)*(mu_s**2)/(2*sigma_s**2))
                c = c+f
            w[j+1] = (c + (n-j-1))/n
            v[j+1] = w[j+1]*np.max([1,v[j]])/w[j]
                
        
        vHist[z,0] = v[len(X)]
        wHist[z,0] = w[len(X)]
    
    thresholdArr[y,0] = np.quantile(vHist,0.95)
    thresholdArrW[y,0] = np.quantile(wHist,0.95)

    
print(time.perf_counter()-start)


#%%
#generative model, sequential inference : 1 change point and multiple observations

#nArr = np.arange(1,1000,10) #array of trial lengths 
#[1,5,10,25,50,75,100]
nArr = [1,5,10,25,50]
signal_lengthArr =[1,3,5,10,15,25,35,50,75] #! 
#[1,3,5,10,15,25,35,50,75,100]
nTrials = 2000 #no. of trials for each ITU
mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution
trialTypesSimulated = np.full((len(nArr),len(signal_lengthArr),5),0.0) #counting rates of h/m/f/c for each trial length $!
trialTypesSimulatedW = np.full((len(nArr),len(signal_lengthArr),5),0.0) #counting rates of h/m/f/c for each trial length $!


start = time.perf_counter()
for l in range(len(nArr)):
    n = nArr[l]
    threshold_v = thresholdArr[l] #!
    threshold_w = thresholdArrW[l]
    y=0
    while signal_lengthArr[y] <= n : #!
        hit = 0; miss = 0; cr = 0; fa = 0
        hitW = 0; missW = 0; crW = 0; faW = 0
        for z in range(nTrials):    
            
            trial_signal = np.random.binomial(1,0.5)  #trial is signal/non signal with prob 0.5 
            start_signal = n-signal_lengthArr[y] #index of sample when signal begins (on signal trial)
            
            trial = np.full((int(round(n)),1),0);#signal values within a trial
            if trial_signal == 1: 
                trial[start_signal:] = np.full((n-start_signal,1),1)
            
            X = np.full((int(round(n)),1),np.nan)
            for x in range(len(X)):
                if trial[x] == 0:
                    X[x] = np.random.normal(mu_n,sigma_n)
                elif trial[x] == 1:
                    X[x] = np.random.normal(mu_s,sigma_s)
                    
            #inference and response
            w = np.full((len(X)+1,1),0.0)
            v = np.full((len(X)+1,1),0.0)
            r = np.full((len(X)+1,1),0.0)
            w[0] = 1; v[0] = 1
            for j in range(len(X)):
                c = 0;
                for i in range(j+1):
                    f = np.product(np.exp(mu_s*X[i:j+1]/(sigma_s**2))) * np.exp((-j+i-1)*(mu_s**2)/(2*sigma_s**2))
                    c = c+f
                w[j+1] = (c + (n-j-1))/n
                v[j+1] = w[j+1]*np.max([1,v[j]])/w[j]
                if v[j] > 1: r[j+1] = r[j]+1
                elif v[j] <= 1: r[j+1] = 1 
            
            response = 2; 
            if v[len(X)] > threshold_v:
                response = 1; 
                if trial_signal == 1: hit = hit+1
                elif trial_signal == 0: fa = fa+1
            elif v[len(X)] < threshold_v: 
                response = 0
                if trial_signal == 1: miss = miss+1
                elif trial_signal == 0: cr = cr+1
                
            responseW = 2; 
            if w[len(X)] > threshold_w:
                responseW = 1; 
                if trial_signal == 1: hitW = hitW+1
                elif trial_signal == 0: faW = faW+1
            elif w[len(X)] < threshold_w: 
                responseW = 0
                if trial_signal == 1: missW = missW+1
                elif trial_signal == 0: crW = crW+1
                
        trialTypesSimulated[l,y,0] = hit; trialTypesSimulated[l,y,1] = miss;
        trialTypesSimulated[l,y,2] = cr; trialTypesSimulated[l,y,3] = fa;

        trialTypesSimulatedW[l,y,0] = hitW; trialTypesSimulatedW[l,y,1] = missW;
        trialTypesSimulatedW[l,y,2] = crW; trialTypesSimulatedW[l,y,3] = faW;
        

        y = y+1
        
    trialTypesSimulated[l,:,4] = norm.ppf(trialTypesSimulated[l,:,0]/(
        trialTypesSimulated[l,:,0]+trialTypesSimulated[l,:,1]))-norm.ppf(
            trialTypesSimulated[l,:,3]/(trialTypesSimulated[l,:,2]+trialTypesSimulated[l,:,3]))

print(time.perf_counter()-start)

#%%
#plotting single trial measures
fig1, ax1 = plt.subplots(2)
ax1[0].plot(trial, label = 'underlying signal')
ax1[0].plot(X, label = 'observations')
ax1[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left'); 
ax1[1].axhline(thresholdArr[0], label = 'threshold')
ax1[1].plot(w[1:], label = 'likelihood ratio')
ax1[1].plot(v[1:], label = ' discrepant group')
ax1[1].plot(r[1:], label = 'run length')
ax1[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1[1].set_xlabel('sample no.')

#%%
#plotting hit rates for different ITU lengths
for l in range(len(nArr)):

    a = trialTypesSimulatedW[l,:,0]/(trialTypesSimulatedW[l,:,0]+trialTypesSimulatedW[l,:,1])
    plt.plot(signal_lengthArr[:len(trialTypesSimulatedW[l,:,:])],a, marker = 'o', label = 'n=%d'%nArr[l])


plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('signal duration')
plt.ylabel('hit rates')

plt.figure()
#plotting hit rates for different ITU lengths
for l in range(len(nArr)):

    a = trialTypesSimulated[l,:,0]/(trialTypesSimulated[l,:,0]+trialTypesSimulated[l,:,1])
    plt.plot(signal_lengthArr[:len(trialTypesSimulated[l,:,:])],a, marker = 'o', label = 'n=%d'%nArr[l])


plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('signal duration')
plt.ylabel('hit rates')
#%%
trialTypesSimulated1, trialTypesSimulated5, trialTypesSimulated10, trialTypesSimulated25 = 1
trialTypesSimulated50, trialTypesSimulated75, trialTypesSimulated100 = 1

#%%
#plotting hit rates vs signal lengths for different ITUs but similar signal durations
a = trialTypesSimulated1[:,0]/(trialTypesSimulated1[:,0]+trialTypesSimulated1[:,1])
plt.scatter(signal_lengthArr[:len(trialTypesSimulated1)],a, label = 'n=1')

a = trialTypesSimulated5[:,0]/(trialTypesSimulated5[:,0]+trialTypesSimulated5[:,1])
plt.plot(signal_lengthArr[:len(trialTypesSimulated5)],a, label = 'n=5', linestyle = '--', marker = 'o')

a = trialTypesSimulated10[:,0]/(trialTypesSimulated10[:,0]+trialTypesSimulated10[:,1])
plt.plot(signal_lengthArr[:len(trialTypesSimulated10)],a, label = 'n=10', linestyle = '--', marker = 'o')

a = trialTypesSimulated25[:,0]/(trialTypesSimulated25[:,0]+trialTypesSimulated25[:,1])
plt.plot(signal_lengthArr[:len(trialTypesSimulated25)],a, label = 'n=25', linestyle = '--', marker = 'o')

a = trialTypesSimulated50[:,0]/(trialTypesSimulated50[:,0]+trialTypesSimulated50[:,1])
plt.plot(signal_lengthArr[:len(trialTypesSimulated50)],a, label = 'n=50', linestyle = '--', marker = 'o')

a = trialTypesSimulated75[:,0]/(trialTypesSimulated75[:,0]+trialTypesSimulated75[:,1])
plt.plot(signal_lengthArr[:len(trialTypesSimulated75)],a, label = 'n=75', linestyle = '--', marker = 'o')

a = trialTypesSimulated100[:,0]/(trialTypesSimulated100[:,0]+trialTypesSimulated100[:,1])
plt.plot(signal_lengthArr[:len(trialTypesSimulated100)],a, label = 'n=100', linestyle = '--', marker = 'o')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('signal duration')
plt.ylabel('hit rates')

#%%
sArr  = [0.125, 0.25, 0.5, 1, 1.5, 10]
for i in range(len(sArr)):
    rangeX = np.arange(0.0,2.55,0.05)
    rangeY = lognormalCDF(rangeX, 0, sArr[i])
    plt.plot(rangeX, rangeY, label = 'sig=%1.3f'%sArr[i])
plt.legend()

#%%
typesExpected = np.full((50,len(nArrExp),5),0.0)
typesSimulated = np.full((50,len(nArr),5),0.0)
mu_sArray = np.full((50,1),0.0)
#%%
for i in range(5):
    plt.plot(nArr, typesSimulated[i,:,0]/(typesSimulated[i,:,0]+typesSimulated[i,:,1]), 
             label = 'mu_s=%1.2f'%mu_sArray[i])
plt.legend(); plt.figure()
for i in range(5):
    plt.plot(nArr, typesSimulated[i,:,3]/(typesSimulated[i,:,3]+typesSimulated[i,:,4]), 
             label = 'mu_s=%1.2f'%mu_sArray[i])
plt.legend(); plt.figure()


#%%
#fixed r - sequential inference
def trial_fixedSignal(trial_length,mu_s, sigma_s, mu_n, sigma_n,p_signal,trial_type,
                      signal_length):
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

def response_fixedSignal(trial_signal, trial, observations,w,v,threshold):

    response = 2;
    hit = 0; miss = 0; cr = 0; fa = 0
    threshold_v = threshold
    if v[len(observations)] > threshold_v:
        response = 1; 
        if trial_signal == 1: hit = hit+1
        elif trial_signal == 0: fa = fa+1
    elif v[len(observations)] < threshold_v: 
        response = 0
        if trial_signal == 1: miss = miss+1
        elif trial_signal == 0: cr = cr+1
            
    return response, hit, miss, cr, fa

#%%
trial_length = 5
mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution
p_signal = 0.5; # probability of a signal trial
trial_type = 1; signal_length = 5

trial_signal, trial, observations, w,v = trial_fixedSignal(trial_length,mu_s, 
   sigma_s, mu_n, sigma_n,p_signal,trial_type,signal_length)

#%%
#plot of result
plt.plot(trial, label = 'underlying signal'); plt.plot(observations, label = 'observations')
plt.legend(); plt.figure()
plt.plot(w[1:], label = 'likelihood ratio'); plt.plot(v[1:], label = 'most discrepant')
plt.legend()


#%%
#finding critical value (threshold) for the statistic : sequential inference,
#fixed signal length
signal_lengthArr =[3,5,10,15,25,35,50,75,100,101] #signal length [1,3,5,10,15,25,35,50,75,100,101]
nArr = [5,10,25,50,75,100] #array of trial lengths  
nTrials = 2000 #no. of trials for each trial length
mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution
p_signal = 0.5 # probability of a signal trial
trial_type = 0

vHist = np.full((nTrials,1),0.0) #histogram of the statistic
thresholdArr = np.full((len(nArr),len(signal_lengthArr)),0.0)
#array of thresholds for each value of n


start = time.perf_counter()
for y in range(len(nArr)):
    n = nArr[y]; #no. of samples (trial length)
    s= 0
    while signal_lengthArr[s] <= n: 
        signal_length = signal_lengthArr[s]
        for z in range(nTrials):    
            
            trial_signal, trial, observations, w,v = trial_fixedSignal(n,mu_s, 
            sigma_s, mu_n, sigma_n,p_signal,trial_type,signal_length)
                    
            vHist[z,0] = v[len(observations)]
        
        thresholdArr[y,s] = np.quantile(vHist,0.95)
        s = s+1
  
print(time.perf_counter()-start)

#%%
#generative model, sequential inference : fixed signal duration
    
#nArr = np.arange(1,1000,10) #array of trial lengths 
#[1,5,10,25,50,75,100]
signal_lengthArr =[3,5,10,15,25,35,50,75,100] #signal length [1,3,5,10,15,25,35,50,75,100]
nArr = [5,10,25,50,75] #array of trial lengths 
nTrials = 2000 #no. of trials for each trial length
mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution
p_signal = 0.5 # probability of a signal trial
signal_length = 4; trial_type = 2

trialTypesSimulated = np.full((len(nArr),len(signal_lengthArr),5),0.0) #counting rates of h/m/f/c for each trial length $!


start = time.perf_counter()
for y in range(len(nArr)):
    n = nArr[y]; #no. of samples (trial length) 
    s=0
    while signal_lengthArr[s] <= n: 
        signal_length = signal_lengthArr[s]        
        threshold_v = thresholdArr[y,s] #!
        hit = 0; miss = 0; cr = 0; fa = 0
        for z in range(nTrials):    
                    
            trial_signal, trial, observations, w,v = trial_fixedSignal(n,mu_s, 
           sigma_s, mu_n, sigma_n,p_signal,trial_type, signal_length)
     
            repsonse, hit0, miss0, cr0, fa0 = response_fixedSignal(trial_signal,
                                    trial, observations,w,v, threshold_v)
        
            hit = hit+hit0; miss = miss+miss0; cr = cr+cr0; fa = fa+fa0
                
        trialTypesSimulated[y,s,0] = hit; trialTypesSimulated[y,s,1] = miss;
        trialTypesSimulated[y,s,2] = cr; trialTypesSimulated[y,s,3] = fa;
        
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
plt.ylabel('hit rates')

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
    

