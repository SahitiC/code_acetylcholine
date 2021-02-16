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
mu_s = 2; sigma_s = 1; #mean and std dev of signal distribution
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
nTrials = 2000
nArr = np.arange(1,1000,100)
r = 1; #length of signal
mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution
vHist = np.full((nTrials,1),0.0) #array of v's to calculate histogram
thresholdArr = np.full((len(nArr),1),0.0) #array of thresholds for each value of n

for na in range(len(nArr)):
    n = nArr[na]; #no. of samples
    
    #under H1:
    for nt in range(nTrials):
        trial_signal = np.random.binomial(1,0.5) #trial is signal/non signal with prob 0.5
        start_signal = np.random.randint(0, n-r+1) #index of sample when signal begins (on signal trial)
        
        trial = np.full((int(round(n)),1),0);#signal values within a trial
        if trial_signal == 1: 
            trial[start_signal:start_signal+r] = np.full((int((r)),1),1)
            
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
            s[j+1] = s[j] + np.exp((mu_s*X[j])/(sigma_s**2))
            h[j+1] = (s[j]+(n-j+1)*c)/(s[j+1] + (n-j)*c)
            v[j+1] = min(h[j+1], v[j])
        
        vHist[nt,0] = v[len(X)]
    
    thresholdArr[na,0] = np.quantile(vHist,0.05)

#%%
#generative model - sequential inference (h,v,s method)
    
hit = 0; fa = 0; miss = 0; cr = 0
n = 10; #no. of samples
r = 1; #length of signal
mu_s = 1; sigma_s = 1; #mean and std dev of signal distribution
mu_n = 0; sigma_n = 1; #mean and std dev of non-signal distribution
threshold_v = 0.7

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
    s[j+1] = s[j] + np.exp((mu_s*X[j])/(sigma_s**2))
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
    
   

#%%

#generative model - sequential inference

nArr = np.arange(1,1000,100) #array of trial lengths 
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

nArrExp = np.arange(1,1000,1) #array of trial lengths 
#nArrExp = [1]
trialTypesExpected = np.full((len(nArrExp),5),0.0)
mu_s = 0.2; sigma_s = 1; #mean and std dev of signal distribution
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
plt.plot(nArrExp, trialTypesExpected[:,0], label ='hits approx')
plt.plot(nArrExp, trialTypesExpected[:,2], label ='fa approx')
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
    

