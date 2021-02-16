import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import fsolve
from scipy.optimize import brentq
import time
#%%
#import data

dataSDT = np.loadtxt("dataSDT.csv")

#%%
#fixed params
mu_n = 0; sigma_n = 1 #mean and std dev for noise 
sigma_s = 1; #std deviation for signal
d1 = 0.025; d2 = 0.050; d3 = 0.500 #duration length

#equation for lrt (solve to get threshold on x)
def lrt(x):
    f = np.exp((mu_n**2-mu_d1**2+2*x*mu_d1-2*x*mu_n)/(2*sigma_s**2)
               )+np.exp((mu_n**2-mu_d2**2+2*x*mu_d2-2*x*mu_n)/(2*sigma_s**2))+np.exp(
                  (mu_n**2-mu_d3**2+2*x*mu_d3-2*x*mu_n)/(2*sigma_s**2)) - 3
    return f

#%%
#simulate a trial 

def trial(mu_n,sigma_n, mu_d1, mu_d2, mu_d3, sigma_s):
    i = np.random.binomial(1,0.5) #signal trial if 1, non-signal otherwise
    if i == 1:
        j = np.random.choice(3, p = np.array([1,1,1])/3)
        if j == 0:
            signal = 25
            x = np.random.normal(mu_d1, sigma_s)
        elif j == 1:
            signal = 50
            x = np.random.normal(mu_d2, sigma_s)
        elif j == 2:
            signal = 500
            x = np.random.normal(mu_d3, sigma_s)
    elif i == 0:
        signal = 0
        x = np.random.normal(mu_n, sigma_n)
    
    #optimal threshold
    c = fsolve(lrt,1)
    
    #response
    if x > c:
        response = 1
    elif x < c:
        response = 0
    else:
        i = np.random.binomial(1,0.5)
        if i == 1:
            response = 1
        elif i == 0:
            response = 0
    
    return response, signal


#%%
#calculating neg loglikelihood of data given parameters
def negloglikelihood(parameter):
    mu_d1 = parameter[0]; mu_d2 = parameter[1]; mu_d3 = parameter[2] 
    #mean for the 3 durations
    c = fsolve(lrt, 1)
    
    L = 0; #initialising -log likelhood
    
    for i in range(len(dat)):
        if dat[i,0] == 1:
            if dat[i,1] == 0:
                L = L - np.log(1-norm(mu_n,sigma_n).cdf(c))
            elif dat[i,1] == 25:            
                L = L-np.log(1-norm(mu_d1,sigma_s).cdf(c))
            elif dat[i,1] == 50:
                L = L-np.log(1-norm(mu_d2,sigma_s).cdf(c))
            elif dat[i,1] == 500:
                L = L-np.log(1-norm(mu_d3,sigma_s).cdf(c))

        elif dat[i,0] == 0:
            if dat[i,1] == 0:
                L = L - np.log(norm(mu_n,sigma_n).cdf(c))
            elif dat[i,1] == 25:            
                L = L-np.log(norm(mu_d1,sigma_s).cdf(c))
            elif dat[i,1] == 50:
                L = L-np.log(norm(mu_d2,sigma_s).cdf(c))
            elif dat[i,1] == 500:
                L = L-np.log(norm(mu_d3,sigma_s).cdf(c))
                
        if np.isnan(L) or np.isinf(L):
            print(L, mu_d1, mu_d2, mu_d3, c)
            break
    
    #print(L,mu_s, sigma_d1, sigma_d2, sigma_d3, c)
        
    return L

#prob responding to non-signal given S,N
#probNS = (norm(mu_s*d1,sigma_d1).cdf(c)+
#norm(mu_s*d2,sigma_d2).cdf(c)+ norm(mu_s*d2,sigma_d2).cdf(c))/3
#probNN = norm(mu_n, sigma_n).cdf(c)  

#%%
#single simulation of data and fitting via minimising loglikelihood
start = time.perf_counter()
n = 200; mu_d1 = 0.5;mu_d2 = 1; mu_d3 = 1.5;
dataSim = np.full((n,2),0)            
for i in range(n):
    dataSim[i] = trial(mu_n,sigma_n, mu_d1, mu_d2, mu_d3, sigma_s)   
dat = dataSim

mu_d1 = 1.5;mu_d2 = 1; mu_d3 = 2; 
result = minimize(negloglikelihood,[mu_d1, mu_d2, mu_d3], 
     bounds = [(None,None),(None,None), (None, None)])
print(time.perf_counter()-start)

#%%
#estimating mu_d1 (starting from 'real' values), doing this over a range of real mu_d1's
mu_d1Arr = [0.1, 0.4, 0.7, 1.0, 1.3]
meanArr = np.full((5,10), 0.0, dtype = 'object'); varArr = np.full((5,10), 0.0, dtype = 'object'); likeArr = np.full((5,10), 0.0)
for i in range(len(mu_d1Arr)): 
    for j in range(10):
        n = 200; mu_d1 = mu_d1Arr[i];mu_d2 = 1; mu_d3 = 1.5;
        dataSim = np.full((n,2),0)            
        for k in range(n):
            dataSim[k] = trial(mu_n,sigma_n, mu_d1, mu_d2, mu_d3, sigma_s)   
        dat = dataSim

        mu_d1 = mu_d1Arr[i];mu_d2 = 1; mu_d3 = 1.5; 
        result = minimize(negloglikelihood,[mu_d1, mu_d2, mu_d3], 
             bounds = [(None,None),(None,None), (None, None)])
        meanArr[i,j] = result['x']; varArr[i,j] = result['hess_inv'].todense(); likeArr[i,j] = result['fun']

m = np.full((5,10), 0.0); std = np.full((5,10), 0.0)
for i in range(len(mu_d1Arr)):
    for j in range(10):
        m[i,j] = meanArr[i,j][0]
        std[i,j]=np.sqrt(varArr[i,j][0,0])

for i in range(10):
    plt.errorbar(mu_d1Arr, m[:,i], linestyle = "None", marker = 'o')
plt.plot(mu_d1Arr, np.mean(std, axis = 1), marker = 'o', label = 'avg std dev across fits')

plt.legend()
plt.xlabel('mu_d1'); plt.ylabel('estimate of mu_d1 from fit'); plt.title('mu_d2 = 1, mu_d3 = 1.5')

#%%
# plot of data under params likelihood
n = 1000; mu_d1 = 0.5;mu_d2 = 1; mu_d3 = 3;
mu_d1Arr = [0.1, 1, 2, 3]

dataSim = np.full((n,2),0)            
for j in range(n):
    dataSim[j] = trial(mu_n,sigma_n, mu_d1, mu_d2, mu_d3, sigma_s)
dat = dataSim
mu_Arr = np.linspace(-1, 7, 100)
nlikehd1 = np.full((100,1), 0.0);nlikehd2 = np.full((100,1), 0.0)
nlikehd3 = np.full((100,1), 0.0)
for j in range(100):
    nlikehd1[j] = negloglikelihood([mu_Arr[j],1,3])
    nlikehd2[j] = negloglikelihood([0.5,mu_Arr[j],3])
    nlikehd3[j] = negloglikelihood([0.5,1,mu_Arr[j]])
plt.plot(mu_Arr,nlikehd1, label = 'mu_d1=%1.2f'%mu_d1); 
plt.scatter(mu_Arr[np.where(nlikehd1 == min(nlikehd1))[0][0]],min(nlikehd1))
#plt.plot(mu_Arr,nlikehd2, label = 'mu_d2=%1.2f'%mu_d2); 
#plt.plot(mu_Arr,nlikehd3, label = 'mu_d3=%1.2f'%mu_d3) 

plt.legend()
plt.xlabel('mu'); plt.ylabel('Negative Log Likelihood')

mu_d1 = 0.5;mu_d2 = 1; mu_d3 = 1.5; 
result = minimize(negloglikelihood,[mu_d1, mu_d2, mu_d3], 
     bounds = [(None,None),(None,None), (None, None)])


#%%
#finding params by starting from a range of initial conditions
n = 10; # no. of iterations
resultArr = np.full((n,4), 0.0, dtype = 'object')

dataSim = np.full((200,2),0)            
for i in range(200):
    dataSim[i] = trial(0,1,0.5,1,1.5,1,0.7)
    
dat = dataSim

for i in range(n):
    mu_d1 = np.random.uniform(0,10,1)
    mu_d2 = np.random.uniform(0,10,1); mu_d3 = np.random.uniform(0.1,10,1)
    c = fsolve(lrt,1)
    resultArr[i,2] = [mu_d1, mu_d2, mu_d3, c]
    result = minimize(negloglikelihood,[mu_d1, mu_d2, mu_d3], 
     bounds = [(None,None),(None,None), (None, None)])
    
    resultArr[i,0] = result['fun']; resultArr[i,1] = result['x']
    resultArr[i,3] = result['hess_inv'].todense()

#v =  np.where(resultArr[:,0]==min(resultArr[:,0]))  
#estimate[j] = resultArr[v,1][0][0]; hessEst =  resultArr[v,3][0][0]
#stderrEst[j] = np.array([ np.sqrt(hessEst[0,0]),  np.sqrt(hessEst[1,1]), np.sqrt(hessEst[2,2])])

#%%
#analysing simulated data
def multiTrial(n, mu_d1, mu_d2, mu_d3):
    dataSim = np.full((n,2),0)            
    for j in range(n):
        dataSim[j] = trial(mu_n,sigma_n, mu_d1, mu_d2, mu_d3, sigma_s)
    
    accuracy = np.full((4, 1), 0.0)
    accuracy[0,0] = len(np.intersect1d(np.where(dataSim[:,1] == 25),np.where(dataSim[:,0] == 1)))/len(np.where(dataSim[:,1] == 25)[0])
    accuracy[1,0] = len(np.intersect1d(np.where(dataSim[:,1] == 50),np.where(dataSim[:,0] == 1)))/len(np.where(dataSim[:,1] == 50)[0])
    accuracy[2,0] = len(np.intersect1d(np.where(dataSim[:,1] == 500),np.where(dataSim[:,0] == 1)))/len(np.where(dataSim[:,1] == 500)[0])
    accuracy[3,0] = len(np.intersect1d(np.where(dataSim[:,1] == 0),np.where(dataSim[:,0] == 0)))/len(np.where(dataSim[:,1] == 0)[0])

    return dataSim, accuracy

#%%
n = 1000; mu_d1 = 0.5;mu_d2 = 1; mu_d3 = 1.5;
d, acc = multiTrial(n, mu_d1, mu_d2, mu_d3)

plt.plot(['25', '50', '500', '0'], acc, marker = 'o')
plt.title('mu_d1=%1.2f, mu_d2=%1.2f, mu_d3=%1.2f' %(mu_d1, mu_d2, mu_d3))

#%%
n = 2000; mu_d1 = 0.5;mu_d2 = 1; mu_d3 = 1.5;
mu_d1Arr = [0.1, 0.4, 0.7, 1.0]
for i in range(len(mu_d1Arr)):
    mu_d1 = mu_d1Arr[i]
    d, acc = multiTrial(n, mu_d1, mu_d2, mu_d3)
    plt.plot(['25', '50', '500', '0'], acc, marker = 'o', label = 'mu_d1=%1.2f'%mu_d1Arr[i])
    plt.title('mu_d2=%1.2f, mu_d3=%1.2f' %(mu_d2, mu_d3))
    #plot expected hit rates:
    c = fsolve(lrt, 1)
    a = [1-norm(mu_d1, sigma_s).cdf(c), 1-norm(mu_d2, sigma_s).cdf(c), 1-norm(mu_d3, sigma_s).cdf(c), norm(mu_n, sigma_n).cdf(c) ]
    plt.plot(['25', '50', '500', '0'], a, marker = 'o', linestyle = 'dashed', label = 'mu_d1=%1.2f'%mu_d1Arr[i])
    
plt.legend(); plt.xlabel('signal duation in ms')
plt.ylabel('accuracy rate')

#%%

dataSDT = np.loadtxt("dataSDT.csv")

for s in range(10):
    sessionNo = s+1+10; #session number to fit
    data = np.full((251,2), np.nan)
    data[:,0]= dataSDT[251*(sessionNo-1):251*sessionNo,1] #response data
    data[:,1]= 1000*dataSDT[251*(sessionNo-1):251*sessionNo,0] #signal duration data
    
    data = data[~np.isnan(data).any(axis=1)] #drop nan rows
    data = data.astype(int)
    
    #fitting real data
    dat = data
    
    nIters = 50; # no. of iterations
    resultArr = np.full((nIters,4), 0.0, dtype = 'object')
    
    for i in range(nIters):
        mu_d1 = np.random.uniform(0,10,1)
        mu_d2 = np.random.uniform(0,10,1); mu_d3 = np.random.uniform(0.1,10,1)
        c = fsolve(lrt,1)
        resultArr[i,2] = [mu_d1, mu_d2, mu_d3, c]
        result = minimize(negloglikelihood,[mu_d1, mu_d2, mu_d3], 
         bounds = [(None,None),(None,None), (None, None)])
        
        resultArr[i,0] = result['fun']; resultArr[i,1] = result['x']
        resultArr[i,3] = result['hess_inv'].todense()
    
    idx =  np.where(resultArr[:,0]==min(resultArr[:,0]))[0][0]
    res = resultArr[idx,1]; resStd = resultArr[13,3]
    #calculating accuracy (expected) based on the params obtained here
    mu_d1 = res[0]; mu_d2 =res[1]; mu_d3 =res[2]; c = 2
    c = fsolve(lrt, 1)
    a = [1-norm(mu_d1, sigma_s).cdf(c), 1-norm(mu_d2, sigma_s).cdf(c), 1-norm(mu_d3, sigma_s).cdf(c), norm(mu_n, sigma_n).cdf(c) ]
    
    dat = data
    aReal = np.full((4, 1), 0.0)
    aReal[0,0] = len(np.intersect1d(np.where(dat[:,1] == 25),np.where(dat[:,0] == 1)))/len(np.where(dat[:,1] == 25)[0])
    aReal[1,0] = len(np.intersect1d(np.where(dat[:,1] == 50),np.where(dat[:,0] == 1)))/len(np.where(dat[:,1] == 50)[0])
    aReal[2,0] = len(np.intersect1d(np.where(dat[:,1] == 500),np.where(dat[:,0] == 1)))/len(np.where(dat[:,1] == 500)[0])
    aReal[3,0] = len(np.intersect1d(np.where(dat[:,1] == 0),np.where(dat[:,0] == 0)))/len(np.where(dat[:,1] == 0)[0])
    
    plt.plot(['25', '50', '500', '0'], a, marker = 'o', label = 'expected accuracy')
    plt.plot(['25', '50', '500', '0'], aReal, marker = 'o', label = 'observed accuracy')
    plt.xlabel('signal duration in (ms)'); plt.title('session#%d, estimated params=%1.2f, %1.2f, %1.2f'%(sessionNo, mu_d1, mu_d2, mu_d3))
    plt.legend(); plt.figure()
    
#%%
dataSDT = np.loadtxt("dataSDTFull.csv")
s = 1
sessionNo = s+1; #session number to fit
data = np.full((251,2), np.nan)
data[:,0]= dataSDT[251*(sessionNo-1):251*sessionNo,1] #response data
data[:,1]= 1000*dataSDT[251*(sessionNo-1):251*sessionNo,0] #signal duration data

data = data[~np.isnan(data).any(axis=1)] #drop nan rows
data = data.astype(int)

#fitting real data
dat = data

nIters = 50; # no. of iterations
resultArr = np.full((nIters,4), 0.0, dtype = 'object')

start = time.perf_counter()
for i in range(nIters):
    mu_d1 = np.random.uniform(0,10,1)
    mu_d2 = np.random.uniform(0,10,1); mu_d3 = np.random.uniform(0.1,10,1)
    c = fsolve(lrt,1)
    resultArr[i,2] = [mu_d1, mu_d2, mu_d3, c]
    result = minimize(negloglikelihood,[mu_d1, mu_d2, mu_d3], 
     bounds = [(None,None),(None,None), (None, None)])
    
    resultArr[i,0] = result['fun']; resultArr[i,1] = result['x']
    resultArr[i,3] = result['hess_inv'].todense()
    
print(time.perf_counter()-start)

idx =  np.where(resultArr[:,0]==min(resultArr[:,0]))[0][0]
res = resultArr[idx,1]; resStd = resultArr[13,3]
#calculating accuracy (expected) based on the params obtained here
mu_d1 = res[0]; mu_d2 =res[1]; mu_d3 =res[2]; c = 2
c = fsolve(lrt, 1)
a = [1-norm(mu_d1, sigma_s).cdf(c), 1-norm(mu_d2, sigma_s).cdf(c), 1-norm(mu_d3, sigma_s).cdf(c), norm(mu_n, sigma_n).cdf(c) ]

dat = data
aReal = np.full((4, 1), 0.0)
aReal[0,0] = len(np.intersect1d(np.where(dat[:,1] == 25),np.where(dat[:,0] == 1)))/len(np.where(dat[:,1] == 25)[0])
aReal[1,0] = len(np.intersect1d(np.where(dat[:,1] == 50),np.where(dat[:,0] == 1)))/len(np.where(dat[:,1] == 50)[0])
aReal[2,0] = len(np.intersect1d(np.where(dat[:,1] == 500),np.where(dat[:,0] == 1)))/len(np.where(dat[:,1] == 500)[0])
aReal[3,0] = len(np.intersect1d(np.where(dat[:,1] == 0),np.where(dat[:,0] == 0)))/len(np.where(dat[:,1] == 0)[0])

plt.plot(['25', '50', '500', '0'], a, marker = 'o', label = 'expected accuracy')
plt.plot(['25', '50', '500', '0'], aReal, marker = 'o', label = 'observed accuracy')
plt.ylabel('signal duration in (ms)'); plt.title('session#%d, estimated params=%1.2f, %1.2f, %1.2f'%(sessionNo, mu_d1, mu_d2, mu_d3))
plt.legend(); plt.figure()

#%%
mu_n = 0; sigma_n = 1; sigma_s =1;
mu_d1 = -0.43; mu_d2 = 0.53; mu_d3 = 0.79;
c = fsolve(lrt, 1)
x = np.arange(-5,5,0.01)
y1 = norm.pdf(x,0,sigma_n); y2 = norm.pdf(x,mu_d1,sigma_s)
y3 = norm.pdf(x,mu_d2,sigma_s); y4 = norm.pdf(x,mu_d3,sigma_s)
plt.plot(x,y1,label ='n'); plt.axvline(c)
plt.plot(x,y2, label ='d1'); plt.plot(x,y3, label ='d2'); 
plt.plot(x,y4, label ='d3')
plt.title('c=%1.2f, mu_d1=%1.2f, mu_d2=%1.2f, mu_d3=%1.2f'%(c, mu_d1, mu_d2, mu_d3)); 
plt.legend(); plt.figure()

dif = [-.4,-.3,-0.2,-.1,0,.1,.2,.3,.4,.5]
mu_d1C = -0.13; mu_d2C = 0.83; mu_d3C = 1.09;
for i in range(len(dif)):
    mu_d1 = mu_d1C+dif[i];mu_d2 = mu_d2C+dif[i]; mu_d3 = mu_d3C+dif[i]
    c = fsolve(lrt, 1)
    y = [1-norm(mu_d1, sigma_s).cdf(c), 1-norm(mu_d2, sigma_s).cdf(c), 1-norm(mu_d3, sigma_s).cdf(c), norm(mu_n, sigma_n).cdf(c) ]
    plt.plot(['25', '50', '500', '0'],y, marker = 'o', label='d=%1.2f'%dif[i])
plt.legend(); plt.xlabel('signal duration'); plt.ylabel('expected accuracy')

#%%
#fitting by minimising error between hit and false alarm rates (observed and expected)
def errorFun(parameter):
    l = len(dat)
    mu_d1 = parameter[0]; mu_d2 = parameter[1]; mu_d3 = parameter[2]  #mean for the 3 durations
    c = fsolve(lrt, 1) #threshold
    hitExpected = 3 - (norm(mu_d1, sigma_s).cdf(c)+norm(mu_d2, sigma_s).cdf(c)+norm(mu_d3, sigma_s).cdf(c))
    faExpected = 1- norm(mu_n, sigma_n).cdf(c)
    err = 0
    for i in range(nSamples):
        sample  = np.random.choice(l, 150, replace = False)
        datSamp = dat[sample]
        hitObserved = (len(np.intersect1d(np.where(datSamp[:,1] == 25),
        np.where(datSamp[:,0] == 1)))+len(np.intersect1d(np.where(
        datSamp[:,1] == 25),np.where(datSamp[:,0] == 1)))+len(np.intersect1d(
            np.where(datSamp[:,1] == 25),np.where(datSamp[:,0] == 1))))/(len(
                np.where(datSamp[:,1] == 25)[0])+len(np.where(datSamp[:,1] == 25)[0])+len(np.where(datSamp[:,1] == 25)[0]))
        faObserved = len(np.intersect1d(np.where(datSamp[:,1] == 0),
        np.where(datSamp[:,0] == 1)))/len(np.where(datSamp[:,1] == 0)[0])
        err = err+ (hitObserved - hitExpected)**2 + (faObserved - faExpected)**2
    err = err/(nSamples)
    return err

#%%
n = 1000; mu_d1 = 0.5;mu_d2 = 1; mu_d3 = 1.5;
dataSim = np.full((n,2),0)            
for j in range(n):
    dataSim[j] = trial(mu_n,sigma_n, mu_d1, mu_d2, mu_d3, sigma_s)
dat = dataSim
       
nSamples = 100; sizeSamples = 150
r = errorFun([0.5,1,1.5])

#%%
#with different  for signal and noise distributions

#%%
mu_n = 0; sigma_n = 1 #mean and std dev for noise 
d1 = 0.025; d2 = 0.050; d3 = 0.500 #duration length

#equation for lrt with different variances(solve to get threshold on x)
def lrtVar(x, mu_d1, mu_d2, mu_d3, sigma_s):
    f = np.exp(((-x**2-mu_d1**2+2*mu_d1*x)/(2*sigma_s**2))+ (
        x**2+mu_n**2-2*mu_n*x)/(2*sigma_n**2)) + np.exp(((-x**2-mu_d2**2+2*mu_d2*x
    )/(2*sigma_s**2))+ (x**2+mu_n**2-2*mu_n*x)/(2*sigma_n**2))+np.exp(((
    -x**2-mu_d3**2+2*mu_d3*x)/(2*sigma_s**2))+ (x**2+mu_n**2-2*mu_n*x)/(2*sigma_n**2))- 3*sigma_s/sigma_n 
    
    return f

#%%
#finding threshold given parameters
def threshold(param):
    mu_d1 = param[0]; mu_d2 = param[1];
    mu_d3 = param[2]; sigma_s = param[3]
    xmin = -25
    xmax = 25
    xarr = np.arange(xmin, xmax, 0.1)
    yarr = lrtVar(xarr,mu_d1, mu_d2, mu_d3, sigma_s); 
    ysign = np.sign(yarr)
    signchange = ((np.roll(ysign, 1) - ysign) != 0).astype(int); signchange[0] = 0
    changepoints = np.where(signchange == 1)[0]
    c = np.full((len(changepoints),1),0.0)
    for i in range(len(changepoints)):
        c[i] = brentq(lrtVar,xarr[changepoints[i]-1],xarr[changepoints[i]],
                      args = (mu_d1, mu_d2, mu_d3, sigma_s))
        
    return c
    

#%%
#generate a single trial
def trialVar(mu_n,sigma_n, mu_d1, mu_d2, mu_d3, sigma_s,c):
    i = np.random.binomial(1,0.5) #signal trial if 1, non-signal otherwise
    if i == 1:
        j = np.random.choice(3, p = np.array([1,1,1])/3)
        if j == 0:
            signal = 25
            x = np.random.normal(mu_d1, sigma_s)
        elif j == 1:
            signal = 50
            x = np.random.normal(mu_d2, sigma_s)
        elif j == 2:
            signal = 500
            x = np.random.normal(mu_d3, sigma_s)
    elif i == 0:
        signal = 0
        x = np.random.normal(mu_n, sigma_n)
    
    #response
    if sigma_s>1:
        if x < c[0]:response = 1
        elif x > c[0] and x < c[1]:response = 0
        elif x > c[1]:response = 1
        else:
            r = np.random.binomial(1,0.5)
            if r == 1:response = 1
            elif r == 0:response = 0
            
    elif sigma_s<1:
        clen = len(c)
        if x < c[0]:response = 0
        for j in range(clen-1):
            if x > c[j] and x < c[j+1]:response = int(0.5*(1+(-1)**j))
            elif x == c[j] :
                r = np.random.binomial(1,0.5)
                if r == 1:response = 1
                elif r == 0:response = 0
        if x > c[clen-1]: response = 0
            
    elif round(sigma_s,1) == 1:
        if x<c[0]: response = 0
        elif x>c[0]: response = 1
        else: 
            r = np.random.binomial(1,0.5)
            if r == 1:response = 1
            elif r == 0:response = 0                                    
            
    return response, signal

#%%
#calculating neg loglikelihood of data given parameters
def negloglikelihoodVar(parameter):
    mu_d1 = parameter[0]; mu_d2 = parameter[1]; mu_d3 = parameter[2] 
    sigma_s = parameter[3] #mean and std for the 3 durations
    
    c = threshold([mu_d1,mu_d2,mu_d3, sigma_s])
    
    L = 0; #initialising -log likelhood
    
    for i in range(len(dat)):
        if len(c) == 0:
            break
        if lrtVar(c[0]-0.5, mu_d1, mu_d2, mu_d3, sigma_s) > 0:
            clen = len(c)
            prob = 0.5*(1+(-1)**clen)
            for j in range(clen):
                if dat[i,1] == 0:prob = prob+(norm(mu_n,sigma_n).cdf(c[j])*(-1)**(j))
                elif dat[i,1] == 25:prob = prob+(norm(mu_d1,sigma_s).cdf(c[j])*(-1)**(j))
                elif dat[i,1] == 50:prob = prob+(norm(mu_d2,sigma_s).cdf(c[j])*(-1)**(j))
                elif dat[i,1] == 500:prob = prob+(norm(mu_d3,sigma_s).cdf(c[j])*(-1)**(j))
                

        elif lrtVar(c[0]-0.5, mu_d1, mu_d2, mu_d3, sigma_s) <= 0:
            clen = len(c)
            prob = 0.5*(1+(-1)**(clen+1))
            for j in range(clen):
                if dat[i,1] == 0:prob = prob+(norm(mu_n,sigma_n).cdf(c[j])*(-1)**(j+1))
                elif dat[i,1] == 25:prob = prob+(norm(mu_d1,sigma_s).cdf(c[j])*(-1)**(j+1))
                elif dat[i,1] == 50:prob = prob+(norm(mu_d2,sigma_s).cdf(c[j])*(-1)**(j+1))
                elif dat[i,1] == 500:prob = prob+(norm(mu_d3,sigma_s).cdf(c[j])*(-1)**(j+1))
        
        if dat[i,0] == 1: 
            L = L - np.log(prob)
        elif dat[i,0] == 0:
            L = L - np.log(1-prob)
             
                                        
        if np.isnan(L) or np.isinf(L):
            print(L, mu_d1, mu_d2, mu_d3,sigma_s, c)
            break
    
    #print(L,mu_s, sigma_d1, sigma_d2, sigma_d3, c)
        
    return L

#%%
#analysing simulated data
def multiTrialVar(n, mu_d1, mu_d2, mu_d3, sigma_s):
    dataSim = np.full((n,2),0)            
    for j in range(n):
        dataSim[j] = trialVar(mu_n,sigma_n, mu_d1, mu_d2, mu_d3, sigma_s)
    
    accuracy = np.full((4, 1), 0.0)
    accuracy[0,0] = len(np.intersect1d(np.where(dataSim[:,1] == 25),np.where(dataSim[:,0] == 1)))/len(np.where(dataSim[:,1] == 25)[0])
    accuracy[1,0] = len(np.intersect1d(np.where(dataSim[:,1] == 50),np.where(dataSim[:,0] == 1)))/len(np.where(dataSim[:,1] == 50)[0])
    accuracy[2,0] = len(np.intersect1d(np.where(dataSim[:,1] == 500),np.where(dataSim[:,0] == 1)))/len(np.where(dataSim[:,1] == 500)[0])
    accuracy[3,0] = len(np.intersect1d(np.where(dataSim[:,1] == 0),np.where(dataSim[:,0] == 0)))/len(np.where(dataSim[:,1] == 0)[0])

    return dataSim, accuracy

#%%
#single simulation of data and fitting via minimising loglikelihood
start = time.perf_counter()
n = 2000; mu_d1 = 0.5;mu_d2 = 1.; mu_d3 = 1.5; sigma_s = 1.5
c = threshold([mu_d1, mu_d2, mu_d3, sigma_s])
dataSim = np.full((n,2),0)            
for i in range(n):
    dataSim[i] = trialVar(mu_n,sigma_n, mu_d1, mu_d2, mu_d3, sigma_s,c)
dat = dataSim
#mu_d1 = 0.5;mu_d2 = 1.; mu_d3 = 1.5; sigma_s = 1.5
mu_d1 = np.random.uniform(0,5,1)
mu_d2 = np.random.uniform(0,5,1); mu_d3 = np.random.uniform(0,5,1)
sigma_s = np.random.uniform(0.1,4, 1)
result = minimize(negloglikelihoodVar,[mu_d1, mu_d2, mu_d3, sigma_s], 
     bounds = [(None,None),(None,None), (None, None), (0.1,None)])
print(time.perf_counter()-start)

#%%
start = time.perf_counter()
n = 200; mu_d1 = 0.5;mu_d2 = 1.; mu_d3 = 1.5; sigma_s = 1.5
c = threshold([mu_d1, mu_d2, mu_d3, sigma_s])
dataSim = np.full((n,2),0)            
for i in range(n):
    dataSim[i] = trialVar(mu_n,sigma_n, mu_d1, mu_d2, mu_d3, sigma_s,c)
dat = dataSim

lkhd = negloglikelihoodVar([ mu_d1, mu_d2, mu_d3, sigma_s])
print(time.perf_counter()-start)

#%%
#calculating expected accuracy for different parameter values
mu_d1 = 0.5;mu_d2 = 1.; mu_d3 = 1.5; sigma_s = 1.5
n = 2000; sigmaArr = [0.1,0.5,0.9,1.3,1.7]

for i in range(len(sigmaArr)):
    sigma_s = sigmaArr[i]
    c = threshold([mu_d1, mu_d2, mu_d3, sigma_s])
    if lrtVar(c[0]-0.5, mu_d1, mu_d2, mu_d3, sigma_s) > 0:
        clen = len(c)
        acc = np.full((4,1), 0.5*(1+(-1)**clen))
        for j in range(clen):
            acc[3,0] = acc[3,0]+(norm(mu_n,sigma_n).cdf(c[j])*(-1)**(j))
            acc[0,0] = acc[0,0]+(norm(mu_d1,sigma_s).cdf(c[j])*(-1)**(j))
            acc[1,0] = acc[1,0]+(norm(mu_d2,sigma_s).cdf(c[j])*(-1)**(j))
            acc[2,0] = acc[2,0]+(norm(mu_d3,sigma_s).cdf(c[j])*(-1)**(j))
    
    elif lrtVar(c[0]-0.5, mu_d1, mu_d2, mu_d3, sigma_s) <= 0:
        clen = len(c)
        acc = np.full((4,1),0.5*(1+(-1)**(clen+1)))
        for j in range(clen):
            acc[3,0] = acc[3,0]+(norm(mu_n,sigma_n).cdf(c[j])*(-1)**(j+1))
            acc[0,0] = acc[0,0]+(norm(mu_d1,sigma_s).cdf(c[j])*(-1)**(j+1))
            acc[1,0] = acc[1,0]+(norm(mu_d2,sigma_s).cdf(c[j])*(-1)**(j+1))
            acc[2,0] = acc[2,0]+(norm(mu_d3,sigma_s).cdf(c[j])*(-1)**(j+1))
    
    acc[3,0] = 1 - acc[3,0]
    plt.plot(['25', '50', '500', '0'], acc, marker = 'o', label = 'sigma_s=%1.2f'%sigma_s)
    
    dataSim = np.full((n,2),0)            
    for i in range(n):
        dataSim[i] = trialVar(mu_n,sigma_n, mu_d1, mu_d2, mu_d3, sigma_s,c)    
    dat = dataSim
    accuracy = np.full((4, 1), 0.0)
    accuracy[0,0] = len(np.intersect1d(np.where(dat[:,1] == 25),np.where(dat[:,0] == 1)))/len(np.where(dat[:,1] == 25)[0])
    accuracy[1,0] = len(np.intersect1d(np.where(dat[:,1] == 50),np.where(dat[:,0] == 1)))/len(np.where(dat[:,1] == 50)[0])
    accuracy[2,0] = len(np.intersect1d(np.where(dat[:,1] == 500),np.where(dat[:,0] == 1)))/len(np.where(dat[:,1] == 500)[0])
    accuracy[3,0] = len(np.intersect1d(np.where(dat[:,1] == 0),np.where(dat[:,0] == 0)))/len(np.where(dat[:,1] == 0)[0])
    
    plt.plot(['25', '50', '500', '0'], accuracy, marker = 'o', linestyle = 'dashed')
    
    
    plt.legend(); plt.xlabel('signal duration'); plt.ylabel('accuracy'); plt.title('n=%d'%n)


#%%

n = 200; nIters = 1;
mu_d1 = 0.5;mu_d2 = 1.0; mu_d3 = 1.5; sigma_s = 1.5
c = threshold([mu_d1, mu_d2, mu_d3, sigma_s])

sigmaArr = [1.2,1.8]
meanArr = np.full((len(sigmaArr),nIters), 0.0, dtype = 'object'); 
varArr = np.full((len(sigmaArr),nIters), 0.0, dtype = 'object'); 
likeArr = np.full((len(sigmaArr),nIters), 0.0)

for i in range(len(sigmaArr)):
    sigma_s = sigmaArr[i]
    for j in range(nIters):
        dataSim = np.full((n,2),0)            
        for k in range(n):
            dataSim[k] = trialVar(mu_n,sigma_n, mu_d1, mu_d2, mu_d3, sigma_s,c)    
        dat = dataSim
        
        result = minimize(negloglikelihoodVar,[mu_d1, mu_d2, mu_d3, sigma_s], 
             bounds = [(None,None),(None,None), (None, None), (0.1,None)])
        
        meanArr[i,j] = result['x']; varArr[i,j] = result['hess_inv'].todense(); likeArr[i,j] = result['fun']
    
v = np.full((len(sigmaArr),nIters), 0.0); std = np.full((len(sigmaArr),nIters), 0.0)
for i in range(len(sigmaArr)):
    for j in range(nIters):
        v[i,j] = meanArr[i,j][3]
        std[i,j]=np.sqrt(varArr[i,j][3,3])
        
for i in range(nIters):
    plt.errorbar(sigmaArr, v[:,i], linestyle = "None", marker = 'o')
plt.plot(sigmaArr, np.mean(std, axis = 1), marker = 'o', label = 'avg std dev across fits')

plt.legend()
plt.xlabel('sigma_s'); plt.ylabel('estimate of sigma_s from fit'); plt.title('mu_d1 = 0.5, mu_d2 = 1, mu_d3 = 1.5, n=%d'%2000)
plt.figure()

    
#%%
nIters = 1; n= 200
sigmaArr = [0.5, 1.5]
meanArr = np.full((len(sigmaArr),1), 0.0); varArr = np.full((len(sigmaArr),1), 0.0); 
likeArr = np.full((len(sigmaArr),1), 0.0)

mu_d1 = 0.5;mu_d2 = 1.0; mu_d3 = 1.5; sigma_s = sigmaArr[i]
c = threshold([mu_d1, mu_d2, mu_d3, sigma_s])

for i in range(len(sigmaArr)):
    resultArr = np.full((nIters,4), 0.0, dtype = 'object')
    dataSim = np.full((n,2),0)            
    for k in range(n):
        dataSim[k] = trialVar(mu_n,sigma_n, mu_d1, mu_d2, mu_d3, sigma_s,c)    
    dat = dataSim
    
    for j in range(nIters):
        mu_d1 = np.random.uniform(0,5,1)
        mu_d2 = np.random.uniform(0,5,1); mu_d3 = np.random.uniform(0,5,1)
        sigma_s = np.random.uniform(0.1,4, 1)
        c = threshold([mu_d1, mu_d2, mu_d3, sigma_s])
        
        resultArr[j,2] = [mu_d1, mu_d2, mu_d3, sigma_s]
        result = minimize(negloglikelihoodVar,[mu_d1, mu_d2, mu_d3, sigma_s], 
            bounds = [(None,None),(None,None), (None, None), (0.1,None)])
        resultArr[j,0] = result['fun']; resultArr[j,1] = result['x']
        resultArr[j,3] = result['hess_inv'].todense()
        
    idx =  np.where(resultArr[:,0]==min(resultArr[:,0]))[0][0]
    meanArr[i,0] = resultArr[idx,1][3]; varArr[i,0]= np.sqrt(resultArr[idx,3][3,3])
    likeArr[i,0] = resultArr[idx,0]   

varArr = varArr.reshape(len(sigmaArr))
plt.errorbar(sigmaArr, meanArr, yerr = varArr, linestyle = "None", marker = 'o')

plt.legend()
plt.xlabel('sigma_s'); plt.ylabel('estimate of sigma_s from fit'); plt.title('mu_d1 = 0.5, mu_d2 = 1, mu_d3 = 1.5, n=%d'%200)
plt.figure()

#%%
def plots(dat,result,session):
    aReal = np.full((4, 1), 0.0)
    aReal[0,0] = len(np.intersect1d(np.where(dat[:,1] == 25),np.where(dat[:,0] == 1)))/len(np.where(dat[:,1] == 25)[0])
    aReal[1,0] = len(np.intersect1d(np.where(dat[:,1] == 50),np.where(dat[:,0] == 1)))/len(np.where(dat[:,1] == 50)[0])
    aReal[2,0] = len(np.intersect1d(np.where(dat[:,1] == 500),np.where(dat[:,0] == 1)))/len(np.where(dat[:,1] == 500)[0])
    aReal[3,0] = len(np.intersect1d(np.where(dat[:,1] == 0),np.where(dat[:,0] == 0)))/len(np.where(dat[:,1] == 0)[0])

    mu_d1 = result[0]; mu_d2 = result[1]; mu_d3 = result[2]; sigma_s = result[3]
    c = threshold([mu_d1, mu_d2, mu_d3, sigma_s])
    if lrtVar(c[0]-0.5, mu_d1, mu_d2, mu_d3, sigma_s) > 0:
        clen = len(c)
        acc = np.full((4,1), 0.5*(1+(-1)**clen))
        for j in range(clen):
            acc[3,0] = acc[3,0]+(norm(mu_n,sigma_n).cdf(c[j])*(-1)**(j))
            acc[0,0] = acc[0,0]+(norm(mu_d1,sigma_s).cdf(c[j])*(-1)**(j))
            acc[1,0] = acc[1,0]+(norm(mu_d2,sigma_s).cdf(c[j])*(-1)**(j))
            acc[2,0] = acc[2,0]+(norm(mu_d3,sigma_s).cdf(c[j])*(-1)**(j))

    elif lrtVar(c[0]-0.5, mu_d1, mu_d2, mu_d3, sigma_s) <= 0:
        clen = len(c)
        acc = np.full((4,1),0.5*(1+(-1)**(clen+1)))
        for j in range(clen):
            acc[3,0] = acc[3,0]+(norm(mu_n,sigma_n).cdf(c[j])*(-1)**(j+1))
            acc[0,0] = acc[0,0]+(norm(mu_d1,sigma_s).cdf(c[j])*(-1)**(j+1))
            acc[1,0] = acc[1,0]+(norm(mu_d2,sigma_s).cdf(c[j])*(-1)**(j+1))
            acc[2,0] = acc[2,0]+(norm(mu_d3,sigma_s).cdf(c[j])*(-1)**(j+1))

    acc[3,0] = 1 - acc[3,0]

    plt.plot(['25', '50', '500', '0'], aReal, marker = 'o', label = 'observed')
    plt.plot(['25', '50', '500', '0'], acc, marker = 'o', label = 'fit')
    plt.legend(); plt.xlabel('signal duration'); plt.ylabel('accuracy'); plt.title('n=%d'%session)
    
#%%
summaryData1 = pd.read_csv('summaryHoweRaw.csv')
summaryData2 = pd.read_csv('summary_data2.csv')
