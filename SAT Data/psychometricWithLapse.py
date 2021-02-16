import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

#%%
dataSDT = np.loadtxt("dataSDT.csv")
sessionNo = 1; #session number to fit
data = np.full((251,2), 0.0)
data[:,0]= dataSDT[251*(sessionNo-1):251*sessionNo,1] #response data
data[:,1]= dataSDT[251*(sessionNo-1):251*sessionNo,0] #signal duration data

data = data[~np.isnan(data).any(axis=1)] #drop nan rows
data = data.astype(int)

dat = data

#%%
def psychFun(x, param):
    alpha = param[0]; beta = param[1]; gamma = param[2]
    lambd = param[3]
    sigmoid = 1/(1+np.exp(-beta*x-alpha))
    psi = gamma+(1-gamma-lambd)*sigmoid #prob of response = 1

    return psi
    
#%%
#generate some data 
def trial(n,parameter):
    alpha = parameter[0]; beta = parameter[1]; gamma = parameter[2]
    lambd = parameter[3]
    x = np.full((n,1),0.0); y = np.full((n,1),0.0) 
    for i in range(n):
        j = np.random.binomial(1,0.5)
        if j == 1:
            r = np.random.choice(3, p = np.array([1,1,1])/3)
            if r == 0:
                x[i] = 0.025
            elif r == 1:
                x[i] = 0.050
            elif r == 2:
                x[i] = 0.500
                
        elif j == 0:
            x[i] = 0
            
        psi = psychFun(x[i],[alpha,beta,gamma,lambd])#prob of response = 1
        
        r = np.random.binomial(1,psi)
        y[i] = r
            
    return y,x

#%%
alpha = 0.2; beta = 5; gamma = 0.1; lambd = 0.1
alphaArr = [-3, -2, -1, 0, 1, 2, 3]
for i in range(len(alphaArr)):
    alpha = alphaArr[i]
    xval = np.linspace(-1,1,50)
    yval = psychFun(xval, [alpha,beta, gamma,lambd])
    plt.plot(xval, yval, label = 'alpha=%1.2f'%alpha)
plt.xlabel('signal duration'); plt.ylabel('prob of responding signal')
plt.title('beta=%1.2f, gamma=%1.2f, lambda=%1.2f'%(beta,gamma,lambd))
plt.legend(); plt.figure()

alpha = 0.2; beta = 5; gamma = 0.1; lambd = 0.1
betaArr = [-2, 2, 4, 6, 8, 10]
for i in range(len(betaArr)):
    beta = betaArr[i]
    xval = np.linspace(-1,1,50)
    yval = psychFun(xval, [alpha,beta, gamma,lambd])
    plt.plot(xval, yval, label = 'beta=%1.2f'%beta)
plt.xlabel('signal duration'); plt.ylabel('prob of responding signal')
plt.title('alpha=%1.2f, gamma=%1.2f, lambda=%1.2f'%(alpha,gamma,lambd))
plt.legend(); plt.figure()

alpha = 0.2; beta = 5; gamma = 0.1; lambd = 0.1
gammaArr = [0.1, 0.3, 0.5, 0.7, 0.9]
for i in range(len(gammaArr)):
    gamma = gammaArr[i]
    xval = np.linspace(-1,1,50)
    yval = psychFun(xval, [alpha,beta, gamma,lambd])
    plt.plot(xval, yval, label = 'gamma=%1.2f'%gamma)
plt.xlabel('signal duration'); plt.ylabel('prob of responding signal')
plt.title('alpha=%1.2f, beta=%1.2f, lambda=%1.2f'%(alpha,beta,lambd))
plt.legend(); plt.figure()

alpha = 0.2; beta = 5; gamma = 0.1; lambd = 0.1
lambdArr = [0.1, 0.3, 0.5, 0.7, 0.9]
for i in range(len(lambdArr)):
    lambd = lambdArr[i]
    xval = np.linspace(-1,1,50)
    yval = psychFun(xval, [alpha,beta, gamma,lambd])
    plt.plot(xval, yval, label = 'lambda=%1.2f'%lambd)
plt.xlabel('signal duration'); plt.ylabel('prob of responding signal')
plt.title('alpha=%1.2f, beta=%1.2f, gamma=%1.2f'%(alpha,beta,gamma))
plt.legend(); plt.figure()

#%%
#observed vs expected psychometric curves
alpha = 0.2; beta = 5; gamma = 0.1; lambd = 0.2
nArr = [200,1000,2000]
for k in range(len(nArr)):
    n = nArr[k]
    alphaArr = [-3, -1, 0, 1, 3]
    for j in range(len(alphaArr)):
        alpha = alphaArr[j]; 
        accuracy = np.full((10, 4), 0.0)
        for i in range(10):
            dat = np.full((n,2),0.0)
            y,x = trial(n, [alpha,beta, gamma,lambd])    
            dat[:,0] = y[:,0]; dat[:,1] = x[:,0] 
        
            accuracy[i,0] = 1-len(np.intersect1d(np.where(dat[:,1] == 0),np.where(dat[:,0] == 0)))/len(np.where(dat[:,1] == 0)[0])
            accuracy[i,1] = len(np.intersect1d(np.where(dat[:,1] == 0.025),np.where(dat[:,0] == 1)))/len(np.where(dat[:,1] == 0.025)[0])
            accuracy[i,2] = len(np.intersect1d(np.where(dat[:,1] == 0.050),np.where(dat[:,0] == 1)))/len(np.where(dat[:,1] == 0.050)[0])
            accuracy[i,3] = len(np.intersect1d(np.where(dat[:,1] == 0.500),np.where(dat[:,0] == 1)))/len(np.where(dat[:,1] == 0.500)[0])
        
        xval = np.linspace(0,0.6,50)
        yval = psychFun(xval, [alpha,beta, gamma,lambd])
        
        plt.plot(xval, yval, label = 'expected, alpha=%1.2f'%alpha)
        plt.errorbar([0, 0.025, 0.05, 0.5], np.mean(accuracy, axis =0),np.std(accuracy, axis =0), marker = 'o', linestyle = 'dashed')
        plt.xlabel('signal duration'); plt.ylabel('prob of responding to signal')
        plt.legend()
    plt.title('n=%1.2f,beta=%1.2f,gamma=%1.2f,lambda%1.2f'%(n,beta,gamma,lambd))
    plt.figure()

#%%
def negloglikelihood(parameter):
    alpha = parameter[0]; beta = parameter[1]; gamma = parameter[2]
    lambd = parameter[3]
    L = 0
    for i in range(len(dat)):
        x = dat[i,1]
        sigmoid = 1/(1+np.exp(-beta*x-alpha))
        psi = gamma+(1-gamma-lambd)*sigmoid #prob of response = 1
        
        if dat[i,0] == 1:
            L = L-np.log(psi)
        elif dat[i,0] == 0:
            L = L-np.log(1-psi)
        
        
        if np.isnan(L) or np.isinf(L):
            print(L, alpha, beta, gamma, lambd)
            break
    
    return L

#%%
alpha = -0.2; beta = 10; gamma = 0.1; lambd = 0.2
dat = np.full((2000,2),0.0)
y,x = trial(2000, [alpha,beta, gamma,lambd])    
dat[:,0] = y[:,0]; dat[:,1] = x[:,0] 

alpha = -0.2; beta = 10; gamma = 0.1; lambd = 0.2
xval = np.linspace(0,1,50)
yval = psychFun(xval, [alpha,beta, gamma,lambd],)
plt.plot(xval, yval, label = 'actual')

result = minimize(negloglikelihood,[alpha,beta, gamma,lambd], 
     bounds = [(None,None),(None,None), (0,0.9), (0,0.9)])

par = result['x']
xval = np.linspace(0,1,50)
yval = psychFun(xval, par)
plt.plot(xval, yval, label = 'fit')
plt.legend()
