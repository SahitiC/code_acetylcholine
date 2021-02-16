#importing required packages
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.stats

#%%
#function to calculate prob of X_j = 0,1 given Y,I, j+1 (no. of samples) and 
#fixed interval at the end of a trial
def probX_YI(Y,I,nSamples,interval):
    fp = int(interval/dt) #fixed period
    Xp0 = 0.0; Xp1 = 0.0 #prob(X=0,1|Y,I) respectively 
    if Y == 0: 
        Xp1 = 0.0
        if nSamples<=I+fp: Xp0 = 1.0
        elif nSamples > I+fp: Xp0 = 0.0
    else: 
        if nSamples<=I: Xp1 = 0.0; Xp0 = 1.0
        elif I<nSamples<=Y+I: Xp1 = 1.0; Xp0 = 0.0
        elif Y+I<nSamples<=Y+I+fp: Xp1 = 0.0; Xp0 = 1.0
        elif nSamples>Y+I+fp: Xp0 = 0.0; Xp1 = 0.0
    return Xp0,Xp1

def probX_YIitu(Y,I,nSamples):
    Xp0 = 0.0; Xp1 = 0.0 #prob(X=0,1|Y,I) respectively 
    if Y == 0: 
        Xp1 = 0.0
        Xp0 = 1.0
    else: 
        if nSamples<=I: Xp1 = 0.0; Xp0 = 1.0
        elif I<nSamples<=Y+I: Xp1 = 1.0; Xp0 = 0.0
        elif Y+I<nSamples: Xp1 = 0.0; Xp0 = 1.0
    return Xp0,Xp1

#generate a trial sequence -- discrete observations
def trialDiscrete(I,Y,Z,dt,eta,interval):
    n = int(1/dt)#time as multiple of dt
    X = np.full((int(round(I*n+Y*n+interval*n)),1),0);#signal values within a trial 
    if Z == 1:
        X[int(round(I*n)):int(round(I*n+Y*n))] = np.full((int((Y)*n),1),1)
    W = np.full((int((I*n+Y*n+interval*n)),1),0);#signal observations within a trial
    for i in range(len(W)):
        j = np.random.binomial(1,eta)
        if j == 1: W[i] = X[i]
        elif j == 0: W[i] = 1 - X[i]
    return X,W

def trialContinuous(I,Y,Z,dt,mu_n,mu_s,sigma_n,sigma_s,interval):
    n = int(1/dt)#time as multiple of dt
    X = np.full((round((I*n+Y*n+interval*n)),1),0);#signal values within a trial 
    if Z == 1:
        X[round(I*n):round(I*n+Y*n)] = np.full((round((Y)*n),1),1)
    W = np.full((round((I*n+Y*n+interval*n)),1),0.0);#signal observations within a trial
    for i in range(len(W)):
        if X[i] == 1: W[i] = np.random.normal(mu_s,sigma_s)
        elif X[i] == 0: W[i] = np.random.normal(mu_n,sigma_n)
    return X,W

#%%
#calculating full joint posterior (forward inference) -- discrete observations
def inferenceDiscrete(p_signal,iti_lb,iti_ub, duration1, duration2,
                          duration3, dt, p_dur, eta, scale,interval): 
    iti_count = int((iti_ub-iti_lb)/dt)+1 #total number of possible iti's
    probWX = np.array([[eta, 1-eta], [1-eta, eta]])
    pZ = [1-p_signal, p_signal]
    #array for joint posterior of Z,Y,I,Xj given W1:j 
    posterior = np.full((2,4,iti_count,2,len(W)+1),0.0)
    posteriorScaled = np.full((2,4,iti_count,2,len(W)+1),0.0) 
    fp = int(interval/dt)
    
    z = [0,1]; y = [0,int(duration1/dt),int(duration2/dt),int(duration3/dt)]; 
    iti=np.arange(0,iti_count,1); x = [0,1]
    
    #starting posterior of Z,Y,I,X0 before trial begins
    for zi in range(len(z)):
        for yi in range(len(y)):
            for ti in range(len(iti)):
                for xi in range(len(x)):
                    if z[zi] == 0: pY_Z = [1,0,0,0]
                    else: pY_Z = [0, 1/3, 1/3, 1/3]
                    posterior[zi,yi,ti,xi,0] = probX_YI(y[yi],iti[ti],0,
                    interval)[xi]*pY_Z[yi]*(1/iti_count)*pZ[zi]
                    #is equal to pX|YI * pY|Z * PI * PZ
                    
    posteriorScaled[:,:,:,:,0] = posterior[:,:,:,:,0]/(np.sum(posterior[:,:,:,:,0]))
    
    for j in range(len(W)):
        for zi in range(len(z)):
            for yi in range(len(y)):
                if j+1< y[yi]:
                    posterior[zi,yi,0:j+1,0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],0,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,0:j+1,:,j], axis =1) 
                    posterior[zi,yi,0:j+1,1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],0,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,0:j+1,:,j], axis =1) 
                    
                    posterior[zi,yi,j+1:1201,0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],j+1,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,j+1:1201,:,j], axis =1) 
                    posterior[zi,yi,j+1:1201,1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],j+1,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,j+1:1201,:,j], axis =1) 
                    
                elif y[yi]+fp > j+1 >= y[yi]:
                    posterior[zi,yi,0:j+1-y[yi],0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],0,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,0:j+1-y[yi],:,j], axis =1) 
                    posterior[zi,yi,0:j+1-y[yi],1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],0,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,0:j+1-y[yi],:,j], axis =1) 
                    
                    posterior[zi,yi,j+1-y[yi]:j+1,0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],j+1-y[yi],j+1,
                    interval)[0]*np.sum(posterior[zi,yi,j+1-y[yi]:j+1,:,j], axis =1) 
                    posterior[zi,yi,j+1-y[yi]:j+1,1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],j+1-y[yi],j+1,
                    interval)[1]*np.sum(posterior[zi,yi,j+1-y[yi]:j+1,:,j], axis =1) 
    
                    
                    posterior[zi,yi,j+1:1201,0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],j+1,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,j+1:1201,:,j], axis =1) 
                    posterior[zi,yi,j+1:1201,1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],j+1,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,j+1:1201,:,j], axis =1)
                    
                elif j+1 >= y[yi]+fp:
                    posterior[zi,yi,0:j+1-y[yi]-fp,0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],0,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,0:j+1-y[yi]-fp,:,j], axis =1) 
                    posterior[zi,yi,0:j+1-y[yi]-fp,1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],0,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,0:j+1-y[yi]-fp,:,j], axis =1) 
                    
                    posterior[zi,yi,j+1-y[yi]-fp:j+1-y[yi],0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],j+1-y[yi]-fp,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,j+1-y[yi]-fp:j+1-y[yi],:,j], axis =1) 
                    posterior[zi,yi,j+1-y[yi]-fp:j+1-y[yi],1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],j+1-y[yi]-fp,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,j+1-y[yi]-fp:j+1-y[yi],:,j], axis =1) 
    
                    posterior[zi,yi,j+1-y[yi]:j+1,0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],j+1-y[yi],j+1,
                    interval)[0]*np.sum(posterior[zi,yi,j+1-y[yi]:j+1,:,j], axis =1) 
                    posterior[zi,yi,j+1-y[yi]:j+1,1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],j+1-y[yi],j+1,
                    interval)[1]*np.sum(posterior[zi,yi,j+1-y[yi]:j+1,:,j], axis =1) 
                    
                    posterior[zi,yi,j+1:1201,0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],j+1,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,j+1:1201,:,j], axis =1) 
                    posterior[zi,yi,j+1:1201,1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],j+1,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,j+1:1201,:,j], axis =1)
                    
                posterior[zi,yi,:,:,j+1] = scale*posterior[zi,yi,:,:,j+1]                
                                
        posteriorScaled[:,:,:,:,j+1] = posterior[:,:,:,:,j+1]/(np.sum(posterior[:,:,:,:,j+1]))
    
    return posteriorScaled    

#calculating full joint posterior (forward inference) -- continuous observations
def inferenceContinuous(p_signal,iti_lb,iti_ub, duration1, duration2,
            duration3, dt, p_dur, mu_n, mu_s, sigma_s, sigma_n, scale,interval):
    iti_count = int((iti_ub-iti_lb)/dt)+1 #total number of possible iti's
    paramWX = np.array([[mu_n, sigma_n], [mu_s, sigma_s]])
    pZ = [1-p_signal, p_signal]
    fp = int(interval/dt)
    
    #array for joint posterior of Z,Y,I,Xj given W1:j 
    posterior = np.full((2,4,iti_count,2,len(W)+1),0.0)
    posteriorScaled = np.full((2,4,iti_count,2,len(W)+1),0.0) 
    
    #starting posterior of Z,Y,I,X0 before trial begins
    z = [0,1]; y = [0,int(duration1/dt),int(duration2/dt),int(duration3/dt)]; 
    iti=np.arange(0,iti_count,1); x = [0,1]
    
    for zi in range(len(z)):
        for yi in range(len(y)):
            for ti in range(len(iti)):
                for xi in range(len(x)):
                    if z[zi] == 0: pY_Z = [1,0,0,0]
                    else: pY_Z = [0, 1/3, 1/3, 1/3]
                    posterior[zi,yi,ti,xi,0] = probX_YI(y[yi],iti[ti],0,
                    interval)[xi]*pY_Z[yi]*(1/iti_count)*pZ[zi]
                    #is equal to pX|YI * pY|Z * PI * PZ
                    
    posteriorScaled[:,:,:,:,0] = posterior[:,:,:,:,0]/(np.sum(posterior[:,:,:,:,0]))

    for j in range(len(W)):
        probWX0 = scipy.stats.norm(paramWX[0,0],paramWX[0,1]).pdf(W[j])
        probWX1 = scipy.stats.norm(paramWX[1,0],paramWX[1,1]).pdf(W[j])
        for zi in range(len(z)):
            for yi in range(len(y)):
                if j+1< y[yi]:
                    posterior[zi,yi,0:j+1,0,j+1] = probWX0*probX_YI(y[yi],0,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,0:j+1,:,j], axis =1) 
                    posterior[zi,yi,0:j+1,1,j+1] = probWX1*probX_YI(y[yi],0,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,0:j+1,:,j], axis =1) 
                    
                    posterior[zi,yi,j+1:1201,0,j+1] = probWX0*probX_YI(y[yi],j+1,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,j+1:1201,:,j], axis =1) 
                    posterior[zi,yi,j+1:1201,1,j+1] = probWX1*probX_YI(y[yi],j+1,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,j+1:1201,:,j], axis =1) 
                    
                elif y[yi]+fp > j+1 >= y[yi]:
                    posterior[zi,yi,0:j+1-y[yi],0,j+1] = probWX0*probX_YI(y[yi],0,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,0:j+1-y[yi],:,j], axis =1) 
                    posterior[zi,yi,0:j+1-y[yi],1,j+1] = probWX1*probX_YI(y[yi],0,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,0:j+1-y[yi],:,j], axis =1) 
                    
                    posterior[zi,yi,j+1-y[yi]:j+1,0,j+1] = probWX0*probX_YI(y[yi],j+1-y[yi],j+1,
                    interval)[0]*np.sum(posterior[zi,yi,j+1-y[yi]:j+1,:,j], axis =1) 
                    posterior[zi,yi,j+1-y[yi]:j+1,1,j+1] = probWX1*probX_YI(y[yi],j+1-y[yi],j+1,
                    interval)[1]*np.sum(posterior[zi,yi,j+1-y[yi]:j+1,:,j], axis =1) 
    
                    
                    posterior[zi,yi,j+1:1201,0,j+1] = probWX0*probX_YI(y[yi],j+1,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,j+1:1201,:,j], axis =1) 
                    posterior[zi,yi,j+1:1201,1,j+1] = probWX1*probX_YI(y[yi],j+1,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,j+1:1201,:,j], axis =1)
                    
                elif j+1 >= y[yi]+fp:
                    posterior[zi,yi,0:j+1-y[yi]-fp,0,j+1] = probWX0*probX_YI(y[yi],0,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,0:j+1-y[yi]-fp,:,j], axis =1) 
                    posterior[zi,yi,0:j+1-y[yi]-fp,1,j+1] = probWX1*probX_YI(y[yi],0,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,0:j+1-y[yi]-fp,:,j], axis =1) 
                    
                    posterior[zi,yi,j+1-y[yi]-fp:j+1-y[yi],0,j+1] = probWX0*probX_YI(y[yi],j+1-y[yi]-fp,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,j+1-y[yi]-fp:j+1-y[yi],:,j], axis =1) 
                    posterior[zi,yi,j+1-y[yi]-fp:j+1-y[yi],1,j+1] = probWX1*probX_YI(y[yi],j+1-y[yi]-fp,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,j+1-y[yi]-fp:j+1-y[yi],:,j], axis =1) 
    
                    posterior[zi,yi,j+1-y[yi]:j+1,0,j+1] = probWX0*probX_YI(y[yi],j+1-y[yi],j+1,
                    interval)[0]*np.sum(posterior[zi,yi,j+1-y[yi]:j+1,:,j], axis =1) 
                    posterior[zi,yi,j+1-y[yi]:j+1,1,j+1] = probWX1*probX_YI(y[yi],j+1-y[yi],j+1,
                    interval)[1]*np.sum(posterior[zi,yi,j+1-y[yi]:j+1,:,j], axis =1) 
                    
                    posterior[zi,yi,j+1:1201,0,j+1] = probWX0*probX_YI(y[yi],j+1,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,j+1:1201,:,j], axis =1) 
                    posterior[zi,yi,j+1:1201,1,j+1] = probWX1*probX_YI(y[yi],j+1,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,j+1:1201,:,j], axis =1)
                    
                posterior[zi,yi,:,:,j+1] = scale*posterior[zi,yi,:,:,j+1]                
                                
        posteriorScaled[:,:,:,:,j+1] = posterior[:,:,:,:,j+1]/(np.sum(posterior[:,:,:,:,j+1]))
    
    return posteriorScaled

#%%
#inference when ITU is fixed
def inferenceDiscreteITU(p_signal,iti_lb,iti_ub, duration1, duration2,
                          duration3, dt, p_dur, eta, scale): 
    iti_count = int((iti_ub-iti_lb)/dt)+1 #total number of possible iti's
    probWX = np.array([[eta, 1-eta], [1-eta, eta]])
    pZ = [1-p_signal, p_signal]
    #array for joint posterior of Z,Y,I,Xj given W1:j 
    posterior = np.full((2,4,iti_count,2,len(W)+1),0.0)
    posteriorScaled = np.full((2,4,iti_count,2,len(W)+1),0.0) 
    
    z = [0,1]; y = [0,int(duration1/dt),int(duration2/dt),int(duration3/dt)]; 
    iti=np.arange(0,iti_count,1); x = [0,1]
    
    #starting posterior of Z,Y,I,X0 before trial begins
    for zi in range(len(z)):
        for yi in range(len(y)):
            for ti in range(len(iti)):
                for xi in range(len(x)):
                    if z[zi] == 0: pY_Z = [1,0,0,0]
                    else: pY_Z = [0, 1/3, 1/3, 1/3]
                    posterior[zi,yi,ti,xi,0] = probX_YI(y[yi],iti[ti],0,
                    interval)[xi]*pY_Z[yi]*(1/iti_count)*pZ[zi]
                    #is equal to pX|YI * pY|Z * PI * PZ
                    
    posteriorScaled[:,:,:,:,0] = posterior[:,:,:,:,0]/(np.sum(posterior[:,:,:,:,0]))
    
    for j in range(len(W)):
        for zi in range(len(z)):
            for yi in range(len(y)):
                if j+1< y[yi]:
                    posterior[zi,yi,0:j+1,0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],0,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,0:j+1,:,j], axis =1) 
                    posterior[zi,yi,0:j+1,1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],0,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,0:j+1,:,j], axis =1) 
                    
                    posterior[zi,yi,j+1:1201,0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],j+1,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,j+1:1201,:,j], axis =1) 
                    posterior[zi,yi,j+1:1201,1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],j+1,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,j+1:1201,:,j], axis =1) 
                    
                elif y[yi]+fp > j+1 >= y[yi]:
                    posterior[zi,yi,0:j+1-y[yi],0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],0,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,0:j+1-y[yi],:,j], axis =1) 
                    posterior[zi,yi,0:j+1-y[yi],1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],0,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,0:j+1-y[yi],:,j], axis =1) 
                    
                    posterior[zi,yi,j+1-y[yi]:j+1,0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],j+1-y[yi],j+1,
                    interval)[0]*np.sum(posterior[zi,yi,j+1-y[yi]:j+1,:,j], axis =1) 
                    posterior[zi,yi,j+1-y[yi]:j+1,1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],j+1-y[yi],j+1,
                    interval)[1]*np.sum(posterior[zi,yi,j+1-y[yi]:j+1,:,j], axis =1) 
    
                    
                    posterior[zi,yi,j+1:1201,0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],j+1,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,j+1:1201,:,j], axis =1) 
                    posterior[zi,yi,j+1:1201,1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],j+1,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,j+1:1201,:,j], axis =1)
                    
                elif j+1 >= y[yi]+fp:
                    posterior[zi,yi,0:j+1-y[yi]-fp,0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],0,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,0:j+1-y[yi]-fp,:,j], axis =1) 
                    posterior[zi,yi,0:j+1-y[yi]-fp,1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],0,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,0:j+1-y[yi]-fp,:,j], axis =1) 
                    
                    posterior[zi,yi,j+1-y[yi]-fp:j+1-y[yi],0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],j+1-y[yi]-fp,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,j+1-y[yi]-fp:j+1-y[yi],:,j], axis =1) 
                    posterior[zi,yi,j+1-y[yi]-fp:j+1-y[yi],1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],j+1-y[yi]-fp,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,j+1-y[yi]-fp:j+1-y[yi],:,j], axis =1) 
    
                    posterior[zi,yi,j+1-y[yi]:j+1,0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],j+1-y[yi],j+1,
                    interval)[0]*np.sum(posterior[zi,yi,j+1-y[yi]:j+1,:,j], axis =1) 
                    posterior[zi,yi,j+1-y[yi]:j+1,1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],j+1-y[yi],j+1,
                    interval)[1]*np.sum(posterior[zi,yi,j+1-y[yi]:j+1,:,j], axis =1) 
                    
                    posterior[zi,yi,j+1:1201,0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],j+1,j+1,
                    interval)[0]*np.sum(posterior[zi,yi,j+1:1201,:,j], axis =1) 
                    posterior[zi,yi,j+1:1201,1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],j+1,j+1,
                    interval)[1]*np.sum(posterior[zi,yi,j+1:1201,:,j], axis =1)
                    
                posterior[zi,yi,:,:,j+1] = scale*posterior[zi,yi,:,:,j+1]                
                                
        posteriorScaled[:,:,:,:,j+1] = posterior[:,:,:,:,j+1]/(np.sum(posterior[:,:,:,:,j+1]))
    
    return posteriorScaled  
#%%
#generate a trial -- discrete observations

#parameters to generate trial
p_signal = 0.5; iti_lb = 0; iti_ub = 6 #prob of signal trial, lower and upper bounds for iti
duration1 = 0.025; duration2 = 0.05; duration3 = 0.5; dt = 0.005 #3 signal durations, time step 
p_dur = [1/3,1/3,1/3]; eta = 0.7 #probability of each duration on a signal trial, confusability
interval = 2.0 #post signal fixed interval


#generate Z,Y,I
Z = 0; Y = 0; I = 0; #variables for signal type, duration, ITI
i = np.random.binomial(1,p_signal) #Z=1 on signal trial and 0 otherwise:
if i == 1: Z=1
elif i == 0: Z=0
if Z == 1: #if Z=1, non-zero Y:
    j = np.random.choice(3, p = np.array(p_dur))
    if j == 0: Y = duration1
    elif j == 1: Y = duration2
    elif j == 2: Y = duration3
elif Z==0: Y = 0; #if Z=0, Y=0
#uniform probability of choosing I from ITI interval:
I = np.random.choice(np.arange(iti_lb,iti_ub+dt,dt)) 

I = 4; Y = duration2; Z =1


start = time.perf_counter()
#generate trial sequence (X,W)
X,W = trialDiscrete(I,Y,Z,dt,eta,interval)

#calculating full joint posterior (forward inference) -- discrete observations
#parameters 'known' by agent:
p_signal = 0.5; iti_lb = 0; iti_ub = 6 #prob of signal trial, lower and upper bounds for iti
duration1 = 0.025; duration2 = 0.05; duration3 = 0.5; dt = 0.005 #3 signal durations, time step 
p_dur = [1/3,1/3,1/3]; eta = 0.7 #probability of each duration on a signal trial, confusability
scale = 2; #scaling to prevent underflow
interval = 2.0; #fixed period at the end
#inference:
posteriorScaled = inferenceDiscrete(p_signal,iti_lb,iti_ub, duration1, duration2,
                          duration3, dt, p_dur, eta, scale, interval)

response = 2; #response generated by the animal
a = posteriorScaled; 
a = np.sum(a, axis = 1); a = np.sum(a, axis = 1); a = np.sum(a, axis = 1)
if a[0,-1] > a[1,-1]: response=0
elif a[0,-1] < a[1,-1]: response =1

print(time.perf_counter()-start)

#%%
#plotting Xj,Wj and posteriors over Z,I,Y,X

t = np.arange(0,len(X)*dt,dt)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.scatter(t, W, label = 'Wj', s=10); ax1.scatter(t, X, label='Xj', s=10)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
a = posteriorScaled; 
a = np.sum(a, axis = 1); a = np.sum(a, axis = 1); a = np.sum(a, axis = 1)
ax2.plot(t, a[1,1:], label = 'P(Z=1|W)', color = 'red')
a = posteriorScaled; 
a = np.sum(a, axis = 0); a = np.sum(a, axis = 0); a = np.sum(a, axis = 0)
ax2.plot(t, a[1,1:], label = 'P(X=1|W)', color = 'green')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); 
ax1.set_title('Y=%1.3fs,I=%1.3fs,eta=%1.2f,interval=%1.1fs'%(Y,I,eta,interval))
ax2.set_xlabel('time in sec')

"""
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
c = posteriorScaled;
c = np.sum(c, axis = 0); c = np.sum(c, axis = 0); c = np.sum(c, axis = 1)
ax1.plot(t, c[1, 1:], label = 'iti = 0.005s'); 
ax1.plot(t, c[200, 1:], label = 'iti = 1s'); ax1.plot(t, c[500, 1:], label = 'iti = 2.5s'); 
ax1.plot(t, c[1000, 1:], label = 'iti = 5s'); 
#ax1.plot(t, c[1200, 1:], label = 'iti = 6s');
ax1.legend(); ax1.set_ylabel('posterior prob of I')
#ax2.plot(t, c[int(I/dt)+1, 1:], label = 'iti =%1.3fs'%(I+dt))
ax2.plot(t, c[int(I/dt), 1:], label = 'iti =%1.3fs'%(I))
#ax2.plot(t, c[int(I/dt)+2, 1:], label = 'iti =%1.3fs'%(I-dt))
ax2.legend(); ax2.set_xlabel('time (in sec)'); ax2.set_ylabel('posterior prob of I')
"""
plt.figure()
c = posteriorScaled;
c = np.sum(c, axis = 0); c = np.sum(c, axis = 0); c = np.sum(c, axis = 1)
plt.plot(t, np.sum(c[1:, 1:], axis =0), label = 'i = 0.005s'); 
plt.plot(t, np.sum(c[200:, 1:], axis =0), label = 'i = 1s'); 
plt.plot(t, np.sum(c[500:, 1:], axis =0), label = 'i = 2.5s'); 
plt.plot(t, np.sum(c[1000:, 1:], axis =0), label = 'i = 5s'); 
plt.plot(t, np.sum(c[1200:, 1:], axis =0), label = 'i = 6s');
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); 
plt.ylabel('posterior probability of I >= i');
plt.xlabel('time (in sec)')

plt.figure()
c = posteriorScaled;
c = np.sum(c, axis = 0); c = np.sum(c, axis = 1); c = np.sum(c, axis = 1)
plt.plot(t, c[0, 1:], label = 'dur = 0ms'); plt.plot(t, c[1,1:], label = 'dur = 25ms') 
plt.plot(t, c[2, 1:], label = 'dur = 50ms'); plt.plot(t, c[3,1:], label = 'dur = 500ms')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); 
plt.xlabel('time (in sec)'); plt.ylabel('posterior prob of Y')


#%%
#generate trial -- continuous observations

#parameters to generate trial
p_signal = 0.5; iti_lb = 0; iti_ub = 6 #prob of signal trial, lower and upper bounds for iti
duration1 = 0.025; duration2 = 0.05; duration3 = 0.5; dt = 0.005 #3 signal durations, time step 
p_dur = [1/3,1/3,1/3]; #prob of each signal duration
mu_n = 0; mu_s = 1 #means of signal and noise gaussians
sigma_n = 1; sigma_s = 1 #standard dev of signal and noise gaussians
interval = 0

#generate Z,Y,I
#uniform probability of choosing I from ITI interval:
I = np.random.choice(np.arange(iti_lb,iti_ub+dt,dt)) 
Z = 0; Y = 0; I = 0; #variables for signal type, duration, ITI
i = np.random.binomial(1,p_signal) #Z=1 on signal trial and 0 otherwise
if i == 1: Z=1
elif i == 0: Z=0
if Z == 1: #if Z=1, non-zero Y
    j = np.random.choice(3, p = np.array(p_dur))
    if j == 0: Y = duration1
    elif j == 1: Y = duration2
    elif j == 2: Y = duration3
elif Z==0: Y = 0; #if Z=0, Y=0

X,W = trialContinuous(I,Y,Z,dt,mu_n,mu_s,sigma_n,sigma_s,interval)

#calculating full joint posterior (forward inference) -- continuous observations
#parameters
p_signal = 0.5; iti_lb = 0; iti_ub = 6 #prob of signal trial, lower and upper bounds for iti
duration1 = 0.025; duration2 = 0.05; duration3 = 0.5; dt = 0.005 #3 signal durations, time step 
p_dur = [1/3,1/3,1/3]; eta = 1.0 #probability of each duration on a signal trial, confusability
mu_n = 0; mu_s = 1 #means of signal and noise gaussians
sigma_n = 1; sigma_s = 1 #standard dev of signal and noise gaussians
interval = 1 #post signal fixed interval
fp = int(interval/dt)
scale = 5 #scaling to prvent underflow
#inference
posteriorScaled = inferenceContinuous(p_signal,iti_lb,iti_ub, duration1, duration2,
            duration3, dt, p_dur, mu_n, mu_s, sigma_s, sigma_n, scale,interval)

response = 2; #response generated by the animal
a = posteriorScaled; 
a = np.sum(a, axis = 1); a = np.sum(a, axis = 1); a = np.sum(a, axis = 1)
if a[0,-1] > a[1,-1]: response=0
elif a[0,-1] < a[1,-1]: response =1

#%%
#generate trial-fixed ITU

#parameters to generate trial
p_signal = 0.5; iti_lb = 0; iti_ub = 6 #prob of signal trial, lower and upper bounds for iti
duration1 = 0.025; duration2 = 0.05; duration3 = 0.5; dt = 0.005 #3 signal durations, time step 
p_dur = [1/3,1/3,1/3]; eta = 1.0 #probability of each duration on a signal trial, confusability


#generate Z,Y,I
Z = 0; Y = 0; I = 0; #variables for signal type, duration, ITI
i = np.random.binomial(1,p_signal) #Z=1 on signal trial and 0 otherwise:
if i == 1: Z=1
elif i == 0: Z=0
if Z == 1: #if Z=1, non-zero Y:
    j = np.random.choice(3, p = np.array(p_dur))
    if j == 0: Y = duration1
    elif j == 1: Y = duration2
    elif j == 2: Y = duration3
elif Z==0: Y = 0; #if Z=0, Y=0
#uniform probability of choosing I from ITI interval:
I = np.random.choice(np.arange(iti_lb,iti_ub+dt,dt))
 
I = 6; Y = duration3; Z=1

interval = 6.5-I-Y #post signal interval
#generate trial sequence (X,W)
X,W = trialDiscrete(I,Y,Z,dt,eta,interval)


#inference
p_signal = 0.5; iti_lb = 0; iti_ub = 6 #prob of signal trial, lower and upper bounds for iti
duration1 = 0.025; duration2 = 0.05; duration3 = 0.5; dt = 0.005 #3 signal durations, time step 
p_dur = [1/3,1/3,1/3]; eta = 1.0 #probability of each duration on a signal trial, confusability
scale = 1; #scaling to prevent underflow

iti_count = int((iti_ub-iti_lb)/dt)+1 #total number of possible iti's
probWX = np.array([[eta, 1-eta], [1-eta, eta]])
pZ = [1-p_signal, p_signal]
#array for joint posterior of Z,Y,I,Xj given W1:j 
posterior = np.full((2,4,iti_count,2,len(W)+1),0.0)
posteriorScaled = np.full((2,4,iti_count,2,len(W)+1),0.0) 

z = [0,1]; y = [0,int(duration1/dt),int(duration2/dt),int(duration3/dt)]; 
iti=np.arange(0,iti_count,1); x = [0,1]

#starting posterior of Z,Y,I,X0 before trial begins
for zi in range(len(z)):
    for yi in range(len(y)):
        for ti in range(len(iti)):
            for xi in range(len(x)):
                if z[zi] == 0: pY_Z = [1,0,0,0]
                else: pY_Z = [0, 1/3, 1/3, 1/3]
                posterior[zi,yi,ti,xi,0] = probX_YIitu(y[yi],iti[ti],0
                )[xi]*pY_Z[yi]*(1/iti_count)*pZ[zi]
                #is equal to pX|YI * pY|Z * PI * PZ
 

for j in range(len(W)):
    for zi in range(len(z)):
        for yi in range(len(y)):
            for ti in range(len(iti)):                   
                #xi = 0
                posterior[zi,yi,ti,0,j+1] = probWX[W[j],0][0]*probX_YIitu(y[yi],iti[ti],j+1
                )[0]*np.sum(posterior[zi,yi,ti,:,j])
                #xi = 1
                posterior[zi,yi,ti,1,j+1] = probWX[W[j],1][0]*probX_YIitu(y[yi],iti[ti],j+1
                )[1]*np.sum(posterior[zi,yi,ti,:,j])
    posteriorScaled[:,:,:,:,j+1] = posterior[:,:,:,:,j+1]/(np.sum(posterior[:,:,:,:,j+1]))

 
#%%
#multiple trials : discrete

#parameters to generate trial and known by agent
p_signal = 0.5; iti_lb = 0; iti_ub = 6 #prob of signal trial, lower and upper bounds for iti
duration1 = 0.025; duration2 = 0.05; duration3 = 0.5; dt = 0.005 #3 signal durations, time step 
p_dur = [1/3,1/3,1/3]; eta = 0.7 #probability of each duration on a signal trial, confusability
interval = 2 #post signal fixed interval
scale = 2; #scaling to prevent underflow
nTrials = 1000 #no. of trials to run


#trialTypeRates = np.full((len(etaTrial),11),0.0)

posternTrials = np.full((nTrials,2),0.0); responsenTrials = np.full((nTrials,1),0.0)
ZnTrials = np.full((nTrials,1),1)#Z 
IYnTrials = np.full((nTrials,2),0.0)#I,Y

start = time.perf_counter()
for i in range(nTrials):
    #generate Z,Y,I
    Z = 0; Y = 0.0; I = 0.0; #variables for signal type, duration, ITI
    Z = np.random.binomial(1,p_signal) #Z=1 on signal trial and 0 otherwise:
    ZnTrials[i,0] = Z
    if Z == 1: #if Z=1, non-zero Y:
        j = np.random.choice(3, p = np.array(p_dur))
        if j == 0: Y = duration1
        elif j == 1: Y = duration2
        elif j == 2: Y = duration3
    elif Z==0: Y = 0; #if Z=0, Y=0
    IYnTrials[i,1] = Y
    #uniform probability of choosing I from ITI interval:
    I = np.random.choice(np.arange(iti_lb,iti_ub+dt,dt)) 
    IYnTrials[i,0] = I


    
    #generate trial sequence (X,W)
    X,W = trialDiscrete(I,Y,Z,dt,eta,interval)
    
    #calculating full joint posterior (forward inference) -- discrete observations
    posteriorScaled = inferenceDiscrete(p_signal,iti_lb,iti_ub, duration1, duration2,
                              duration3, dt, p_dur, eta, scale, interval)
    
    response = 2; #response generated by the animal
    a = posteriorScaled; 
    a = np.sum(a, axis = 1); a = np.sum(a, axis = 1); a = np.sum(a, axis = 1)
    postZ = a
    if postZ[0,-1] > postZ[1,-1]: response=0
    elif postZ[0,-1] < postZ[1,-1]: response =1
    
    posternTrials[i,:] = postZ[:,-1]; responsenTrials[i,0] = response
    
print(time.perf_counter()-start)

#counting trial types
hit = 0; miss = 0; cr = 0; fa = 0; omit = 0; cueTrials = 0; noncueTrials=0
hit25 = 0; hit50 = 0; hit500 = 0; miss25 = 0; miss50 = 0; miss500 = 0;
for k in range(nTrials):
    if ZnTrials[k,0] == 1: 
        cueTrials = cueTrials+1
        if responsenTrials[k,0] == 1:
            hit = hit +1
            if round(IYnTrials[k,1],3) == duration1: hit25 = hit25+1
            elif round(IYnTrials[k,1],3) == duration2: hit50 = hit50+1
            elif round(IYnTrials[k,1],3) == duration3: hit500 = hit500+1   
        elif responsenTrials[k,0] == 0:
            miss = miss +1
            if round(IYnTrials[k,1],3) == duration1: miss25 = miss25+1
            elif round(IYnTrials[k,1],3) == duration2: miss50 = miss50+1
            elif round(IYnTrials[k,1],3) == duration3: miss500 = miss500+1
        else: omit = omit +1
        
    elif ZnTrials[k,0] == 0:
        noncueTrials = noncueTrials+1
        if responsenTrials[k,0] == 0:
            cr = cr +1
        elif responsenTrials[k,0] == 1:
            fa = fa + 1
        else: omit = omit+1
"""                
    trialTypeRates[e,0] = hit/(hit+miss); trialTypeRates[e,1] = miss/(hit+miss); 
    trialTypeRates[e,2] = cr/(cr+fa); trialTypeRates[e,3] = fa/(cr+fa);
    trialTypeRates[e,4] = omit/(cueTrials+noncueTrials);
    trialTypeRates[e,5] = hit25/(hit25+miss25); trialTypeRates[e,6] = hit50/(hit50+miss50);
    trialTypeRates[e,7] = hit500/(hit500+miss500); trialTypeRates[e,8] = miss25/(hit25+miss25);        
    trialTypeRates[e,9] = miss50/(hit50+miss50); trialTypeRates[e,10] = miss500/(hit500+miss500);
"""
#%%
#plots for ITI and accuracy
typ = np.full((nTrials,5),0) #whether each trial is a hit,miss,fa,cr,omit
i = np.intersect1d(np.where(ZnTrials == 1)[0],np.where(responsenTrials == 1)[0])
typ[i,0] = 1 
i = np.intersect1d(np.where(ZnTrials == 1)[0],np.where(responsenTrials == 0)[0])
typ[i,1] = 1
i = np.intersect1d(np.where(ZnTrials == 0)[0],np.where(responsenTrials == 0)[0])
typ[i,2] = 1
i = np.intersect1d(np.where(ZnTrials == 0)[0],np.where(responsenTrials == 1)[0])
typ[i,3] = 1

hist = np.full((6,4),0.0)
for i in range(6):
    a = np.where(np.logical_and(IYnTrials[:,0]>=i, IYnTrials[:,0]<(i+1)))[0]
    hist[i,1] = len(np.intersect1d(a, np.where(typ[:,0]==1)))
    hist[i,2] = len(np.intersect1d(a, np.where(typ[:,2]==1)))
    hist[i,0] = len(a)
    
plt.bar([1,2,3,4,5,6],hist[:,1]/hist[:,0])
plt.bar([1,2,3,4,5,6],hist[:,2]/hist[:,0])

#plots for trends from multiple trials: variable eta
"""
dur = ['25', '50', '500', '0']
for i in range(len(etaTrial)):
    plt.plot(dur, [trialTypeRates[i,5],trialTypeRates[i,6],trialTypeRates[i,7],
                   trialTypeRates[i,2]], linestyle = 'dashed', marker ='o', 
                  label='eta=%1.1f'%etaTrial[i])
plt.legend(loc ='best'); plt.xlabel('signal duration in ms');
plt.ylabel('accuracy')
"""
#%%
#obselete code, old algorithms for filtering

#parameters learnt by the animals
p_signal = 0.5; iti_lb = 0; iti_ub = 6 #prob of signal trial, lower and upper bounds for iti
duration1 = 0.025; duration2 = 0.05; duration3 = 0.5; dt = 0.005 #3 signal durations, time step 
p_dur = [1/3,1/3,1/3]; eta = 0.9 #probability of each duration on a signal trial, confusability
iti_count = int((iti_ub-iti_lb)/dt)+1 #total number of possible iti's
dur = [int(duration1*(1/dt)),int(duration2*(1/dt)),int(duration3*(1/dt))]# signal duration multiple of dt

#algm1 : directly caluclating marginal posterior for Z:
n = int(1/dt)#time as multiple of dt
posterior = np.full((int((I*n+Y*n))+1,2),np.nan) #posterior for Z = 0 and 1
posteriorScaled = np.full((int((I*n+Y*n))+1,2),np.nan) #normalised posterior for Z = 0 and 1
posterior[0,:] = [p_signal, 1-p_signal]#initial posterior
aArr =  np.full((int((I*n+Y*n)),2),np.nan)#prob(wj|w1:j-1,z) for z=0,1
probWX = np.array([[eta, 1-eta], [1-eta, eta]]) #prob of W given X
for j in range(len(W)):
    #posterior Z = 0
    #when Z=0, probX=1 is 0 and Y=0;
    if j+1 < iti_count-1:
        a0 = (iti_count-j-1)*(probWX[W[j],1]*0 + probWX[W[j],0]*1)/(iti_count-j-1)# prob(wj|w1:j-1,zi) for zi=0
        posterior[j+1,0] = posterior[j,0]*a0
    elif j+1>=iti_count-1:
        a0 = 0
        posterior[j+1,0] = posterior[j,0]*a0
    #posterior Z = 1; Y = 25,50,500
    a1=0 #prob(wj|w1:j-1,zi) for zi=0
    for y in range(len(dur)): #add a1 for all the non-zero Y's
        if j+1<=iti_count-1:
            if j+1 <= dur[y]:
                temp = p_dur[y]*((j+1)*(probWX[W[j],1]*1 + probWX[W[j],0]*0) + (iti_count-j-1
                      )*(probWX[W[j],1]*0 + probWX[W[j],0]*1))/iti_count
            elif j+1 > dur[y]:
                temp = p_dur[y]*(dur[y]*(probWX[W[j],1]*1 + probWX[W[j],0]*0) + (iti_count-j-1
                      )*(probWX[W[j],1]*0 + probWX[W[j],0]*1))/(iti_count-j-1+dur[y])
            a1 = a1+temp
        elif j+1 > iti_count-1:
            pdurW = np.array(p_dur);
            if j-dur[y] < iti_count-1:
                temp = pdurW[y]*(iti_count-j-1+dur[y])*(probWX[W[j],1]*1 + probWX[W[j],0]*0)/(iti_count-j-1+dur[y])
            elif j-dur[y] >= iti_count-1:
                temp = 0
                pdurW[y] = 0; pdurW = pdurW/(sum(pdurW))
            a1 = a1+temp
            
    aArr[j,:] = [a0,a1]
    posterior[j+1,1] = posterior[j,1]*a1 #posterior on j from j-1
    #normalised posteriors:
    posteriorScaled[j+1,0] = posterior[j+1,0]/(a0*posterior[j,0]+a1*posterior[j,1]) 
    posteriorScaled[j+1,1] = posterior[j+1,1]/(a0*posterior[j,0]+a1*posterior[j,1])

t = np.arange(0,len(X)*dt,dt)
plt.scatter(t,W, label='Wj', s=10); plt.scatter(t,X, label='Xj', s=10); 
plt.legend(); plt.xlabel('time in sec'); plt.title('Y=%1.3fsec,I=%1.3fsec'%(Y,I))
plt.plot(t, posteriorScaled[1:,0], label = 'Z=0', color = 'red')
plt.plot(t, posteriorScaled[1:,1], label = 'Z=1', color = 'green')
plt.legend(); plt.xlabel('time in sec'); plt.ylabel('posterior prob of Z given W')

#algm 2: calculating the total joint posterior cell by cell (slow!)
z = [0,1]; y = [0,int(duration1/dt),int(duration2/dt),int(duration3/dt)]; 
iti=np.arange(0,iti_count,1); x = [0,1]

for j in range(len(W)):
    for zi in range(len(z)):
        for yi in range(len(y)):
            for ti in range(len(iti)):                   
                #xi = 0
                posterior[zi,yi,ti,0,j+1] = probWX[W[j],0][0]*probX_YI(y[yi],iti[ti],j+1
                )[0]*np.sum(posterior[zi,yi,ti,:,j])
                #xi = 1
                posterior[zi,yi,ti,1,j+1] = probWX[W[j],1][0]*probX_YI(y[yi],iti[ti],j+1
                )[1]*np.sum(posterior[zi,yi,ti,:,j])
    posteriorScaled[:,:,:,:,j+1] = posterior[:,:,:,:,j+1]/(np.sum(posterior[:,:,:,:,j+1]))

 