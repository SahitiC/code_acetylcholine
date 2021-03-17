#importing required packages
import numpy as np
import matplotlib.pyplot as plt
import time

#%%
b = np.arange(0.0,1.02,0.01) #discrete belief space use for b0 and b1
etaL = 0.6; etaH = 0.9 #two levels of eta for the two internal states
I = np.array([0,1]) #internal state space : choose low or high eta levels
O = np.array([0,1]) #observation space: 0/1
s = np.array([0,1,2]) #environmental state space
I_N = np.array([0,1]) #states to choose at N (H0 or H1) 
PX_s = np.array([[[etaL,1-etaL,etaL],[1-etaL,etaL,1-etaL]],[[etaH,1-etaH,etaH],[1-etaH,etaH,1-etaH]]])
R = np.array([[1,0],[0,1]]) 
#R00,R01,R10,R11 (Rij = rewards on choosing Hi when Hj is true)
c00 = 0.01; c10 = 0.01; c01 = 0.02; c11 = 0.02 
#magnitude of costs on going from i to j internal states
c = np.array([[c00,c01],[c10,c11]])

n = 10 #trial length

value = np.full((len(b),len(b),len(I),n),np.nan) #value for each state for all n time steps
policy = np.full((len(b),len(b),len(I),n),np.nan) #corresponding policy

#try optimal policy for last time step where I doesn't matter
for i in range(len(b)):#belief for state = 1
    j = 0
    while (b[i]+b[j])<=1:#belief for state = 2
        #immediate reward,position 0 is for choosing H0 and 1 is for choosing H1
        r = [(b[i]+b[j])*R[0,1]+(1-b[i]-b[j])*R[0,0], 
             (b[i]+b[j])*R[1,1]+(1-b[i]-b[j])*R[1,0]] 
        value[i,j,0,n-1] = np.max(r) #maximum reward
        policy[i,j,0,n-1] = np.where(r == value[i,j,0,n-1])[0][0] #position/choice which gives max reward
        j=j+1
value[:,:,1,n-1] = value[:,:,0,n-1]
policy[:,:,1,n-1] = policy[:,:,0,n-1]    
    
#current internal state = 0
for i in range(len(b)):
    #immediate reward on choosing I = 0 vs 1
    r = [-c[0,0],-c[0,1]]
    EValue = []
    #sum(P)