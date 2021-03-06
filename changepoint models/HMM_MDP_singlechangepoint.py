#importing required packages
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import pandas as pd
import matplotlib

#%%
def interpolate(x, x_lb, x_ub, dx,func_lb,func_ub):
    interpolated_func = 0.0
    interpolated_func = (((x-x_lb)*func_ub) + ((x_ub-x)*func_lb))/dx
    return interpolated_func


def softmax(x, beta):
    """Compute softmax values for each sets of scores in x."""
    x = np.array(x)
    return np.exp(beta*x) / np.sum(np.exp(beta*x), axis=0)


#%%
def getOptimalPolicy(b,I,O,etaL,etaH,s,I_N,PX_s,R,cost,p_signal,n,db,
                     compare,beta):
    c=cost    
    value = np.full((len(b),len(I),n+1),np.nan) #value for each state for all n time steps
    policy = np.full((len(b),len(I),n+1),np.nan) #corresponding policy
    #belief space is now 1-D; 
    value0 = np.full((len(b),len(I),n+1),np.nan)#collecting values for I0
    value1 = np.full((len(b),len(I),n+1),np.nan)#collecting values for I1
    
    for i in range(len(b)):#belief for state = 1
        #immediate reward,position 0 is for choosing H0 and 1 is for choosing H1
        r = [b[i]*R[0,1]+(1-b[i])*R[0,0],(b[i])*R[1,1]+(1-b[i])*R[1,0]] 
        value0[i,0,n] = r[0];
        value1[i,0,n] = r[1];
        value[i,0,n] = np.max(r) #maximum reward
        policy[i,0,n] = np.where(r == value[i,0,n])[0][0] #position/choice which gives max reward
        
    value[:,1,n] = value[:,0,n]
    value0[:,1,n] = value0[:,0,n]
    value1[:,1,n] = value1[:,0,n]
    policy[:,1,n] = policy[:,0,n]    
    
    for t1 in range(n):
        t = n-1-t1
        probStart = 1/((n/p_signal)-t)
        transition_matrix = np.array([[1-probStart,probStart],[0,1]])
        #current internal state = 0
        iters = len(b)
        if etaH == 1:
            value[int(1/db),0,t] =1.0; policy[int(1/db),0,t] =1 #use only when etaH=1
            value[int(1/db),1,t] =1.0; policy[int(1/db),1,t] =1 #use only when etaH=1
            value0[int(1/db),0,t] = 1-c[0,0]; value0[int(1/db),1,t] = 1-c[1,0]; 
            value1[int(1/db),0,t] = 1.-c[0,1]; value1[int(1/db),1,t] = 1.-c[1,1];
            iters=len(b)-1
        for i in range(iters):#iterate over current belief state
            bArr = [1-b[i],b[i]]
            arr = np.array([[np.matmul(bArr,transition_matrix)*PX_s[0,0,:],#a=0,obs=0
                              np.matmul(bArr,transition_matrix)*PX_s[0,1,:]],#a=0,obs=1
                            [np.matmul(bArr,transition_matrix)*PX_s[1,0,:],#a=1,obs=0
                             np.matmul(bArr,transition_matrix)*PX_s[1,1,:]]])#a=1,obs=1
    
            b1Arr = np.array([[arr[0,0,:]/np.sum(arr[0,0,:]),
                              arr[0,1,:]/np.sum(arr[0,1,:])],
                            [arr[1,0,:]/np.sum(arr[1,0,:]),
                             arr[1,1,:]/np.sum(arr[1,1,:])]]) #next belief state for each 
            closest_index = np.array(b1Arr/db, dtype=int) #closest lower index for each a & obs
            
            if (closest_index == len(b)-1).any():        
                future_v_lb = np.array([[value[closest_index[0,0,1],0,t+1],
                                value[closest_index[0,1,1],0,t+1]],
                                [value[closest_index[1,0,1],1,t+1],
                                value[closest_index[1,1,1],1,t+1]]]) #lower bound value for each a & obs
                future_v_ub = np.array([[value[closest_index[0,0,1],0,t+1],
                                value[closest_index[0,1,1],0,t+1]],
                                [value[closest_index[1,0,1],1,t+1],
                                value[closest_index[1,1,1],1,t+1]]]) #upper bound value for each a & obs
                
                future_vb1Arr =  future_v_lb
                
            else: 
                future_v_lb = np.array([[value[closest_index[0,0,1],0,t+1],
                                value[closest_index[0,1,1],0,t+1]],
                                [value[closest_index[1,0,1],1,t+1],
                                value[closest_index[1,1,1],1,t+1]]]) #lower bound value for each a & obs
                future_v_ub = np.array([[value[closest_index[0,0,1]+1,0,t+1],
                                value[closest_index[0,1,1]+1,0,t+1]],
                                [value[closest_index[1,0,1]+1,1,t+1],
                                value[closest_index[1,1,1]+1,1,t+1]]]) #upper bound value for each a & obs
                
                future_vb1Arr =  interpolate(b1Arr[:,:,1], b[closest_index[:,:,1]], 
                                      b[closest_index[:,:,1]+1], db, future_v_lb, 
                                      future_v_ub) #interpolate
                    
            
            future_v0 = np.sum(arr[0,0,:])*future_vb1Arr[0,0]+np.sum(
                arr[0,1,:])*future_vb1Arr[0,1] #next value of a=0 
            future_v1 = np.sum(arr[1,0,:])*future_vb1Arr[1,0]+np.sum(
                arr[1,1,:])*future_vb1Arr[1,1] #next value of a=0 
            v  = np.array([-c[0,0]+future_v0,-c[0,1]+future_v1]) #add costs
            value0[i,0,t] = v[0]; #qvalue for a=0
            value1[i,0,t] = v[1] #qvalue for a=1
            
            #choose optimal action
            if compare == 0: #max
            
                value[i,0,t] = np.max([round(v[0],3), round(v[1],3)])
                
                if round(v[0],3)==round(v[1],3):policy[i,0,t]=2
                elif round(v[0],3)>round(v[1],3):policy[i,0,t]=0
                elif round(v[0],3)<round(v[1],3):policy[i,0,t]=1   
                       
            elif compare == 1:#soft max
                
                r = np.round(softmax(v,beta),3)
                 
                value[i,0,t] = np.round(v[np.where(r == np.max(
                    r))[0][0]],3)   
                
                if r[0]== r[1]:policy[i,0,t]=2
                elif r[0]>r[1]:policy[i,0,t]=0
                elif r[0]<r[1]:policy[i,0,t]=1   
           
                                    
        #current internal state = 1 (similar steps as above for difft current state)
        for i in range(iters):#belief for state = 1
    
            bArr = [1-b[i],b[i]]
            arr = np.array([[np.matmul(bArr,transition_matrix)*PX_s[0,0,:],
                              np.matmul(bArr,transition_matrix)*PX_s[0,1,:]],
                            [np.matmul(bArr,transition_matrix)*PX_s[1,0,:],
                             np.matmul(bArr,transition_matrix)*PX_s[1,1,:]]])
    
            b1Arr = np.array([[arr[0,0,:]/np.sum(arr[0,0,:]),
                              arr[0,1,:]/np.sum(arr[0,1,:])],
                            [arr[1,0,:]/np.sum(arr[1,0,:]),
                             arr[1,1,:]/np.sum(arr[1,1,:])]])
            closest_index = np.array(b1Arr/db, dtype=int)
            
            if (closest_index == len(b)-1).any():        
                future_v_lb = np.array([[value[closest_index[0,0,1],0,t+1],
                                value[closest_index[0,1,1],0,t+1]],
                                [value[closest_index[1,0,1],1,t+1],
                                value[closest_index[1,1,1],1,t+1]]])
                future_v_ub = np.array([[value[closest_index[0,0,1],0,t+1],
                                value[closest_index[0,1,1],0,t+1]],
                                [value[closest_index[1,0,1],1,t+1],
                                value[closest_index[1,1,1],1,t+1]]])
                
                future_vb1Arr =  future_v_lb
                
            else: 
                future_v_lb = np.array([[value[closest_index[0,0,1],0,t+1],
                                value[closest_index[0,1,1],0,t+1]],
                                [value[closest_index[1,0,1],1,t+1],
                                value[closest_index[1,1,1],1,t+1]]])
                future_v_ub = np.array([[value[closest_index[0,0,1]+1,0,t+1],
                                value[closest_index[0,1,1]+1,0,t+1]],
                                [value[closest_index[1,0,1]+1,1,t+1],
                                value[closest_index[1,1,1]+1,1,t+1]]])
                
                future_vb1Arr =  interpolate(b1Arr[:,:,1], b[closest_index[:,:,1]], 
                                      b[closest_index[:,:,1]+1], db, future_v_lb, 
                                      future_v_ub)
    
                    
            
            future_v0 = np.sum(arr[0,0,:])*future_vb1Arr[0,0]+np.sum(
                arr[0,1,:])*future_vb1Arr[0,1]
            future_v1 = np.sum(arr[1,0,:])*future_vb1Arr[1,0]+np.sum(
                arr[1,1,:])*future_vb1Arr[1,1]
            v = np.array([-c[1,0]+future_v0,-c[1,1]+future_v1])
            value0[i,1,t] = v[0]; 
            value1[i,1,t] = v[1]
            
            if compare == 0:
            
                value[i,1,t] = np.max([round(v[0],3), round(v[1],3)])
                
                if round(v[0],3)==round(v[1],3):policy[i,1,t]=2
                elif round(v[0],3)>round(v[1],3):policy[i,1,t]=0
                elif round(v[0],3)<round(v[1],3):policy[i,1,t]=1   
                       
            elif compare == 1:
                
                r = np.round(softmax(v,beta),3)
                 
                value[i,1,t] = v[np.where(r == np.max(r))[0][0]]
                
                
                if r[0]== r[1]:policy[i,1,t]=2
                elif r[0]>r[1]:policy[i,1,t]=0
                elif r[0]<r[1]:policy[i,1,t]=1   
           
    return value0,value1,value,policy

#%%
#simpler case: two states, single changepoint
db = 0.001
b = np.arange(0.0,1.0+(db),db) #discrete belief space to use for b0 and b1
rounding=3
b = np.round(b,rounding)
etaL = 0.5; etaH = 1.0 #two levels of eta for the two internal states
I = np.array([0,1]) #internal state space : choose low or high eta levels
O = np.array([0,1]) #observation space: 0/1
s = np.array([0,1]) #environmental state space
I_N = np.array([0,1]) #states to choose at N (H0 or H1) 
PX_s = np.array([[[etaL,1-etaL],[1-etaL,etaL]],[[etaH,1-etaH],[1-etaH,etaH]]])
R = np.array([[1,0],[0,1]]) #R00,R01,R10,R11 (Rij = rewards on choosing Hi when Hj is true)
c00 = 0.00; c10 = 0.00; c01 = 0.00; c11 = 0.00
#magnitude of costs on going from i to j internal states
cost = np.array([[c00,c01],[c10,c11]])
p_signal = 0.5; 

compare = 1; beta = 100

trial_length = 10 #trial 

value0,value1,value,policy=getOptimalPolicy(b,I,O,etaL,
                        etaH,s,I_N,PX_s,R,cost,p_signal,trial_length,db,
                        compare,beta)

#%%
#heatmaps
plt.imshow(value[:,0,:-1],interpolation='nearest',extent=[0,9,1,0],aspect=9)
plt.ylabel('belief state'); plt.xlabel('time'); 
plt.title('value, cost01=%1.2f,cost11=%1.2f,etaL=%1.1f,etaH=%1.1f'%(
    c01,c11,etaL,etaH)); plt.colorbar();
plt.figure()
plt.imshow(value0[:,0,:-1]-value1[:,0,:-1],interpolation='nearest',extent=[0,9,1,0],aspect=9)
plt.ylabel('belief state'); plt.xlabel('time'); 
plt.title('Q0-Q1,cost01=%1.2f,cost11=%1.2f,etaL=%1.1f,etaH=%1.1f'%(
    c01,c11,etaL,etaH)); plt.colorbar();
plt.figure()
plt.imshow(value0[:,0,:-1],interpolation='nearest',extent=[0,9,1,0],aspect=9)
plt.ylabel('belief state'); plt.xlabel('time'); 
plt.title('Q0, cost01=%1.2f,cost11=%1.2f,etaL=%1.1f,etaH=%1.1f'%(
    c01,c11,etaL,etaH)); plt.colorbar();
plt.figure()
plt.imshow(value1[:,0,:-1],interpolation='nearest',extent=[0,9,1,0],aspect=9)
plt.ylabel('belief state'); plt.xlabel('time'); 
plt.title('Q1,,cost01=%1.2f,cost11=%1.2f,etaL=%1.1f,etaH=%1.1f'%(
    c01,c11,etaL,etaH)); plt.colorbar();
plt.figure()

cmap = matplotlib.cm.get_cmap('viridis', 3)
norm = matplotlib.colors.BoundaryNorm(np.arange(0, 2, 1), cmap.N)
mat = plt.imshow(policy[:,0,:-1],cmap=cmap,vmin =-0.5,vmax = 2.5,
           interpolation='nearest',extent=[0,9,1,0],aspect=9)
plt.ylabel('belief state'); plt.xlabel('time'); 
plt.title('policy,cost01=%1.2f,cost11=%1.2f,etaL=%1.1f,etaH=%1.1f'%(
    c01,c11,etaL,etaH)); plt.colorbar(mat,ticks=np.linspace(0,2,3));
plt.figure()

#%%
#lineplots
n=trial_length
for j in range(10):
    i=n-j
    plt.plot(b,value[:,0,i], label='value'); 
    plt.plot(b,value0[:,0,i], label='value0')
    plt.plot(b,value1[:,0,i], label='value1'); 
    plt.legend(); 
    plt.title('t=%d,n=%d,etaL=%1.2f, etaH=%1.2f,cij=%1.2f'%(i+1,n,etaL,etaH,cost[0,1]));
    plt.xlabel('belief state');plt.figure()
    
    """
    plt.plot(b,policy[:,0,i], label='policy')
    plt.legend();plt.title('t=%d,n=%d,etaL=%1.2f, etaH=%1.2f,cij=%1.2f'%(i+1,n,etaL,etaH,cost[0,1]));
    plt.xlabel('belief state');plt.figure()
    """
    
for j in range(5):
    i=n-j-1
    plt.plot(b,value[:,1,i], label='value'); 
    plt.plot(b,value0[:,1,i], label='value0')
    plt.plot(b,value1[:,1,i], label='value1'); 
    plt.legend(); plt.title('t=%d,n=%d,c01=%1.12f,c1=%1.2f'%(i+1,n,cost[0,1],cost[1,1]));
    plt.xlabel('belief state');plt.figure()
    plt.plot(b,policy[:,1,i], label='policy')
    plt.legend();plt.title('t=%d,n=%d,c01=%1.12f,c1=%1.2f'%(i+1,n,cost[0,1],cost[1,1]));
    plt.xlabel('belief state');plt.figure()
    

#%%
#inference with executing the policy: single change point
def generate_trialPolicy(trial_length, p_signal, signal_length_type,
                   signal_length):
    n = trial_length
    trial_signal = np.random.binomial(1,p_signal) #trial is signal/non signal with prob 0.5
    
    if signal_length_type == 0:
        start_signal = np.random.randint(0, n) #index of sample when signal begins (on signal trial)
    elif signal_length_type == 1:
        start_signal = np.random.randint(0, n-signal_length+1) #index of sample when signal begins (on signal trial)
    
    trial = np.full((int(round(n))+1,1),0);#state values within a trial
    if trial_signal == 1: 
        trial[start_signal+1:] = np.full((n-start_signal,1),1)
    
    return trial,trial_signal


def inferenceDiscretePolicy(trial,trial_length,p_signal, 
                            etaL,etaH,value0, value1,cost,b,db,beta,rounding):
    n = trial_length  
    
    observation = np.full((int(round(n)),1),0)
    posterior = np.full((n+1,2),0.0) #posterior for states 0,1,2
    posterior[0,:] = [1.0,0.0] # posterior at j = -1
    posterior[0,:] = posterior[0,:]/(np.sum(posterior[0,:]))
    #array for internal state and eta
    internalState = np.full((n+1,1),0); #IS 
    internalState[0] = 0 #initialise initial IS
    etaArr = np.full((n,1),0.0) #corresponding eta (determined by IS)
    action = np.full((n,1),0); #array for action = IS for each t>0
    eta = [etaL,etaH] #the two eta levels
    
    #observation and inference 
    for j in range(n):
        
        idx = int(posterior[j,1]/db)
        if idx < int(1/db):
            q0 = interpolate(posterior[j,1],b[idx], b[idx+1], db, 
                             value0[idx,internalState[j][0],j],
                             value0[idx+1,internalState[j][0],j])
            q1 = interpolate(posterior[j,1],b[idx], b[idx+1], db, 
                     value1[idx,internalState[j][0],j],
                     value1[idx+1,internalState[j][0],j])
            
        elif idx >= int(1/db):
            q0 = value0[idx,internalState[j][0],j]
            q1 = value1[idx,internalState[j][0],j]
        
        q = np.array([q0,q1])
        r = softmax(q,beta)
        #r = np.round(softmax(q,beta),rounding)
        if r[0]== r[1]:internalState[j+1]=1; action[j] = 1; etaArr[j] = eta[1]
        elif r[0]>r[1]:internalState[j+1]=0; action[j] = 0; etaArr[j] = eta[0]
        elif r[0]<r[1]:internalState[j+1]=1; action[j] = 1;  etaArr[j] = eta[1] 

        
        if trial[j+1] == 0:
            observation[j] = np.random.binomial(1,1-etaArr[j])
        elif trial[j+1] == 1:
            observation[j] = np.random.binomial(1,etaArr[j])
        
        probStart = 1/((n/p_signal)-j)
        transition_matrix = np.array([[1-probStart,probStart],
                                      [0,1]])
        emission_matrix = np.array([[etaArr[j][0],1-etaArr[j][0]],
                                    [1-etaArr[j][0],etaArr[j][0]]])
        emission_probability = emission_matrix[observation[j],:]
        
        posterior[j+1,:] = emission_probability*np.matmul(posterior[j,:],
                                                transition_matrix)
        posterior[j+1,:] = posterior[j+1,:]/(np.sum(posterior[j+1,:]))
        

    return observation, posterior, internalState, action

def generate_responseDiscretePolicy(trial,posterior):
    response = 10
    hit = 0; miss = 0; cr = 0; fa = 0
    trial_signal = 0
    if sum(trial) > 0: trial_signal = 1
    inferred_state = np.where(posterior[len(trial)-1,:] == max(
        posterior[len(trial)-1,:]))[0]
    if inferred_state ==0:
        response = 0;
        if trial_signal==0: cr =cr+1
        elif trial_signal==1: miss =miss+1
    elif inferred_state==1:
        response = 1;
        if trial_signal==0: fa=fa+1
        elif trial_signal==1: hit=hit+1
    
    return inferred_state,response, hit, miss, cr, fa  




#%%
#simpler case: two states, single changepoint
db = 0.001
b = np.arange(0.0,1.0+(db),db) #discrete belief space to use for b0 and b1
b = np.round(b,3);rounding = 3
etaL = 0.5; etaH = 1.0 #two levels of eta for the two internal states
I = np.array([0,1]) #internal state space : choose low or high eta levels
O = np.array([0,1]) #observation space: 0/1
s = np.array([0,1]) #environmental state space
I_N = np.array([0,1]) #states to choose at N (H0 or H1) 
PX_s = np.array([[[etaL,1-etaL],[1-etaL,etaL]],[[etaH,1-etaH],[1-etaH,etaH]]])
R = np.array([[1,0],[0,1]]) #R00,R01,R10,R11 (Rij = rewards on choosing Hi when Hj is true)
c00 = 0.00; c10 = 0.00; c01 = 0.03; c11 = 0.02
#magnitude of costs on going from i to j internal states
cost = np.array([[c00,c01],[c10,c11]])
p_signal = 0.5; 

compare = 1; beta = 100; 

trial_length = 10 #trial 

value0,value1,value,policy=getOptimalPolicy(b,I,O,etaL,
                        etaH,s,I_N,PX_s,R,cost,p_signal,trial_length,db,
                        compare,beta)

#%%

signal_length_type = 0; signal_length=1

trial,trial_signal = generate_trialPolicy(trial_length, p_signal, signal_length_type,
                   signal_length)
observation,posterior, internalState, action = inferenceDiscretePolicy(
    trial,trial_length,p_signal, etaL,etaH,value0, value1,cost,b,db, beta, rounding)
inferred_state,response, hit, miss, cr, fa  = generate_responseDiscretePolicy(
    trial,posterior)


#%%
fig,ax = plt.subplots(2,1)
t = np.arange(0,trial_length+1,1)
ax[0].plot(t[1:],trial[1:], label='underlying signal')
ax[0].scatter(t[1:],observation, label='observations',color ='orange')
ax[0].plot(t,posterior[:,1], label='posterior for s=1')

ax[1].plot(t,internalState[:], label= 'internal state', linestyle ='dashed', 
           marker ='o')
ax[1].plot(t[1:],action, label = 'action', linestyle ='dashed', 
           marker ='o')

ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left');
ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

#%%
db = 0.001
b = np.arange(0.0,1.0+(db),db) #discrete belief space to use for b0 and b1
b = np.round(b,3);rounding = 3
etaL = 0.7; etaH = 0.8 #two levels of eta for the two internal states
I = np.array([0,1]) #internal state space : choose low or high eta levels
O = np.array([0,1]) #observation space: 0/1
s = np.array([0,1]) #environmental state space
I_N = np.array([0,1]) #states to choose at N (H0 or H1) 
PX_s = np.array([[[etaL,1-etaL],[1-etaL,etaL]],[[etaH,1-etaH],[1-etaH,etaH]]])
R = np.array([[1,0],[0,1]]) #R00,R01,R10,R11 (Rij = rewards on choosing Hi when Hj is true)
c00 = 0.00; c10 = 0.00; c01 = 0.03; c11 = 0.03
#magnitude of costs on going from i to j internal states
cost = np.array([[c00,c01],[c10,c11]])
p_signal = 0.5; 

compare = 1; beta = 50; 

trial_length = 10 #trial 
value0,value1,value,policy=getOptimalPolicy(b,I,O,etaL,
                        etaH,s,I_N,PX_s,R,cost,p_signal,trial_length,db,
                        compare,beta)

#%%
#average trajectories

nTrials = 2000
posteriorTrials = np.full((nTrials,trial_length+1,2),0.0)
actionTrials = np.full((nTrials,trial_length,1),0.0)
trialType = np.full((nTrials,1),0.0)

signal_length_type = 0; signal_length=1

for nT in range(nTrials):
    trial,trial_signal = generate_trialPolicy(trial_length, p_signal, signal_length_type,
                   signal_length)
    observation,posterior, internalState, action = inferenceDiscretePolicy(
    trial,trial_length,p_signal, etaL,etaH,value0, value1,cost,b,db, beta, rounding)
    inferred_state,response, hit, miss, cr, fa  = generate_responseDiscretePolicy(
    trial,posterior)
    posteriorTrials[nT,:,:] = posterior
    trialType[nT] = trial_signal
    actionTrials[nT,:,:] = action

#%%
t = np.arange(0,trial_length+1,1)
#for signal trials
cmap = matplotlib.cm.get_cmap('Pastel1', 3)
norm = matplotlib.colors.BoundaryNorm(np.arange(0, 2, 1), cmap.N)
mat = plt.imshow(policy[:,1,:-1],cmap=cmap,vmin =-0.5,vmax = 2.5,
           interpolation='nearest',extent=[0,9,1,0],aspect=9)
plt.ylabel('belief state'); plt.xlabel('time'); 
cbar = plt.colorbar(mat,ticks=np.linspace(0,2,3));
cbar.set_ticklabels(['L','H','L/H'])


avgPosterior = np.average(posteriorTrials[np.where(trialType==1)[0],:,:],axis=0)
avgAction = np.average(actionTrials[np.where(trialType==1)[0],:,:],axis=0)

stdPosterior = np.std(posteriorTrials[np.where(trialType==1)[0],:,:],axis=0)
stdAction = np.std(actionTrials[np.where(trialType==1)[0],:,:],axis=0)


plt.plot(t[:-1],avgPosterior[1:,1],label ='avg b(1)-signal trials') 
plt.fill_between(t[:-1,],avgPosterior[1:,1]-stdPosterior[1:,1],
                 avgPosterior[1:,1]+stdPosterior[1:,1],alpha=0.2)
plt.plot(t[:-1],avgAction[:,0],label ='avg action-signal trials')
plt.fill_between(t[:-1,],avgAction[:,0]-stdAction[:,0],
                 avgAction[:,0]+stdAction[:,0],alpha=0.1)
plt.legend()
plt.xlabel('time'); 
plt.ylim(1.02,-0.02)

 
#for  non-signal trials
avgPosterior = np.average(posteriorTrials[np.where(trialType==0)[0],:,:],axis=0)
avgAction = np.average(actionTrials[np.where(trialType==0)[0],:,:],axis=0)
stdPosterior = np.std(posteriorTrials[np.where(trialType==0)[0],:,:],axis=0)
stdAction = np.std(actionTrials[np.where(trialType==0)[0],:,:],axis=0)


plt.plot(t[:-1],avgPosterior[1:,1],label ='avg b(1)-non signal trials') 
plt.fill_between(t[:-1,],avgPosterior[1:,1]-stdPosterior[1:,1],
                 avgPosterior[1:,1]+stdPosterior[1:,1],alpha=0.2)
plt.plot(t[:-1],avgAction[:,0],label ='avg action-non signal trials')
plt.fill_between(t[:-1,],avgAction[:,0]-stdAction[:,0],
                 avgAction[:,0]+stdAction[:,0],alpha=0.1)

plt.xlabel('time'); 

plt.ylim(1.02,-0.02)

plt.title('optimal action, etaH=%1.2f, etaL=%1.2f,ci1=%1.2f' 
          %(etaH,etaL,c01)) 
plt.legend(bbox_to_anchor=(1.2, 1), loc='upper left')

#%%  
#avg plots without overlaying on heatmaps

#signal
avgPosterior = np.average(posteriorTrials[np.where(trialType==1)[0],:,:],axis=0)
avgAction = np.average(actionTrials[np.where(trialType==1)[0],:,:],axis=0)

stdPosterior = np.std(posteriorTrials[np.where(trialType==1)[0],:,:],axis=0)
stdAction = np.std(actionTrials[np.where(trialType==1)[0],:,:],axis=0)


plt.plot(t[:-1],avgPosterior[1:,1],label ='avg b(1)-signal trials') 
plt.fill_between(t[:-1,],avgPosterior[1:,1]-stdPosterior[1:,1],
                 avgPosterior[1:,1]+stdPosterior[1:,1],alpha=0.2)
plt.plot(t[:-1],avgAction[:,0],label ='avg action-signal trials')
plt.fill_between(t[:-1,],avgAction[:,0]-stdAction[:,0],
                 avgAction[:,0]+stdAction[:,0],alpha=0.2)
plt.legend()
plt.xlabel('time'); 
plt.title('etaH=%1.2f, etaL=%1.2f, ci1=%1.2f'
                              %(etaH,etaL,c11)) 
plt.ylim(1.02,-0.02)

plt.figure()

#non-signal
avgPosterior = np.average(posteriorTrials[np.where(trialType==0)[0],:,:],axis=0)
avgAction = np.average(actionTrials[np.where(trialType==0)[0],:,:],axis=0)
stdPosterior = np.std(posteriorTrials[np.where(trialType==0)[0],:,:],axis=0)
stdAction = np.std(actionTrials[np.where(trialType==0)[0],:,:],axis=0)


plt.plot(t[:-1],avgPosterior[1:,1],label ='avg b(1)-non signal trials') 
plt.fill_between(t[:-1,],avgPosterior[1:,1]-stdPosterior[1:,1],
                 avgPosterior[1:,1]+stdPosterior[1:,1],alpha=0.2)
plt.plot(t[:-1],avgAction[:,0],label ='avg action-non signal trials')
plt.fill_between(t[:-1,],avgAction[:,0]-stdAction[:,0],
                 avgAction[:,0]+stdAction[:,0],alpha=0.2)

plt.xlabel('time'); 

plt.ylim(1.02,-0.02)

plt.title('etaH=%1.2f, etaL=%1.2f, ci1=%1.2f' %(etaH,etaL,c11)) 
plt.legend(bbox_to_anchor=(1.2, 1), loc='upper left')