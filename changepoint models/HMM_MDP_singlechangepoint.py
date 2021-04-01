#importing required packages
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns

#%%
def interpolate(x, x_lb, x_ub, dx,func_lb,func_ub):
    interpolated_func = 0.0
    interpolated_func = (((x-x_lb)*func_ub) + ((x_ub-x)*func_lb))/dx
    return interpolated_func



#%%

def getOptimalPolicy(b,I,O,etaL,etaH,s,I_N,PX_s,R,cost,p_signal,n):
    c=cost    
    value = np.full((len(b),len(I),n),np.nan) #value for each state for all n time steps
    policy = np.full((len(b),len(I),n),np.nan) #corresponding policy
    #belief space is now 1-D; 
    value0 = np.full((len(b),len(I),n),np.nan)#collecting values for I0
    value1 = np.full((len(b),len(I),n),np.nan)#collecting values for I1
    
    for i in range(len(b)):#belief for state = 1
        #immediate reward,position 0 is for choosing H0 and 1 is for choosing H1
        r = [b[i]*R[0,1]+(1-b[i])*R[0,0],(b[i])*R[1,1]+(1-b[i])*R[1,0]] 
        value0[i,0,n-1] = r[0];
        value1[i,0,n-1] = r[1];
        value[i,0,n-1] = np.max(r) #maximum reward
        policy[i,0,n-1] = np.where(r == value[i,0,n-1])[0][0] #position/choice which gives max reward
        
    value[:,1,n-1] = value[:,0,n-1]
    value0[:,1,n-1] = value0[:,0,n-1]
    value1[:,1,n-1] = value1[:,0,n-1]
    policy[:,1,n-1] = policy[:,0,n-1]    
    
    for t1 in range(n-1):
        t = n-2-t1
        probStart = 1/((n/p_signal)-t)
        transition_matrix = np.array([[1-probStart,probStart],[0,1]])
        #current internal state = 0
        iters = len(b)
        if etaH == 1:
            value[1000,0,t] =1.0; policy[1000,0,t] =1 #use only when etaH=1
            value[1000,1,t] =1.0; policy[1000,1,t] =1 #use only when etaH=1
            value0[1000,0,t] = 1-c[0,0]; value0[1000,1,t] = 1-c[1,0]; 
            value1[1000,0,t] = 1.-c[0,1]; value1[1000,1,t] = 1.-c[1,1];
            iters=len(b)-1
        for i in range(iters):#belief for state = 1
            bArr = [1-b[i],b[i]]
            arr = np.array([[np.matmul(bArr,transition_matrix)*PX_s[0,0,:],#I'=0,obs=0
                              np.matmul(bArr,transition_matrix)*PX_s[0,1,:]],#I'=0,obs=1
                            [np.matmul(bArr,transition_matrix)*PX_s[1,0,:],#I'=1,obs=0
                             np.matmul(bArr,transition_matrix)*PX_s[1,1,:]]])#I'=1,obs=1
    
            b1Arr = np.array([[arr[0,0,:]/np.sum(arr[0,0,:]),
                              arr[0,1,:]/np.sum(arr[0,1,:])],
                            [arr[1,0,:]/np.sum(arr[1,0,:]),
                             arr[1,1,:]/np.sum(arr[1,1,:])]])
            closest_index = np.array(b1Arr/0.001, dtype=int)
            
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
                                      b[closest_index[:,:,1]+1], 0.001, future_v_lb, 
                                      future_v_ub)
                    
            
            future_v0 = np.sum(arr[0,0,:])*future_vb1Arr[0,0]+np.sum(
                arr[0,1,:])*future_vb1Arr[0,1]
            future_v1 = np.sum(arr[1,0,:])*future_vb1Arr[1,0]+np.sum(
                arr[1,1,:])*future_vb1Arr[1,1]
            v = np.array([-c[0,0]+future_v0,-c[0,1]+future_v1])
            value0[i,0,t] = v[0]; 
            value1[i,0,t] = v[1]
            value[i,0,t] = np.max(v)
            
            
            if round(v[0],3)==round(v[1],3):policy[i,0,t]=2
            elif round(v[0],3)>round(v[1],3):policy[i,0,t]=0
            elif round(v[0],3)<round(v[1],3):policy[i,0,t]=1   
                       
                
                
        #current internal state = 1
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
            closest_index = np.array(b1Arr/0.001, dtype=int)
            
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
                                      b[closest_index[:,:,1]+1], 0.001, future_v_lb, 
                                      future_v_ub)
    
                    
            
            future_v0 = np.sum(arr[0,0,:])*future_vb1Arr[0,0]+np.sum(
                arr[0,1,:])*future_vb1Arr[0,1]
            future_v1 = np.sum(arr[1,0,:])*future_vb1Arr[1,0]+np.sum(
                arr[1,1,:])*future_vb1Arr[1,1]
            v = [-c[1,0]+future_v0,-c[1,1]+future_v1]
            value0[i,1,t] = v[0]; 
            value1[i,1,t] = v[1]
            value[i,1,t] = np.max(v)
            
            if round(v[0],3)==round(v[1],3):policy[i,1,t]=2
            elif round(v[0],3)>round(v[1],3):policy[i,1,t]=0
            elif round(v[0],3)<round(v[1],3):policy[i,1,t]=1   
            
    return value0,value1,value,policy
                

#%%
#simpler case: two states, single changepoint
b = np.arange(0.0,1.001,0.001) #discrete belief space to use for b0 and b1
b = np.round(b,3)
etaL = 0.5; etaH = 0.9 #two levels of eta for the two internal states
I = np.array([0,1]) #internal state space : choose low or high eta levels
O = np.array([0,1]) #observation space: 0/1
s = np.array([0,1]) #environmental state space
I_N = np.array([0,1]) #states to choose at N (H0 or H1) 
PX_s = np.array([[[etaL,1-etaL],[1-etaL,etaL]],[[etaH,1-etaH],[1-etaH,etaH]]])
R = np.array([[1,0],[0,1]]) #R00,R01,R10,R11 (Rij = rewards on choosing Hi when Hj is true)
c00 = 0.00; c10 = 0.00; c01 = 0.02; c11 = 0.02
#magnitude of costs on going from i to j internal states
cost = np.array([[c00,c01],[c10,c11]])
p_signal = 0.5; 

n = 10 #trial 

value0,value1,value,policy=getOptimalPolicy(b,I,O,etaL,etaH,s,I_N,PX_s,R,cost,p_signal,n)

#%%
for j in range(10):
    i=n-j-1
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
#value map
sns.heatmap(value[:,0,:], yticklabels = 100, cmap='coolwarm')
plt.ylabel('belief state'); plt.xlabel('time'); 
plt.title('Value'); 
plt.figure()
sns.heatmap(value0[:,0,:]-value1[:,0,:],yticklabels = 100, cmap='coolwarm')
plt.ylabel('belief state'); plt.xlabel('time'); 
plt.title('Q0-Q1')
plt.figure()
sns.heatmap(value0[:,0,:],yticklabels = 100, cmap='coolwarm')
plt.ylabel('belief state'); plt.xlabel('time'); 
plt.title('Q0')
plt.figure()
sns.heatmap(value1[:,0,:],yticklabels = 100, cmap='coolwarm')
plt.ylabel('belief state'); plt.xlabel('time'); 
plt.title('Q1')
plt.figure()
sns.heatmap(policy[:,0,:],yticklabels = 100, cmap='coolwarm')
plt.ylabel('belief state'); plt.xlabel('time'); 
plt.title('policy')
plt.figure()

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
    
    return trial


def inferenceDiscretePolicy(trial,trial_length,p_signal, 
                            etaL,etaH,value0, value1,cost,b,db):
    n = trial_length  
    
    observation = np.full((int(round(n)),1),0)
    posterior = np.full((n+1,2),0.0) #posterior for states 0,1,2
    posterior[0,:] = [1.0,0.0] # posterior at j = -1
    posterior[0,:] = posterior[0,:]/(np.sum(posterior[0,:]))
    #array for internal state and eta
    internalState = np.full((n,1),0); internalState[0] = 0
    etaArr = np.full((n,1),0.0); etaArr[0] = etaL
    action = np.full((n,1),0); #array for action
    eta = [etaL,etaH] #the two eta levels
    
    #observation and inference for first n-1 timepoints
    for j in range(n-1):
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
        idx = int(posterior[j+1,1]/db)
        if idx < int(1/db):
            q0 = interpolate(posterior[j+1,1],b[idx], b[idx+1], db, 
                             value0[idx,internalState[j][0],j],
                             value0[idx+1,internalState[j][0],j])
            q1 = interpolate(posterior[j+1,1],b[idx], b[idx+1], db, 
                     value1[idx,internalState[j][0],j],
                     value1[idx+1,internalState[j][0],j])
            
        elif idx >= int(1/db):
            q0 = interpolate(posterior[j+1,1],b[idx], b[idx], db, 
                             value0[idx,internalState[j][0],j],
                             value0[idx,internalState[j][0],j])
            q1 = interpolate(posterior[j+1,1],b[idx], b[idx], db, 
                     value1[idx,internalState[j][0],j],
                     value1[idx,internalState[j][0],j])
        #print(q0,q1, posterior[j+1,1])
        if round(q0,4)==round(q1,3): internalState[j+1]=1; action[j]=2
        elif round(q0,4)>round(q1,3): internalState[j+1]=0; action[j]=0
        elif round(q0,4)<round(q1,3): internalState[j+1]=1; action[j]=1 
        
        etaArr[j+1] = eta[internalState[j+1][0]]
    
    #observation, inference for last timepoint
    if trial[n] == 0:
        observation[n-1] = np.random.binomial(1,1-etaArr[n-1])
    elif trial[n] == 1:
        observation[n-1] = np.random.binomial(1,etaArr[n-1])
        
    probStart = 1/((n/p_signal)-n+1)
    transition_matrix = np.array([[1-probStart,probStart],[0,1]])
    emission_matrix = np.array([[etaArr[n-1][0],1-etaArr[n-1][0]],
                                [1-etaArr[n-1][0],etaArr[n-1][0]]])
    emission_probability = emission_matrix[observation[n-1],:]
        
    posterior[n,:] = emission_probability*np.matmul(posterior[n-1,:],
                                                    transition_matrix)
    posterior[n,:] = posterior[n,:]/(np.sum(posterior[n,:]))

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
b = np.arange(0.0,1.001,0.001) #discrete belief space to use for b0 and b1
b = np.round(b,3)
etaL = 0.5; etaH = 0.9 #two levels of eta for the two internal states
I = np.array([0,1]) #internal state space : choose low or high eta levels
O = np.array([0,1]) #observation space: 0/1
s = np.array([0,1]) #environmental state space
I_N = np.array([0,1]) #states to choose at N (H0 or H1) 
PX_s = np.array([[[etaL,1-etaL],[1-etaL,etaL]],[[etaH,1-etaH],[1-etaH,etaH]]])
R = np.array([[1,0],[0,1]]) #R00,R01,R10,R11 (Rij = rewards on choosing Hi when Hj is true)
c00 = 0.00; c10 = 0.00; c01 = 0.05; c11 = 0.05
#magnitude of costs on going from i to j internal states
cost = np.array([[c00,c01],[c10,c11]])
p_signal = 0.5; 

n = 10 #trial 

value0,value1,value,policy=getOptimalPolicy(b,I,O,etaL,etaH,s,I_N,
                                            PX_s,R,cost,p_signal,n)

#%%
trial_length=10; p_signal = 0.5; 
signal_length_type = 0; signal_length=1
etaL = 0.7; etaH = 0.9
costs = np.array([[0.0,0.0],[0.02,0.02]])

b = np.arange(0.0,1.001,0.001) #discrete belief space to use for b0 and b1
b = np.round(b,3); db = 0.001

trial = generate_trialPolicy(trial_length, p_signal, 0,1)
observation,posterior, internalState, action = inferenceDiscretePolicy(
    trial,trial_length,p_signal, etaL,etaH,value0, value1,costs,b,db)
inferred_state,response, hit, miss, cr, fa  = generate_responseDiscretePolicy(
    trial,posterior)
action[10-1] = response


#%%
fig,ax = plt.subplots(2,1)
t = np.arange(1,trial_length+1,1)
ax[0].plot(t,trial[1:], label='underlying signal')
ax[0].scatter(t,observation, label='observations',color ='orange')
ax[0].plot(t,posterior[1:,1], label='posterior for s=1')

ax[1].plot(t,internalState, label= 'internal state', linestyle ='dashed', 
           marker ='o')
ax[1].plot(t,action, label = 'chosen action', linestyle ='dashed', 
           marker ='o')

ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left');
ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')