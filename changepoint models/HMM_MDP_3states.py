#importing required packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import seaborn as sns

#%%
def interpolate1d(x, x_0, x_1, dx,func_0,func_1):
    interpolated_func = 0.0
    interpolated_func = (((x-x_0)*func_1) + ((x_1-x)*func_0))/dx
    return interpolated_func

def interpolate2d(x, x_0, x_1, dx, y, y_0, y_1, dy, 
                  funcx0_y0,funcx1_y0,funcx0_y1,funcx1_y1):
    interpolate_y0 = interpolate1d(x,x_0,x_1,dx,funcx0_y0, funcx1_y0)
    interpolate_y1 = interpolate1d(x,x_0,x_1,dx,funcx0_y1, funcx1_y1)
    interpolated_func = interpolate1d(y,y_0,y_1,dy,interpolate_y0,interpolate_y1)
    return interpolated_func

def softmax(x, beta):
    """Compute softmax values for each sets of scores in x."""
    x = np.array(x)
    return np.exp(beta*x) / np.sum(np.exp(beta*x), axis=0)



#%%
def getPolicy(b,db,rounding,etaL,etaH,I,O,s,I_N,PX_s,R,cost,p_signal,q,
              trial_length,compare,beta):
    c = cost
    n = trial_length 
    value = np.full((len(b),len(b),len(I),n+1),np.nan) #value for each state for all n time steps
    policy = np.full((len(b),len(b),len(I),n+1),np.nan) #corresponding policy
    
    value0 = np.full((len(b),len(b),len(I),n+1),np.nan) #value for each state for all n time steps
    value1 = np.full((len(b),len(b),len(I),n+1),np.nan) #value for each state for all n time steps
    
    
    
    #optimal policy for last time step where IS doesn't matter
    for i in range(len(b)):#belief for state = 1
        j = 0
        while (b[i]+b[j])<=1:#belief for state = 2
            #immediate reward,position 0 is for choosing H0 and 1 is for choosing H1
            r = [(b[i]+b[j])*R[0,1]+(1-b[i]-b[j])*R[0,0], 
                 (b[i]+b[j])*R[1,1]+(1-b[i]-b[j])*R[1,0]] 
            value[i,j,0,n] = np.max(r) #maximum reward
            policy[i,j,0,n] = np.where(r == value[i,j,0,n])[0][0] #position/choice which gives max reward
            j=j+1
    value[:,:,1,n] = value[:,:,0,n]
    policy[:,:,1,n] = policy[:,:,0,n]    
    
    for t1 in range(n):
        t = n-1-t1
        probStart = 1/((n/p_signal)-t)
        transition_matrix = np.array([[1-probStart,probStart,0],[0,1-q,q],[0,0,1]])
        
        #current internal state = 0 
        for i in range(len(b)):#belief for state = 1
            j = 0
            iterj = 1;
            if etaH == 1:
                value[i,int(1/db)-i,0,t] =1.0; policy[i,int(1/db)-i,0,t] =1 #use only when etaH=1
                value[i,int(1/db)-i,1,t] =1.0; policy[i,int(1/db)-i,1,t] =1 #use only when etaH=1
                value0[i,int(1/db)-i,0,t] = 1-c[0,0]; 
                value1[i,int(1/db)-i,0,t] = 1.-c[0,1]; 
                iterj = 1-db
    
            while round(b[i]+b[j], rounding)<=iterj:#belief for state = 2
    
                bArr = [1-b[i]-b[j],b[i],b[j]]
                arr = np.array([[np.matmul(bArr,transition_matrix)*PX_s[0,0,:],
                                  np.matmul(bArr,transition_matrix)*PX_s[0,1,:]],
                                [np.matmul(bArr,transition_matrix)*PX_s[1,0,:],
                                 np.matmul(bArr,transition_matrix)*PX_s[1,1,:]]])
    
                b1Arr = np.array([[arr[0,0,:]/np.sum(arr[0,0,:]),
                                  arr[0,1,:]/np.sum(arr[0,1,:])],
                                [arr[1,0,:]/np.sum(arr[1,0,:]),
                                 arr[1,1,:]/np.sum(arr[1,1,:])]])
                closest_index = np.array(b1Arr/db, dtype=int)
                
                if (closest_index[:,:,1]+closest_index[:,:,2]>= len(b)-3).any():
                    futurevx0_y0 = np.array([[value[closest_index[0,0,1],closest_index[0,0,2],0,t+1],
                                    value[closest_index[0,1,1],closest_index[0,1,2],0,t+1]],
                                    [value[closest_index[1,0,1],closest_index[1,0,2],1,t+1],
                                    value[closest_index[1,1,1],closest_index[1,1,2],1,t+1]]])
                    futurevx0_y1 = np.array([[value[closest_index[0,0,1],closest_index[0,0,2],0,t+1],
                                    value[closest_index[0,1,1],closest_index[0,1,2],0,t+1]],
                                    [value[closest_index[1,0,1],closest_index[1,0,2],1,t+1],
                                    value[closest_index[1,1,1],closest_index[1,1,2],1,t+1]]])
                    futurevx1_y0 = np.array([[value[closest_index[0,0,1],closest_index[0,0,2],0,t+1],
                                    value[closest_index[0,1,1],closest_index[0,1,2],0,t+1]],
                                    [value[closest_index[1,0,1],closest_index[1,0,2],1,t+1],
                                    value[closest_index[1,1,1],closest_index[1,1,2],1,t+1]]])
                    futurevx1_y1 = np.array([[value[closest_index[0,0,1],closest_index[0,0,2],0,t+1],
                                    value[closest_index[0,1,1],closest_index[0,1,2],0,t+1]],
                                    [value[closest_index[1,0,1],closest_index[1,0,2],1,t+1],
                                    value[closest_index[1,1,1],closest_index[1,1,2],1,t+1]]])
        
        
                    future_vb1Arr = futurevx0_y0
                    
                else: 
                    futurevx0_y0 = np.array([[value[closest_index[0,0,1],closest_index[0,0,2],0,t+1],
                                    value[closest_index[0,1,1],closest_index[0,1,2],0,t+1]],
                                    [value[closest_index[1,0,1],closest_index[1,0,2],1,t+1],
                                    value[closest_index[1,1,1],closest_index[1,1,2],1,t+1]]])
                    futurevx0_y1 = np.array([[value[closest_index[0,0,1],closest_index[0,0,2]+1,0,t+1],
                                    value[closest_index[0,1,1],closest_index[0,1,2]+1,0,t+1]],
                                    [value[closest_index[1,0,1],closest_index[1,0,2]+1,1,t+1],
                                    value[closest_index[1,1,1],closest_index[1,1,2]+1,1,t+1]]])
                    futurevx1_y0 = np.array([[value[closest_index[0,0,1]+1,closest_index[0,0,2],0,t+1],
                                    value[closest_index[0,1,1]+1,closest_index[0,1,2],0,t+1]],
                                    [value[closest_index[1,0,1]+1,closest_index[1,0,2],1,t+1],
                                    value[closest_index[1,1,1]+1,closest_index[1,1,2],1,t+1]]])
                    futurevx1_y1 = np.array([[value[closest_index[0,0,1]+1,closest_index[0,0,2]+1,0,t+1],
                                    value[closest_index[0,1,1]+1,closest_index[0,1,2]+1,0,t+1]],
                                    [value[closest_index[1,0,1]+1,closest_index[1,0,2]+1,1,t+1],
                                    value[closest_index[1,1,1]+1,closest_index[1,1,2]+1,1,t+1]]])
        
        
                    future_vb1Arr = interpolate2d(b1Arr[:,:,1], b[closest_index[:,:,1]],
                                    b[closest_index[:,:,1]+1], db, b1Arr[:,:,2], 
                                    b[closest_index[:,:,2]], b[closest_index[:,:,2]+1],
                                    db,futurevx0_y0,futurevx0_y1,futurevx1_y0,futurevx1_y1)
                
                v0 = np.sum(arr[0,0,:])*future_vb1Arr[0,0]+np.sum(
                    arr[0,1,:])*future_vb1Arr[0,1]
                v1 = np.sum(arr[1,0,:])*future_vb1Arr[1,0]+np.sum(
                    arr[1,1,:])*future_vb1Arr[1,1]
                v = np.array([-c[0,0]+v0,-c[0,1]+v1])
                value0[i,j,0,t] = v[0]; 
                value1[i,j,0,t] = v[1]
              
                
                #choose optimal action
                if compare == 0: #max
                
                    value[i,j,0,t] = np.max([round(v[0],rounding), round(v[1],rounding)])
                    
                    if round(v[0],rounding)==round(v[1],rounding):policy[i,j,0,t]=2
                    elif round(v[0],rounding)>round(v[1],rounding):policy[i,j,0,t]=0
                    elif round(v[0],rounding)<round(v[1],rounding):policy[i,j,0,t]=1   
                           
                elif compare == 1:#soft max
                    
                    r = np.round(softmax(v,beta),rounding)
                     
                    value[i,j,0,t] = np.round(v[np.where(r == np.max(
                        r))[0][0]],rounding)   
                    
                    if r[0]== r[1]:policy[i,j,0,t]=2
                    elif r[0]>r[1]:policy[i,j,0,t]=0
                    elif r[0]<r[1]:policy[i,j,0,t]=1   
                
                j = j+1
                
        #current internal state = 1
        for i in range(len(b)):#belief for state = 1
            j = 0
            iterj = 1;
            if etaH == 1:
                value[i,int(1/db)-i,0,t] =1.0; policy[i,int(1/db)-i,0,t] =1 #use only when etaH=1
                value[i,int(1/db)-i,1,t] =1.0; policy[i,int(1/db)-i,1,t] =1 #use only when etaH=1
                value0[i,int(1/db)-i,1,t] = 1-c[1,0]; 
                value1[i,int(1/db)-i,1,t] = 1.-c[1,1];
                iterj = 1-db
    
            while round(b[i]+b[j], rounding)<=iterj:#belief for state = 2
    
                bArr = [1-b[i]-b[j],b[i],b[j]]
                arr = np.array([[np.matmul(bArr,transition_matrix)*PX_s[0,0,:],
                                  np.matmul(bArr,transition_matrix)*PX_s[0,1,:]],
                                [np.matmul(bArr,transition_matrix)*PX_s[1,0,:],
                                 np.matmul(bArr,transition_matrix)*PX_s[1,1,:]]])
    
                b1Arr = np.array([[arr[0,0,:]/np.sum(arr[0,0,:]),
                                  arr[0,1,:]/np.sum(arr[0,1,:])],
                                [arr[1,0,:]/np.sum(arr[1,0,:]),
                                 arr[1,1,:]/np.sum(arr[1,1,:])]])
                closest_index = np.array(b1Arr/db, dtype=int)
                
                if (closest_index[:,:,1]+closest_index[:,:,2]>= len(b)-3).any():
                    futurevx0_y0 = np.array([[value[closest_index[0,0,1],closest_index[0,0,2],0,t+1],
                                    value[closest_index[0,1,1],closest_index[0,1,2],0,t+1]],
                                    [value[closest_index[1,0,1],closest_index[1,0,2],1,t+1],
                                    value[closest_index[1,1,1],closest_index[1,1,2],1,t+1]]])
                    futurevx0_y1 = np.array([[value[closest_index[0,0,1],closest_index[0,0,2],0,t+1],
                                    value[closest_index[0,1,1],closest_index[0,1,2],0,t+1]],
                                    [value[closest_index[1,0,1],closest_index[1,0,2],1,t+1],
                                    value[closest_index[1,1,1],closest_index[1,1,2],1,t+1]]])
                    futurevx1_y0 = np.array([[value[closest_index[0,0,1],closest_index[0,0,2],0,t+1],
                                    value[closest_index[0,1,1],closest_index[0,1,2],0,t+1]],
                                    [value[closest_index[1,0,1],closest_index[1,0,2],1,t+1],
                                    value[closest_index[1,1,1],closest_index[1,1,2],1,t+1]]])
                    futurevx1_y1 = np.array([[value[closest_index[0,0,1],closest_index[0,0,2],0,t+1],
                                    value[closest_index[0,1,1],closest_index[0,1,2],0,t+1]],
                                    [value[closest_index[1,0,1],closest_index[1,0,2],1,t+1],
                                    value[closest_index[1,1,1],closest_index[1,1,2],1,t+1]]])
        
        
                    future_vb1Arr = futurevx0_y0 
                    
                else: 
                    futurevx0_y0 = np.array([[value[closest_index[0,0,1],closest_index[0,0,2],0,t+1],
                                    value[closest_index[0,1,1],closest_index[0,1,2],0,t+1]],
                                    [value[closest_index[1,0,1],closest_index[1,0,2],1,t+1],
                                    value[closest_index[1,1,1],closest_index[1,1,2],1,t+1]]])
                    futurevx0_y1 = np.array([[value[closest_index[0,0,1],closest_index[0,0,2]+1,0,t+1],
                                    value[closest_index[0,1,1],closest_index[0,1,2]+1,0,t+1]],
                                    [value[closest_index[1,0,1],closest_index[1,0,2]+1,1,t+1],
                                    value[closest_index[1,1,1],closest_index[1,1,2]+1,1,t+1]]])
                    futurevx1_y0 = np.array([[value[closest_index[0,0,1]+1,closest_index[0,0,2],0,t+1],
                                    value[closest_index[0,1,1]+1,closest_index[0,1,2],0,t+1]],
                                    [value[closest_index[1,0,1]+1,closest_index[1,0,2],1,t+1],
                                    value[closest_index[1,1,1]+1,closest_index[1,1,2],1,t+1]]])
                    futurevx1_y1 = np.array([[value[closest_index[0,0,1]+1,closest_index[0,0,2]+1,0,t+1],
                                    value[closest_index[0,1,1]+1,closest_index[0,1,2]+1,0,t+1]],
                                    [value[closest_index[1,0,1]+1,closest_index[1,0,2]+1,1,t+1],
                                    value[closest_index[1,1,1]+1,closest_index[1,1,2]+1,1,t+1]]])
        
        
                    future_vb1Arr = interpolate2d(b1Arr[:,:,1], b[closest_index[:,:,1]],
                                    b[closest_index[:,:,1]+1], db, b1Arr[:,:,2], 
                                    b[closest_index[:,:,2]], b[closest_index[:,:,2]+1],
                                    db,futurevx0_y0,futurevx0_y1,futurevx1_y0,futurevx1_y1)
    
                v0 = np.sum(arr[0,0,:])*future_vb1Arr[0,0]+np.sum(
                    arr[0,1,:])*future_vb1Arr[0,1]
                v1 = np.sum(arr[1,0,:])*future_vb1Arr[1,0]+np.sum(
                    arr[1,1,:])*future_vb1Arr[1,1]
                v = np.array([-c[1,0]+v0,-c[1,1]+v1])
                value0[i,j,1,t] = v[0]; 
                value1[i,j,1,t] = v[1]
                
                #choose optimal action
                if compare == 0: #max
                
                    value[i,j,1,t] = np.max([round(v[0],rounding), round(v[1],rounding)])
                    
                    if round(v[0],rounding)==round(v[1],rounding):policy[i,j,1,t]=2
                    elif round(v[0],rounding)>round(v[1],rounding):policy[i,j,1,t]=0
                    elif round(v[0],rounding)<round(v[1],rounding):policy[i,j,1,t]=1   
                           
                elif compare == 1:#soft max
                    
                    r = np.round(softmax(v,beta),rounding)
                     
                    value[i,j,1,t] = np.round(v[np.where(r == np.max(
                        r))[0][0]],rounding)   
                    
                    if r[0]== r[1]:policy[i,j,1,t]=2
                    elif r[0]>r[1]:policy[i,j,1,t]=0
                    elif r[0]<r[1]:policy[i,j,1,t]=1   
                
                j = j+1
    
    return value,value0,value1,policy

           

#%%
#use the function, getPolicy
db = 0.01
b = np.arange(0.0,1.+2*db,db) #discrete belief space use for b0,b1 and b2
rounding = 2;
b = np.round(b,rounding)

etaL = 0.6; etaH = 0.9 #two levels of eta for the two internal states

I = np.array([0,1]) #internal state space : choose low or high eta levels
O = np.array([0,1]) #observation space: 0/1
s = np.array([0,1,2]) #environmental state space
I_N = np.array([0,1]) #states to choose at N (H0 or H1) 
PX_s = np.array([[[etaL,1-etaL,etaL],[1-etaL,etaL,1-etaL]],[[etaH,1-etaH,etaH],[1-etaH,etaH,1-etaH]]])
R = np.array([[1,0],[0,1]]) 
#R00,R01,R10,R11 (Rij = rewards on choosing Hi when Hj is true)

c00 = 0.00; c10 = 0.00; c01 = 0.02; c11 = 0.02
#magnitude of costs on going from i to j internal states
cost = np.array([[c00,c01],[c10,c11]])
p_signal = 0.2; q = 0.3
trial_length = 10 #trial length

compare = 1; beta = 50; 


start = time.perf_counter()

value,value0,value1,policy = getPolicy(b,db,rounding,etaL,etaH,I,O,s,I_N,PX_s,R,cost,p_signal,q,
              trial_length,compare,beta)

print(time.perf_counter()-start) 

#%%

i = 9; j = 0
plt.imshow(value[:,:,j,i], extent=[0,1,1,0]);

plt.ylabel('belief for signal'); plt.xlabel('belief for postsignal')
#plt.title('value, t=%d'%(i,)); plt.colorbar(); plt.figure()
plt.title('value, t=%d,q=%1.1f,costi1=%1.2f,etaL=%1.1f,etaH=%1.1f'%(i,
            q,c11,etaL,etaH)); plt.colorbar(); plt.figure()

plt.imshow(value0[:,:,j,i], extent=[0,1,1,0]);
plt.ylabel('belief for signal'); plt.xlabel('belief for postsignal')
plt.title('Q0, t=%d,q=%1.1f,costi1=%1.2f,eta=%1.1f'%(i,q,c11,etaL)); plt.colorbar(); plt.figure()

plt.imshow(value1[:,:,j,i], extent=[0,1,1,0]);
plt.ylabel('belief for signal'); plt.xlabel('belief for postsignal')
plt.title('Q1, t=%d,q=%1.1f,costi1=%1.2f,eta=%1.1f'%(i,q,c11,etaH)); plt.colorbar(); plt.figure()

plt.imshow(value0[:,:,j,i]-value1[:,:,j,i], extent=[0,1,1,0]);
plt.ylabel('belief for signal'); plt.xlabel('belief for postsignal')
plt.title('Q0-Q1, t=%d,q=%1.1f,costi1=%1.2f,etaL=%1.1f,etaH=%1.1f'%(i,
                 q,c11,etaL,etaH)); plt.colorbar(); plt.figure()


#cmap = matplotlib.colors.ListedColormap(colorsList)
cmap = matplotlib.cm.get_cmap('viridis', 3)
norm = matplotlib.colors.BoundaryNorm(np.arange(0, 2, 1), cmap.N)
plt.ylabel('belief for signal'); plt.xlabel('belief for postsignal')
#plt.title('policy, t=%d'%(i,)); 
plt.title('policy, t=%d,q=%1.1f,costi1=%1.2fetaL=%1.1f,etaH=%1.1f'%(i,
                q,c11,etaL,etaH)); 

mat = plt.imshow(policy[:,:,j,i],cmap=cmap,vmin =-0.5,vmax = 2.5, extent=[0,1,1,0])
plt.colorbar(mat,ticks=np.linspace(0,2,3)); 


#%%
#slices of value functions at specific b(1)
t = 8
plt.plot(b,value0[0,:,0,t], label = 'value0, b(1)=0, t=%d'%(t))
plt.plot(b,value1[0,:,0,t], label = 'value1, b(1)=0, t=%d'%(t))
plt.legend(); plt.xlabel('b(2)'); plt.figure()
plt.plot(b,value0[300,:,0,t], label = 'value0, b(1)=0.3, t=%d'%(t))
plt.plot(b,value1[300,:,0,t], label = 'value1, b(1)=0.3, t=%d'%(t))
plt.legend(); plt.xlabel('b(2)'); plt.figure()
plt.plot(b,value0[600,:,0,t], label = 'value0, b(1)=0.6, t=%d'%(t))
plt.plot(b,value1[600,:,0,t], label = 'value1, b(1)=0.6, t=%d'%(t))
plt.legend(); plt.xlabel('b(2)'); plt.figure()
plt.plot(b,value0[900,:,0,t], label = 'value0, b(1)=0.9, t=%d'%(t))
plt.plot(b,value1[900,:,0,t], label = 'value1, b(1)=0.9, t=%d'%(t))
plt.legend(); plt.xlabel('b(2)'); plt.figure()


#%%
#projections of these value functions on particular belief states
plt.imshow(value[[20,20,20,50,50,80],
                 [20,50,80,20,50,20],0,:])
plt.yticks(np.arange(6),['[0.2,0.2]','[0.2,0.5]','[0.2,0.8]',
                         '[0.5,0.2]','[0.5,0.5]','[0.8,0.2]'])
plt.xlabel('time'); plt.ylabel('[b(1),b(2)]')
plt.colorbar()

#%%
#forward run
#inference with executing the policy: 3 states
def generate_trialPolicy(trial_length, p_signal, q,
                         signal_length_type,signal_length):
    n = trial_length
    trial_signal = np.random.binomial(1,p_signal) #trial is signal/non signal with prob 0.5
    
    if signal_length_type == 0:
        start_signal = np.random.randint(0, n) #index of sample when signal begins (on signal trial)
        end_signal = np.random.geometric(q, size=1) #time points for which signal will stay on
    elif signal_length_type == 1:
        start_signal = np.random.randint(0, n-signal_length+1) #index of sample when signal begins (on signal trial)
        end_signal = signal_length #length of signal is fixed
    
    trial = np.full((int(round(n))+1,1),0);#environmental states within a trial
    if trial_signal == 1: 
        r = min(end_signal, n-start_signal)
        trial[start_signal+1:start_signal+int(r)+1] = np.full((int(r),1),1)
        trial[start_signal+int(r)+1:n+1] = np.full((n-start_signal-int(r),1),2) 
              
    return trial, trial_signal

def inferenceDiscretePolicy(trial,trial_length,p_signal,etaL,etaH,
                            value0,value1,cost,b,db,beta,rounding):
        
    n = trial_length  
    
    observation = np.full((int(round(n)),1),0)
    posterior = np.full((n+1,3),0.0) #posterior for states 0,1,2
    posterior[0,:] = [1.0,0.0,0.0] # posterior at j = -1
    posterior[0,:] = posterior[0,:]/(np.sum(posterior[0,:]))
    #array for internal state and eta
    internalState = np.full((n+1,1),0); #IS 
    internalState[0] = 0 #initialise initial IS at j=-1 ot t=0
    etaArr = np.full((n,1),0.0) #corresponding eta (determined by IS)
    action = np.full((n,1),0); #array for action = IS for each t>0
    eta = [etaL,etaH] #the two eta levels
    
    
    for j in range(n): #t=j+1
        
        #choosing action
        idx = np.array(posterior[j,1:]/db, dtype=int)
        if np.sum(idx) < int(1/db):
            q0 = interpolate2d(posterior[j,1],b[idx[0]],b[idx[0]+1],db,
                               posterior[j,2],b[idx[1]],b[idx[1]+1],db,
                               value0[idx[0],idx[1],internalState[j][0],j],
                               value0[idx[0],idx[1]+1,internalState[j][0],j],
                               value0[idx[0]+1,idx[1],internalState[j][0],j],
                               value0[idx[0]+1,idx[1]+1,internalState[j][0],j])
            q1 = interpolate2d(posterior[j,1],b[idx[0]],b[idx[0]+1],db,
                               posterior[j,2],b[idx[1]],b[idx[1]+1],db,
                               value1[idx[0],idx[1],internalState[j][0],j],
                               value1[idx[0],idx[1]+1,internalState[j][0],j],
                               value1[idx[0]+1,idx[1],internalState[j][0],j],
                               value1[idx[0]+1,idx[1]+1,internalState[j][0],j])
            
        elif np.sum(idx) >= int(1/db):
            q0 = value0[idx[0],idx[1],internalState[j][0],j]
            q1 = value1[idx[0],idx[1],internalState[j][0],j]
            
        Q = np.array([q0,q1])
        r = np.round(softmax(Q,beta),rounding)
        if r[0]==r[1]:internalState[j+1]=1; action[j] = 1; etaArr[j] = eta[1]
        elif r[0]>r[1]:internalState[j+1]=0; action[j] = 0; etaArr[j] = eta[0]
        elif r[0]<r[1]:internalState[j+1]=1; action[j] = 1;  etaArr[j] = eta[1] 

        #observation based on new internal state
        if trial[j+1] == 0:
            observation[j] = np.random.binomial(1,1-etaArr[j])
        elif trial[j+1] == 1:
            observation[j] = np.random.binomial(1,etaArr[j])
        elif trial[j+1] == 2:
            observation[j] = np.random.binomial(1,1-etaArr[j])

        #transition matrix
        probStart = 1/((n/p_signal)-j)
        transition_matrix = np.array([[1-probStart,probStart,0],
                                      [0,1-q,q],[0,0,1]])
        #emission matrix and probability
        emission_matrix = np.array([[etaArr[j][0],1-etaArr[j][0],etaArr[j][0]],
                                    [1-etaArr[j][0],etaArr[j][0],1-etaArr[j][0]]])
        emission_probability = emission_matrix[observation[j][0]]
        
        #belief update
        posterior[j+1,:] = emission_probability*np.matmul(posterior[j,:],transition_matrix)
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

trial_length = 100;
p_signal = 0.5; q = 0.1
etaL = 0.6; etaH = 0.9 #two levels of eta for the two internal states
signal_length_type = 0; signal_length = 10
db = 0.01
b = np.arange(0.0,1.+2*db,db) #discrete belief space use for b0,b1 and b2
rounding = 3;
b = np.round(b,rounding)

c00 = 0.00; c10 = 0.00; c01 = 0.03; c11 = 0.02
#magnitude of costs on going from i to j internal states
cost = np.array([[c00,c01],[c10,c11]])
p_signal = 0.5; q = 0.05
compare = 1; beta = 50; 

I = np.array([0,1]) #internal state space : choose low or high eta levels
O = np.array([0,1]) #observation space: 0/1
s = np.array([0,1,2]) #environmental state space
I_N = np.array([0,1]) #states to choose at N (H0 or H1) 
PX_s = np.array([[[etaL,1-etaL,etaL],[1-etaL,etaL,1-etaL]],[[etaH,1-etaH,etaH],[1-etaH,etaH,1-etaH]]])
R = np.array([[1,0],[0,1]]) 
#R00,R01,R10,R11 (Rij = rewards on choosing Hi when Hj is true)


start = time.perf_counter()

value,value0,value1,policy = getPolicy(b,db,rounding,etaL,etaH,I,O,s,I_N,PX_s,R,cost,p_signal,q,
              trial_length,compare,beta)

print(time.perf_counter()-start) 

#%%
signal_length_type = 0; signal_length = 1


trial, trial_signal = generate_trialPolicy(trial_length, p_signal, q,
                         signal_length_type,signal_length)
observation, posterior, internalState, action = inferenceDiscretePolicy(trial,trial_length,p_signal,etaL,etaH,
                            value0,value1,cost,b,db,beta,rounding)
inferred_state,response,hit,miss,cr,fa = generate_responseDiscretePolicy(trial,posterior)


#%%
fig,ax = plt.subplots(2,1)
t = np.arange(0,trial_length+1,1)
ax[0].plot(t[1:],trial[1:], label='underlying signal')
ax[0].scatter(t[1:],observation, label='observations',color ='orange')
ax[0].plot(t,posterior[:,1], label='posterior for s=1')
ax[0].plot(t,posterior[:,2], label='posterior for s=2')

ax[1].plot(t,internalState[:], label= 'internal state', linestyle ='dashed', 
           marker ='o')
ax[1].plot(t[1:],action, label = 'action', linestyle ='dashed', 
           marker ='o')

ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left');
ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

ax[0].set_title('etaH=%1.2f, etaL=%1.2f, c01=%1.2f, c11=%1.2f, q=%1.2f'
                              %(etaH,etaL,c01,c11,q)) 

#%%

nTrials = 3000
posteriorTrials = np.full((nTrials,trial_length+1,3),0.0)
actionTrials = np.full((nTrials,trial_length,1),0.0)
trialType = np.full((nTrials,1),0.0)

for nT in range(nTrials):
    trial, trial_signal = generate_trialPolicy(trial_length, p_signal, q,
                             signal_length_type,signal_length)
    observation, posterior, internalState, action = inferenceDiscretePolicy(trial,trial_length,p_signal,etaL,etaH,
                            value0,value1,cost,b,db,beta,rounding)
    inferred_state,response,hit,miss,cr,fa = generate_responseDiscretePolicy(trial,posterior)
    posteriorTrials[nT,:,:] = posterior
    trialType[nT] = trial_signal
    actionTrials[nT,:,:] = action
    
#%%
t = np.arange(0,trial_length+1,1)
#for signal trials
avgPosterior = np.average(posteriorTrials[np.where(trialType==1)[0],:,:],axis=0)
avgAction = np.average(actionTrials[np.where(trialType==1)[0],:,:],axis=0)
plt.plot(t,avgPosterior[:,1],label ='b(1)') 
plt.plot(t,avgPosterior[:,2],label ='b(2)') 
plt.plot(t[1:],avgAction[:,0],label ='action',marker='o')
plt.legend()
plt.xlabel('time'); 
plt.title('Avg runs--signal trials, etaH=%1.2f, etaL=%1.2f, c01=%1.2f, c11=%1.2f, q=%1.2f'
                              %(etaH,etaL,c01,c11,q)) 

plt.figure()
 
#for signal trials
avgPosterior = np.average(posteriorTrials[np.where(trialType==0)[0],:,:],axis=0)
avgAction = np.average(actionTrials[np.where(trialType==0)[0],:,:],axis=0)
plt.plot(t,avgPosterior[:,1],label ='b(1)') 
plt.plot(t,avgPosterior[:,2],label ='b(2)') 
plt.plot(t[1:],avgAction[:,0],label ='action',marker='o')
plt.legend()
plt.xlabel('time'); plt.title('Avg runs--non-signal trials, etaH=%1.2f, etaL=%1.2f,c01=%1.2f, c11=%1.2f,,q=%1.1f'
                              %(etaH,etaL,c01,c11,q))       


#%%
#looking at performance at different q and trial length

#parameters:
c00 = 0.00; c10 = 0.00; c01 = 0.02; c11 = 0.02
#magnitude of costs on going from i to j internal states
cost = np.array([[c00,c01],[c10,c11]])
etaL = 0.6; etaH = 0.9 #two levels of eta for the two internal states
compare = 1; beta = 50;  #use softmax with beta
trial_length = 10;
p_signal = 0.5; q = 0.1
signal_length_type = 0; signal_length = 10

db = 0.01
b = np.arange(0.0,1.+2*db,db) #discrete belief space use for b0,b1 and b2
rounding = 2;
b = np.round(b,rounding)

nTrials = 100

trial_lengthArr = [1,5,10,25,50] #75,100,125
qArr = [0.01,0.2,0.5,0.7] #0.01,0.2,0.5,0.7
trialTypeRates = np.full((len(qArr),len(trial_lengthArr),4),np.nan)

start = time.perf_counter()

for t in range(len(trial_lengthArr)):
    trial_length= trial_lengthArr[t]
    for s in range(len(qArr)) :
        q = qArr[s]
        trial_type = np.full((nTrials,3),0) #signal trial or not, start point of signal, signal len
        hit = 0; miss = 0; cr = 0; fa = 0
        value,value0,value1,policy = getPolicy(b,db,rounding,etaL,etaH,I,O,s,I_N,PX_s,R,cost,p_signal,q,
              trial_length,compare,beta)
        #trials:
        for k in range(nTrials):        

            trial, trial_signal = generate_trialPolicy(trial_length, p_signal, q,
                                     signal_length_type,signal_length)
            observation, posterior, internalState, action = inferenceDiscretePolicy(trial,trial_length,p_signal,etaL,etaH,
                                    value0,value1,cost,b,db,beta,rounding)
            inferred_state,response,hit0,miss0,cr0,fa0 = generate_responseDiscretePolicy(trial,posterior)
            
            hit = hit+hit0; miss = miss+miss0; cr = cr+cr0; fa = fa+fa0
            
        trialTypeRates[s,t,0] = hit; trialTypeRates[s,t,1] = miss;
        trialTypeRates[s,t,2] = cr; trialTypeRates[s,t,3] = fa;
        s = s+1

print(time.perf_counter()-start) 

#%%
#plot performance rates
for l in range(len(qArr)):

    a = trialTypeRates[l,:,0]/(trialTypeRates[l,:,0]+trialTypeRates[l,:,1])
    plt.plot(trial_lengthArr,a, marker = 'o', label = 'q=%1.3f'%qArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('trial length')
plt.ylabel('hit rates'); 
plt.title('etaH=%1.2f, etaL=%1.2f,c01=%1.2f, c11=%1.2f,,q=%1.1f'
                              %(etaH,etaL,c01,c11,q)) 
plt.figure()

for l in range(len(qArr)):

    a = trialTypeRates[l,:,3]/(trialTypeRates[l,:,2]+trialTypeRates[l,:,3])
    plt.plot(trial_lengthArr,a, marker = 'o', label = 'q=%1.3f'%qArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('trial length')
plt.ylabel('fa rates'); 
plt.title('etaH=%1.2f, etaL=%1.2f,c01=%1.2f, c11=%1.2f,,q=%1.1f'
                              %(etaH,etaL,c01,c11,q)) 
plt.figure()

for l in range(len(trial_lengthArr)):

    a = trialTypeRates[:,l,0]/(trialTypeRates[:,l,0]+trialTypeRates[:,l,1])
    plt.plot(qArr,a, marker = 'o', label = 'trial_length=%d'%trial_lengthArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('signal length')
plt.ylabel('hit rates');
plt.title('etaH=%1.2f, etaL=%1.2f,c01=%1.2f, c11=%1.2f,,q=%1.1f'
                              %(etaH,etaL,c01,c11,q)) 
plt.figure()

for l in range(len(trial_lengthArr)):

    a = trialTypeRates[:,l,3]/(trialTypeRates[:,l,2]+trialTypeRates[:,l,3])
    plt.plot(qArr,a, marker = 'o', label = 'trial_length=%d'%trial_lengthArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('signal length')
plt.ylabel('fa rates'); 
plt.title('etaH=%1.2f, etaL=%1.2f,c01=%1.2f, c11=%1.2f,,q=%1.1f'
                              %(etaH,etaL,c01,c11,q)) 
plt.figure()

#%%
#obselete code
func = generate_trialPolicy  
trial_length = 10 #trial length
p_signal = 0.5 #prob of signal trial
q = 0.1 #constant probability of leaving
nTrials = 1000
signal_length_type = 0; signal_length = 10
hit =0; cr=0; miss=0; fa=0
   
trial_type = np.full((nTrials,3),0)
for k in range(nTrials):
    trial = func(trial_length, p_signal, q, signal_length_type, signal_length)
    observation, posterior, internalState, action = inferenceDiscretePolicy(trial,trial_length,p_signal,etaL,etaH,
                            value0,value1,cost,b,db,beta,rounding)
    inferred_state,response,hit,miss,cr,fa = generate_responseDiscretePolicy(trial,posterior)

    trial_signal = 0
    if sum(trial) > 0: trial_signal =1 
    if trial_signal == 1:
        trial_type[k,0] = 1; 
        trial_type[k,1] = np.intersect1d(np.where(trial[1:] == 1)[0], 
                                     np.where(trial[:-1] == 0)[0]) #start signal
        trial_type[k,2] = len(np.where(trial == 1)[0]) #signal length
    