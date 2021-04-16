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
db = 0.001
b = np.arange(0.0,1.+2*db,db) #discrete belief space use for b0,b1 and b2
rounding = 3;
b = np.round(b,rounding)

etaL = 0.5; etaH = 0.9 #two levels of eta for the two internal states
I = np.array([0,1]) #internal state space : choose low or high eta levels
O = np.array([0,1]) #observation space: 0/1
s = np.array([0,1,2]) #environmental state space
I_N = np.array([0,1]) #states to choose at N (H0 or H1) 
PX_s = np.array([[[etaL,1-etaL,etaL],[1-etaL,etaL,1-etaL]],[[etaH,1-etaH,etaH],[1-etaH,etaH,1-etaH]]])
R = np.array([[1,0],[0,1]]) 
#R00,R01,R10,R11 (Rij = rewards on choosing Hi when Hj is true)
c00 = 0.00; c10 = 0.00; c01 = 0.03; c11 = 0.03
#magnitude of costs on going from i to j internal states
c = np.array([[c00,c01],[c10,c11]])

p_signal = 0.5; q = 0.1

n = 10 #trial length

compare = 1; beta = 30; 

#%%

value = np.full((len(b),len(b),len(I),n+1),np.nan) #value for each state for all n time steps
policy = np.full((len(b),len(b),len(I),n+1),np.nan) #corresponding policy

value0 = np.full((len(b),len(b),len(I),n+1),np.nan) #value for each state for all n time steps
value1 = np.full((len(b),len(b),len(I),n+1),np.nan) #value for each state for all n time steps


start = time.perf_counter()
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
            value0[i,int(1/db)-i,0,t] = 1-c[0,0]; value0[i,int(1/db)-i,1,t] = 1-c[1,0]; 
            value1[i,int(1/db)-i,0,t] = 1.-c[0,1]; value1[i,int(1/db)-i,1,t] = 1.-c[1,1];
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
            value0[i,int(1/db)-i,0,t] = 1-c[0,0]; value0[i,int(1/db)-i,1,t] = 1-c[1,0]; 
            value1[i,int(1/db)-i,0,t] = 1.-c[0,1]; value1[i,int(1/db)-i,1,t] = 1.-c[1,1];
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

            
print(time.perf_counter()-start) 


#%%

i = 6
plt.imshow(value[:,:,0,i], extent=[0,1,1,0]);
plt.ylabel('belief for signal'); plt.xlabel('belief for postsignal')
#plt.title('value, t=%d'%(i,)); plt.colorbar(); plt.figure()
plt.title('value, t=%d,q=%1.1f,costi1=%1.2f,etaL=%1.1f,etaH=%1.1f'%(i,
            q,c11,etaL,etaH)); plt.colorbar(); plt.figure()

plt.imshow(value0[:,:,0,i], extent=[0,1,1,0]);
plt.ylabel('belief for signal'); plt.xlabel('belief for postsignal')
plt.title('Q0, t=%d,q=%1.1f,costi1=%1.2f,eta=%1.1f'%(i,q,c11,etaL)); plt.colorbar(); plt.figure()

plt.imshow(value1[:,:,0,i], extent=[0,1,1,0]);
plt.ylabel('belief for signal'); plt.xlabel('belief for postsignal')
plt.title('Q1, t=%d,q=%1.1f,costi1=%1.2f,eta=%1.1f'%(i,q,c11,etaH)); plt.colorbar(); plt.figure()

plt.imshow(value0[:,:,0,i]-value1[:,:,0,i], extent=[0,1,1,0]);
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
mat = plt.imshow(policy[:,:,0,i],cmap=cmap,vmin =-0.5,vmax = 2.5, extent=[0,1,1,0])
plt.colorbar(mat,ticks=np.linspace(0,2,3)); 


#%%
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
plt.plot(value0[0,:,0,9])
plt.plot(value1[0,:,0,9])
