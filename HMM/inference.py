import numpy as np
import matplotlib.pyplot as plt
#import time

#%%

#response at the end of a stimulus window
n = 0.6
obs = np.loadtxt('obs.txt') #load observation sequence
response = 0
f = np.full((len(obs)+1,3), np.nan) #array of f values for start, cue, non-cue
p = np.full((len(obs)+1,3), np.nan) #array of posterior prob for start, cue, non-cue
transition1 = np.array([[1, 0, 0],[0,1,0],[0,0,1]]) #transition probabilities
e = np.array([[1,0,0],[0,n,1-n],[0,1-n,n]]) #emission probabilities

#inital f and p values
f[0,:] = np.array([1,0,0]) 
p[0,:] = f[0,:]/np.sum(f[0,:])

for i in range(len(obs)):
    if i in [1, 11, 21, 31, 41, 51, 61] and f[i,0]>0:
        #transition probability at above timesteps
        transition2 = np.array([[1-(10/(70-i)), 5/(70-i), 5/(70-i)],[0,1,0],[0,0,1]])
        f[i+1, :] = e[int(obs[i]),:]*np.matmul(f[i, :], transition2)
    else:
        #calculaitng joint probabilities
        f[i+1, :] = e[int(obs[i]),:]*np.matmul(f[i, :], transition1)
    
    
    #prob of state at i+1 given observations till now
    p[i+1, :] = f[i+1,:]/np.sum(f[i+1,:])
    
if p[len(obs),1] > 0.6: #cue is 1
    response = 1
elif p[len(obs), 2] > 0.6: #noncue is 2
    response = 2  


#%%

#response at a treshold
n = 0.6
obs = np.loadtxt('obs.txt') #load observation sequence
response = 0
f = np.full((len(obs)+1,3), np.nan) #array of f values for start, cue, non-cue
p = np.full((len(obs)+1,3), np.nan) #array of posterior prob for start, cue, non-cue
transition1 = np.array([[1, 0, 0],[0,1,0],[0,0,1]]) #transition probabilities
e = np.array([[1,0,0],[0,n,1-n],[0,1-n,n]]) #emission probabilities

#inital f and p values
f[0,:] = np.array([1,0,0]) 
p[0,:] = f[0,:]/np.sum(f[0,:])

#starttime = time.perf_counter()

for i in range(len(obs)):
    if p[i, 1] > 0.9:
        response = 1 #resposne to cue = 1
        break
    elif p[i, 2] > 0.9:
        response = 2 #response to no cue = 2    
        break        
    if i in [1, 11, 21, 31, 41, 51, 61] and f[i,0]>0:
        #transition probability at above timesteps
        transition2 = np.array([[1-(10/(70-i)), 5/(70-i), 5/(70-i)],[0,1,0],[0,0,1]])
        f[i+1, :] = e[int(obs[i]),:]*np.matmul(f[i, :], transition2)
        startTime = i #time step number at which it leaves start
    else:
        #calculaitng joint probabilities
        f[i+1, :] = e[int(obs[i]),:]*np.matmul(f[i, :], transition1)
    
    
    #prob of state at i+1 given observations till now
    p[i+1, :] = f[i+1,:]/np.sum(f[i+1,:])
    
rt = (i - startTime)*0.01 #reaction time is time taken from startTime
                          #to cross a threshold posterior
    
#print(time.perf_counter()-starttime) 

#%%
timeAxis = np.linspace(0, (i+1)*0.01, i+1)
plt.plot(timeAxis, p[1:i+2,0]); plt.xlabel('time'); plt.ylabel('prob of start')
plt.figure()
plt.plot(timeAxis, p[1:i+2,1]); plt.xlabel('time'); plt.ylabel('prob of cue')
plt.figure()
plt.plot(timeAxis, p[1:i+2,2]); plt.xlabel('time'); plt.ylabel('prob of no cue')
plt.figure()

