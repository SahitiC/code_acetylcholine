import numpy as np
import matplotlib.pyplot as plt
import time

#%%
#simulate sequence of observations based on the parameters

p = 0.5 # on transition out of start, prob to cue 
n = 0.60 # confusabitlity
# on transition out of start, prob (non)cue to (non)cue is 1
w = 0.1 #length of stimulus window
t = 0 # start time
ts = 0 #start time of stimulus 
dt = 0.01 #(sec) : time step
x = 'start' #start hidden state
obs = [] #observed emmisions
states = [] #hidden states
count = 0 #counter

starttime = time.perf_counter()

while round(ts,2) < w:
    
    states.append(x)
    
     #emissions 0, 1, 2 = nothing, cue,no-cue
    if x == 'cue':
        k = np.random.binomial(1,n)
        if k == 1:
            obs.append(1) 
        else:
            obs.append(2)
        ts = ts+dt
    elif x == 'nocue':
        k = np.random.binomial(1,n)
        if k == 1:
            obs.append(2)
        else:
            obs.append(1)
        ts = ts+dt
    elif x == 'start':
        obs.append(0)
       
      
    if count in [0, 10, 20, 30, 40, 50, 60] and x == 'start':
        r = 10/(70-count) #prob of transition out of start
        #print(r)
        i = np.random.binomial(1, r) #transition out of start if i == 1
        
        if i == 1:
            j = np.random.binomial(1, p) #on transitioning out of start, prob of 
            if j == 1:                      #cue is p 
                x = 'cue'
            else:
                x = 'nocue'
        
    t = t+dt
    count = count +1
    
np.savetxt('obs.txt', obs)

print(time.perf_counter()-starttime)    
            
#%%

timeAxis = np.linspace(0, t, int(t/dt))
plt.scatter(timeAxis, states); plt.xlabel('time'); plt.ylabel('hidden states')
plt.figure()
plt.scatter(timeAxis, obs); plt.xlabel('time'); plt.ylabel('observations')

