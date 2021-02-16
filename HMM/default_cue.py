import numpy as np
import matplotlib.pyplot as plt
import time

def hmmDC(pc, n1, n2, w, dt, rounding):

    t = 0 # start time
    ts = 0 #start time of stimulus 
    
    x = 'default' #start hidden state
    obs = [] #observations
    states = [] #hidden states
    count = 0 #counter
    transitionYes = 0 #keeps track of whether transition has happened
    foreperiodSteps = int((6/dt)+1)
    
    
    while round(ts,rounding) < w: #this ends when w is over, rounding to avoid
                                  #floating point no. comaparison errors
        states.append(x)
        
        
        if transitionYes == 1:
            ts = ts+dt
        
        #generating observations, 1 is cue, 0 is default
        if x == 'cue':
            k = np.random.binomial(1,n2)
            if k == 1:
                obs.append(1) 
            else:
                obs.append(0)
                
        if x == 'default':
            k = np.random.binomial(1,n1)
            if k == 1:
                obs.append(0) 
            else:
                obs.append(1)
        
        if count < foreperiodSteps and transitionYes == 0:
            #
            r = 1/(foreperiodSteps-count)
            #print(r)
            i = np.random.binomial(1, r) #transition out of default if i == 1
            if i == 1:
                transitionYes = 1
                #tLeft = round(t,rounding)
                j = np.random.binomial(1, pc) #on transitioning out of default, prob of 
                if j == 1:                      #cue is p, and going back to default is 1-p
                    x = 'cue'
                else:
                    x = 'default'
        
        #print(r, t, count, ts, sep=' ')
        t = t+dt
        count = count +1
        
            
    return obs, states


def responseDC(obs, criterion, dt, n1, n2, pc, scaling):
    #0 is default, 1 is cue
    respond = 2
    p = np.full((len(obs)+1,2), np.nan) #array of posterior prob for default, cue
    fs = np.full((len(obs)+1,2), np.nan) #array of scaled f values for default, cue
    
    transition1 = np.array([[1, 0],[0,1]]) #transition probabilities in general
    e = np.array([[n1,1-n1],[1-n2,n2]]) #emission probabilities
    foreperiodSteps = int((6/dt)+1)
    
    
    fs[0,:] = np.array([1,0])
    p[0,:] = fs[0,:]/np.sum(fs[0,:])
    
    for i in range(len(obs)):
         if i < foreperiodSteps:
             r = 1/(foreperiodSteps-i)
             #print(r, i, sep= ' ')
             transition2 = np.array([[1-pc*r,pc*r],[0,1]])
             #transition probability at above timesteps
             fs[i+1, :] =  scaling*e[:,int(obs[i])]*np.matmul(fs[i,:], transition2)
             #calculaitng joint probabilities
         else:
             fs[i+1, :] = scaling*e[:,int(obs[i])]*np.matmul(fs[i,:], transition1)
             #calculaitng joint probabilities
             
         p[i+1, :] = fs[i+1,:]/np.sum(fs[i+1,:])
     
    if p[len(obs),0] > criterion: #default is 0
        respond = 0
    elif p[len(obs), 1] > criterion: #cue is 1
        respond = 1
        
    return respond, fs, p
                       
    
#%%

# a single run of hmmDC and plots

start = time.perf_counter()

scaling = 1.75 #scaling factor to prevent underflows in forward algorithm
dt = 0.005; rounding = 3 #timestep and rounding 
n1 = 0.7; n2 = 0.9; pc = 0.5 #confusability of default, cue; prob of going to cue after transition
window = 0.5; criterion = 0.5 #window length for stimulus, criterion for response
obs, states= hmmDC(pc,n1, n2, window, dt, rounding)
respond, fs, p = responseDC(obs, criterion, dt, n1, n2, pc, scaling)

print(time.perf_counter()-start)   

#plots
timeaxis = np.linspace(0, len(obs)*dt, len(obs))
plt.scatter(timeaxis, states); plt.xlabel('time (in sec)'); plt.ylabel('hidden states')   
plt.figure()
plt.scatter(timeaxis, obs); plt.xlabel('time (in sec)'); plt.ylabel('observations')
plt.figure()
plt.plot(timeaxis, p[1:,0]); plt.xlabel('time (in sec)'); plt.ylabel('posterior probability of default state')
plt.figure()
plt.plot(timeaxis, p[1:,1]); plt.xlabel('time (in sec)'); plt.ylabel('posterior probability of cue state')   

#%%

dt = 0.005; rounding = 3 #timestep and rounding 
pc = 0.5 #prob of going to cue after transition
criterion = 0.5 #criterion for response

scaling = 1.75 #scaling factor to prevent underflows in forward algorithm
n1 = 0.7 #confusability of default
n2 = 0.6 #confusability of cue
window = 0.025

start = time.perf_counter()

array = np.full((1000,2), np.nan) #array of states and responses for different trials
array1 = np.full((1000,2), 0.0) #array of pCue and pDefault
#arraytLeft = np.full((1000,1), np.nan) #array of time at which state has left start
for i in range(1000):
    obs=[]; states = []; 
    obs, states = hmmDC(pc,n1, n2, window, dt, rounding)
    respond, fs, p = responseDC(obs, criterion, dt, n1, n2, pc, scaling)
    if states[len(states)-1] == 'default':
        array[i,0] = 0
    elif states[len(states)-1] == 'cue':
        array[i,0] = 1
    
    array[i,1] = respond
    array1[i,0] = p[len(p)-1, 1] #pCue
    array1[i,1] = p[len(p)-1, 0] #pDefault
    #arraytLeft[i,0] = tLeft

hit = 0; miss = 0; cr = 0; fa = 0; omit = 0; cueTrials= 0; defaultTrials=0;
pCueHitAvg = 0; pCueFaAvg=0; pCueOmitAvg=0;

for k in range(len(array)):
    if array[k, 0] == 1:
        cueTrials = cueTrials+1
        if array[k,1] == 1:
            hit = hit +1
            pCueHitAvg = (pCueHitAvg*(hit-1) + array1[k,0])/(hit)
        elif array[k,1] == 0:
            miss = miss +1
        else:
            omit = omit +1
            pCueOmitAvg = (pCueOmitAvg*(omit-1) + array1[k,0])/(omit)
    elif array[k,0] == 0:
        defaultTrials = defaultTrials+1
        if array[k,1] == 0:
            cr = cr +1
        elif array[k,1] == 1:
            fa = fa + 1
            pCueFaAvg = (pCueFaAvg*(fa-1) + array1[k,0])/(fa)
        else:
            omit = omit+1
            pCueOmitAvg = (pCueOmitAvg*(omit-1) + array1[k,0])/(omit)
            
print(time.perf_counter()-start)

#%%

n1 = [0.6, 0.7, 0.8, 0.9]
n2 = [0.6, 0.7, 0.8, 0.9]

scaling = 1.75 #scaling factor to prevent underflows in forward algorithm

pc = 0.5; dt = 0.005; rounding = 3

criterion = 0.5; window = 0.1

trialTypeNos = np.full((len(n1),len(n2),5), 0)
trialTypePercent = np.full((len(n1),len(n2),3), 0.0)
postAvg =  np.full((len(n1),len(n2),3), 0.0)
#different confusability, threshold, 5 differnet trial type nos., avg posteriors of cue/ default
#at the end of stim window

start = time.perf_counter()

for a in range(len(n1)):
    for b in range(len(n2)):
        array = np.full((100,2), np.nan) #array of states and responses for different trials
        array1 = np.full((100,2), 0.0) #array of pCue and pDefault
        #arraytLeft = np.full((1000,1), np.nan) #array of time at which state has left start
        for i in range(100):
            obs=[]; states = []; 
            obs, states = hmmDC(pc,n1[a], n2[b], window, dt, rounding)
            respond, fs, p = responseDC(obs, criterion, dt, n1[a], n2[b], pc, scaling)
            if states[len(states)-1] == 'default':
                array[i,0] = 0
            elif states[len(states)-1] == 'cue':
                array[i,0] = 1
            
            array[i,1] = respond
            array1[i,0] = p[len(p)-1, 1] #pCue
            array1[i,1] = p[len(p)-1, 0] #pDefault
            #arraytLeft[i,0] = tLeft
        
        hit = 0; miss = 0; cr = 0; fa = 0; omit = 0; cueTrials= 0; defaultTrials=0
        pCueHitAvg = 0; pCueFaAvg=0; pCueOmitAvg=0;
        
        for k in range(len(array)):
            if array[k, 0] == 1:
                cueTrials = cueTrials+1
                if array[k,1] == 1:
                    hit = hit +1
                    pCueHitAvg = (pCueHitAvg*(hit-1) + array1[k,0])/(hit)
                elif array[k,1] == 0:
                    miss = miss +1
                else:
                    omit = omit +1
                    pCueOmitAvg = (pCueOmitAvg*(omit-1) + array1[k,0])/(omit)
            elif array[k,0] == 0:
                defaultTrials = defaultTrials+1
                if array[k,1] == 0:
                    cr = cr +1
                elif array[k,1] == 1:
                    fa = fa + 1
                    pCueFaAvg = (pCueFaAvg*(fa-1) + array1[k,0])/(fa)
                else:
                    omit = omit+1
                    pCueOmitAvg = (pCueOmitAvg*(omit-1) + array1[k,0])/(omit)
                    
        trialTypeNos[a,b,0] = hit; trialTypeNos[a,b,1] = miss
        trialTypeNos[a,b,2] = cr; trialTypeNos[a,b,3] = fa;
        trialTypeNos[a,b,4] = omit  
        
        trialTypePercent[a,b,0] = hit*100/(cueTrials) #percent hits of signal trials
        trialTypePercent[a,b,1] = fa*100/(defaultTrials) #percent fa of nonsignal trials
        trialTypePercent[a,b,2] = omit*100/(cueTrials+defaultTrials) #percent omits of all trials

        postAvg[a,b,0] = pCueHitAvg
        postAvg[a,b,1] = pCueFaAvg
        postAvg[a,b,2] = pCueOmitAvg                      
       
print(time.perf_counter()-start)

#%%

for p in range(len(n1)):
    plt.plot(n2, trialTypePercent[p,:,0], marker = 'o', label='n1= %1.2f' %n1[p])
    plt.legend()
plt.xlabel('n2'); plt.ylabel('% hits of signal trials')
plt.figure()    

for p in range(len(n1)):
    plt.plot(n2, trialTypePercent[p,:,1], marker = 'o', label='n1= %1.2f' %n1[p])
    plt.legend()
plt.xlabel('n2'); plt.ylabel('% fas of non-signal trials')
plt.figure()

for p in range(len(n1)):
    plt.plot(n2, trialTypePercent[p,:,2], marker = 'o', label='n1= %1.2f' %n1[p])
    plt.legend()
plt.xlabel('n2'); plt.ylabel('% omits of total trials')
plt.figure()

for p in range(len(n2)):
    plt.plot(n1, trialTypePercent[:,p,0], marker = 'o', label='n2= %1.2f' %n2[p])
    plt.legend()
plt.xlabel('n1'); plt.ylabel('% hits of signal trials')
plt.figure()    

for p in range(len(n2)):
    plt.plot(n1, trialTypePercent[:,p,1], marker = 'o', label='n2= %1.2f' %n2[p])
    plt.legend()
plt.xlabel('n1'); plt.ylabel('% fas of non-signal trials')
plt.figure()

for p in range(len(n2)):
    plt.plot(n1, trialTypePercent[:,p,2], marker = 'o', label='n2= %1.2f' %n2[p])
    plt.legend()
plt.xlabel('n1'); plt.ylabel('% omits of total trials')
plt.figure()


for p in range(len(n1)):
    plt.plot(n2, postAvg[p,:,0], marker = 'o', label='n1=%1.2f'%n1[p])
    plt.legend()
plt.xlabel('n2'); plt.ylabel('Avg posterior prob of cue on hit trials at end of window')
plt.figure()

for p in range(len(n1)):
    plt.plot(n2, postAvg[p,:,1], marker = 'o', label='n1=%1.2f'%n1[p])
    plt.legend()
plt.xlabel('n2'); plt.ylabel('Avg posterior prob of cue on fa trials at end of window')
plt.figure()

for p in range(len(n1)):
    plt.plot(n2, postAvg[p,:,2], marker = 'o', label='n1=%1.2f'%n1[p])
    plt.legend()
plt.xlabel('n2'); plt.ylabel('Avg posterior prob of cue on omit trials at end of window')
plt.figure()

for p in range(len(n2)):
    plt.plot(n1, postAvg[:,p,0], marker = 'o', label='n2=%1.2f'%n2[p])
    plt.legend()
plt.xlabel('n1'); plt.ylabel('Avg posterior prob of cue on hit trials at end of window')
plt.figure()

for p in range(len(n2)):
    plt.plot(n1, postAvg[:,p,1], marker = 'o', label='n2=%1.2f'%n2[p])
    plt.legend()
plt.xlabel('n1'); plt.ylabel('Avg posterior prob of cue on fa trials at end of window')
plt.figure()

for p in range(len(n2)):
    plt.plot(n1, postAvg[:,p,2], marker = 'o', label='n2=%1.2f'%n2[p])
    plt.legend()
plt.xlabel('n1'); plt.ylabel('Avg posterior prob of cue on omit trials at end of window')
plt.figure()

