import numpy as np
import matplotlib.pyplot as plt
import time

#function hmm takes in p, n and returns observations emmitted by model
def hmm(p,n,w,dt,rounding): #inputs are prob to cue/noncue on leaving start and confusability, 
                            #timestep value, rounding decimal for comparison to end the hmm
    t = 0 # start time
    ts = 0 #start time of stimulus 
    x = 'start' #start hidden state
    obs = [] #observed emmisions
    states = [] #hidden states
    count = 0 #counter
    timepoints = (np.array([.0, .1, .2, .3, .4, .5, .6])*1/dt).astype(int)
    
    while round(ts,rounding) < w:
       
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
            
        if count in timepoints and x == 'start':
            r = int(.1/dt)/(int(.7/dt)-count) #prob of transition out of start
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
    return obs, states[int(t/0.01)-1]

#function response takes in observations of a trial, and returns the final response
#after inference    
def response(obs, threshold,dt,n): #gives response according to a set threshold
    respond = 0
    f = np.full((len(obs)+1,3), np.nan) #array of f values for start, cue, non-cue
    p = np.full((len(obs)+1,3), np.nan) #array of posterior prob for start, cue, non-cue
    transition1 = np.array([[1, 0, 0],[0,1,0],[0,0,1]]) #transition probabilities
    e = np.array([[1,0,0],[0,n,1-n],[0,1-n,n]]) #emission probabilities
    
    #inital f and p values
    f[0,:] = np.array([1,0,0]) 
    p[0,:] = f[0,:]/np.sum(f[0,:])
    
    for i in range(len(obs)):
        if p[i, 1] > threshold:
            respond = 1 #response to cue = 1
            break
        elif p[i, 2] > threshold:
            respond = 2 #response to nocue = 2    
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
        
    rt = (i - startTime)*dt #reaction time is time taken from startTime
                          #to cross a threshold posterior        
    return respond, rt

def response2(obs, criterion, dt, n): #gives response at the end of a stimulus window
    respond = 0
    f = np.full((len(obs)+1,3), np.nan) #array of f values for start, cue, non-cue
    p = np.full((len(obs)+1,3), np.nan) #array of posterior prob for start, cue, non-cue
    transition1 = np.array([[1, 0, 0],[0,1,0],[0,0,1]]) #transition probabilities
    e = np.array([[1,0,0],[0,n,1-n],[0,1-n,n]]) #emission probabilities
    timepoints = (np.array([.0, .1, .2, .3, .4, .5, .6])*1/dt).astype(int)+1
    
    #inital f and p values
    f[0,:] = np.array([1,0,0]) 
    p[0,:] = f[0,:]/np.sum(f[0,:]) 
    
    for i in range(len(obs)):
        if i in timepoints and f[i,0]>0:
            #transition probability at above timesteps
            r = int(.1/dt)/(int(.7/dt)-i)
            transition2 = np.array([[1-r, 0.5*r, 0.5*r],[0,1,0],[0,0,1]])
            f[i+1, :] = e[int(obs[i]),:]*np.matmul(f[i, :], transition2)
        else:
        #calculaitng joint probabilities
            f[i+1, :] = e[int(obs[i]),:]*np.matmul(f[i, :], transition1)
    
    
    #prob of state at i+1 given observations till now
    p[i+1, :] = f[i+1,:]/np.sum(f[i+1,:])
    
    if p[len(obs),1] > criterion: #cue is 1
        respond = 1
    elif p[len(obs), 2] > criterion: #nocue is 2
        respond = 2  
        
    return respond, p[len(obs),1], p[len(obs), 2] #return response, posteriors 
                                                  #of cue and noncue at the end of w

#%%

#doing mulitple  wth response1

p = 0.5;# n = 0.7; thresh = 0.9

confusability = np.linspace(0.6,1.0,1)
threshold = np.linspace(0.6,0.98, 1)
trialTypeNos = np.full((len(confusability),len(threshold),5), 0)
trialTypePercent = np.full((len(confusability),len(threshold),3), 0.0)
rtVal = np.full((len(confusability), len(threshold)), 0.0)
#different confusability, threshold, 5 differnettrial type nos.

starttime = time.perf_counter()

for c in range(len(confusability)):
    for t in range(len(threshold)):
        array = np.full((1000,2), 0) #array of states and responses for different trials
        rtAvg = 0 #rt avg across trials
        for j in range(1000):
            obs = []
            respond = 0; state = np.nan
            obs, state = hmm(p, confusability[c],0.5,0.01,2)
            respond, rt = response(obs, threshold[t], 0.01, confusability[c])
            rtAvg = (rtAvg*(j) + rt)/(j+1)
            if state == 'cue':
                array[j,0] = 1
            elif state == 'nocue':
                array[j,0] = 2
            array[j,1] = respond
            
        rtVal[c,t] = rtAvg
        #counting trial types
        hit = 0; miss = 0; cr = 0; fa = 0; omit = 0; cueTrials= 0; nocueTrials=0
            
        for k in range(len(array)):
            if array[k, 0] == 1:
                cueTrials = cueTrials+1
                if array[k,1] == 1:
                    hit = hit +1
                elif array[k,1] == 2:
                    miss = miss +1
                else:
                    omit = omit +1
            elif array[k,0] == 2:
                nocueTrials = nocueTrials+1
                if array[k,1] == 2:
                    cr = cr +1
                elif array[k,1] == 1:
                    fa = fa + 1
                else:
                    omit = omit+1
        
        trialTypeNos[c,t,0] = hit; trialTypeNos[c,t,1] = miss
        trialTypeNos[c,t,2] = cr; trialTypeNos[c,t,3] = fa;
        trialTypeNos[c,t,4] = omit

        trialTypePercent[c,t,0] = hit*100/(cueTrials) #percent hits of signal trials
        trialTypePercent[c,t,1] = fa*100/(nocueTrials) #percent fa of nonsignal trials
        trialTypePercent[c,t,2] = omit*100/(cueTrials + nocueTrials) #percent omits of all trials

print(time.perf_counter()-starttime)

#%%

#plotting hit and fa% against confusability and threshold

for p in range(10):
    plt.plot(threshold, trialTypePercent[p,:,0], label='n= %1.2f' %confusability[p])
    plt.legend()
plt.xlabel('threshold'); plt.ylabel('% hits of signal trials')
plt.figure()
for p in range(10):
    plt.plot(threshold, trialTypePercent[p,:,1], label='n= %1.2f' %confusability[p])
    plt.legend()
plt.xlabel('threshold'); plt.ylabel('% fa of non-signal trials')
plt.figure() 
for p in range(10):
    plt.plot(confusability, trialTypePercent[:,p,0], label='t= %1.2f' %threshold[p])
    plt.legend()
plt.xlabel('confusability'); plt.ylabel('% hits of signal trials')
plt.figure()
for p in range(10):
    plt.plot(confusability, trialTypePercent[:,p,1], label='t= %1.2f' %threshold[p])
    plt.legend()
plt.xlabel('confusability'); plt.ylabel('% fa of non-signal trials')
plt.figure()
for p in range(10):
    plt.plot(threshold, rtVal[p,:], label = 'n=%1.2f' %confusability[p])
    plt.legend()
plt.xlabel('threshold'); plt.ylabel('Avg (across trials) reaction times')
plt.figure()
for p in range(10):
    plt.plot(confusability, rtVal[:,p], label = 't=%1.2f' %threshold[p])
    plt.legend()
plt.xlabel('confusability'); plt.ylabel('Avg (across trials) reaction times')
plt.figure()
for p in range(10):
    plt.plot(threshold,trialTypePercent[p,:,2], label = 'n=%1.2f' %confusability[p])
    plt.legend()
plt.xlabel('threshold'); plt.ylabel('% omits of total trials')
plt.figure()
for p in range(10):
    plt.plot(confusability, trialTypePercent[:,p,2], label = 't=%1.2f' %threshold[p])
    plt.legend()
plt.xlabel('confusability'); plt.ylabel('% omits of total trials')

#%%

p = 0.5; dt = 0.01; rounding = 2; criterion = 0.5
confusability = np.linspace(0.6,1.0,10)
window = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5])
trialTypeNos = np.full((len(confusability),len(window),5), 0)
trialTypePercent = np.full((len(confusability),len(window),3), 0.0)
postAvg =  np.full((len(confusability),len(window),3), 0.0)
#different confusability, threshold, 5 differnet trial type nos., avg posteriors of cue/ no cue
#at the end of stim window

start = time.perf_counter()

for c in range(len(confusability)):
    for t in range(len(window)):
        array = np.full((1000,2), 0) #array of states and responses for different trials
        array1 = np.full((1000,2), 0.0) #array of pCue and pNocue
        for j in range(1000):
            obs = []
            respond = 0; state = np.nan
            obs, state = hmm(p, confusability[c], window[t],dt, rounding)
            respond, pCue, pNocue = response2(obs,criterion,dt, confusability[c])
            if state == 'cue':
                array[j,0] = 1
            elif state == 'nocue':
                array[j,0] = 2
            array[j,1] = respond
            array1[j,0] = pCue
            array1[j,1] = pNocue

        #counting trial types
        hit = 0; miss = 0; cr = 0; fa = 0; omit = 0; cueTrials = 0; noncueTrials =0 
        pCueHitAvg = 0; pCueFaAvg=0; pCueOmitAvg=0;
        #avg posterior of cue in hit trials, fa trials and omit trials at the end of w
        for k in range(len(array)):
            if array[k, 0] == 1:
                cueTrials = cueTrials+1
                if array[k,1] == 1:
                    hit = hit +1
                    pCueHitAvg = (pCueHitAvg*(hit-1) + array1[k,0])/(hit)
                elif array[k,1] == 2:
                    miss = miss +1
                else:
                    omit = omit +1
                    pCueOmitAvg = (pCueOmitAvg*(omit-1) + array1[k,0])/(omit)
            elif array[k,0] == 2:
                noncueTrials = noncueTrials+1
                if array[k,1] == 2:
                    cr = cr +1
                elif array[k,1] == 1:
                    fa = fa + 1
                    pCueFaAvg = (pCueFaAvg*(fa-1) + array1[k,0])/(fa)
                else:
                    omit = omit+1
                    pCueOmitAvg = (pCueOmitAvg*(omit-1) + array1[k,0])/(omit)

        trialTypeNos[c,t,0] = hit; trialTypeNos[c,t,1] = miss
        trialTypeNos[c,t,2] = cr; trialTypeNos[c,t,3] = fa;
        trialTypeNos[c,t,4] = omit

        trialTypePercent[c,t,0] = hit*100/(cueTrials) #percent hits of signal trials
        trialTypePercent[c,t,1] = fa*100/(noncueTrials) #percent fa of nonsignal trials
        trialTypePercent[c,t,2] = omit*100/(cueTrials+noncueTrials) #percent omits of all trials

        postAvg[c,t,0] = pCueHitAvg
        postAvg[c,t,1] = pCueFaAvg
        postAvg[c,t,2] = pCueOmitAvg
        
print(time.perf_counter()-start)

#%%

for p in range(10):
    plt.plot(window, trialTypePercent[p,:,0], label='n= %1.2f' %confusability[p])
    plt.legend()
plt.xlabel('window'); plt.ylabel('% hits of signal trials')
plt.figure()
for p in range(10):
    plt.plot(window, trialTypePercent[p,:,1], label='n= %1.2f' %confusability[p])
    plt.legend()
plt.xlabel('window'); plt.ylabel('% fa of non-signal trials')
plt.figure() 
for p in range(10):
    plt.plot(confusability, trialTypePercent[:,p,0], label='w= %1.2f' %window[p])
    plt.legend()
plt.xlabel('confusability'); plt.ylabel('% hits of signal trials')
plt.figure()
for p in range(10):
    plt.plot(confusability, trialTypePercent[:,p,1], label='w= %1.2f' %window[p])
    plt.legend() 
plt.xlabel('confusability'); plt.ylabel('% fa of non-signal trials')
plt.figure()
for p in range(10):
    plt.plot(window, trialTypePercent[p,:,2], label='n=%1.2f'%confusability[p])
    plt.legend()
plt.xlabel('window'); plt.ylabel('% omits of total trials')
plt.figure()
for p in range(10):
    plt.plot(confusability, trialTypePercent[:,p,2], label='w=%1.2f'%window[p])
    plt.legend()
plt.xlabel('confusability'); plt.ylabel('% omits of total trials')
plt.figure()
for p in range(10):
    plt.plot(window, postAvg[p,:,0], label='n=%1.2f'%confusability[p])
    plt.legend()
plt.xlabel('window'); plt.ylabel('Avg posterior prob of cue on hit trials at end of window')
plt.figure()
for p in range(10):
    plt.plot(window, postAvg[p,:,1], label='n=%1.2f'%confusability[p])
    plt.legend()
plt.xlabel('window'); plt.ylabel('Avg posterior prob of cue on fa trials at end of window')
plt.figure()
for p in range(10):
    plt.plot(window, postAvg[p,:,2], label='n=%1.2f'%confusability[p])
    plt.legend()
plt.xlabel('window'); plt.ylabel('Avg posterior prob of cue on omit trials at end of window')

#%%

array = np.full((100,2), 0) #array of states and responses for different trials
array1 = np.full((100,2), 0.0)
pc = 0.5; n = 0.6; w = 0.1; dt = 0.01; rounding = 3
criterion = 0.5   

for j in range(100):
    obs = []
    respond = 0; state = np.nan
    obs, state = hmm(pc, n, w, dt, rounding)
    respond, pCue, pNocue = response2(obs, criterion, dt, n)
    if state == 'cue':
        array[j,0] = 1
    elif state == 'nocue':
        array[j,0] = 2
    array[j,1] = respond
    array1[j,0] = pCue
    array1[j,1] = pNocue
    
    #counting trial types
hit = 0; miss = 0; cr = 0; fa = 0; omit = 0; cueTrials = 0; noncueTrials=0
pCueHitAvg = 0; pcueFaAvg=0; pCueOmitAvg=0
    
for k in range(len(array)):
    if array[k, 0] == 1:
        cueTrials = cueTrials+1
        if array[k,1] == 1:
            hit = hit +1
            pCueHitAvg = (pCueHitAvg*(hit-1) + array1[k,0])/(hit)
        elif array[k,1] == 2:
            miss = miss +1
        else:
            omit = omit +1
            print(array1[k,0])
            pCueOmitAvg = (pCueOmitAvg*(omit-1) + array1[k,0])/(omit)
    elif array[k,0] == 2:
        noncueTrials = noncueTrials+1
        if array[k,1] == 2:
            cr = cr +1
        elif array[k,1] == 1:
            fa = fa + 1
            pcueFaAvg = (pcueFaAvg*(fa-1) + array1[k,0])/(fa)
        else:
            omit = omit+1
            print(array1[k,0])
            pCueOmitAvg = (pCueOmitAvg*(omit-1) + array1[k,0])/(omit)
  
 