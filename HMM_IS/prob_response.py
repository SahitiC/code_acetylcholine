import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sns


def hmm(pc, n1, n2, w, dt, rounding):
    """
    Returns the hidden states and corresponding observations generated for one trial run of the HMM, based
    on the specified parameters 
    
    Args:
        pc (scalar): prob of going to cue given that transition out of default has ocurred
        n1 (scalar): confusability associated with the default state
        n2 (scalar): confusability associated with the cue state
        w (scalar): length of the trial window specifying the duration of simulation
        dt (scalar): the value of teh timestep to be used for simulation
        rounding(scalar): number of digits to round to for comparison
        
    Returns:
        (list): list of hidden states generated
        (list): list of the corresponding observations generated
    """

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
                if j == 1:                    #cue is pc, and going back to default is 1-pc
                    x = 'cue'
                else:
                    x = 'default'
        
        #print(r, t, count, ts, sep=' ')
        t = t+dt
        count = count +1
        
            
    return obs, states

def responseProb(obs, dt, n1, n2, pc, scaling, prevInternalState, reward, costM, costS,
                 pRes):
    
    """
    Returns the response and internal state at the end of a trial, and the posteriors inferred based on the observations
    
    Args:
        obs(list): the list of observations based on which inference must take place
        dt (scalar): the value of teh timestep to be used for simulation
        n1 (scalar): confusability associated with the default state
        n2 (scalar): confusability associated with the cue state
        pc (scalar): prob of going to cue given that transition out of default has ocurred
        scaling (scalar): scaling factor to be used to prevent underflows during dynamic programming
        prevInternalStat (scalar): internal state of the animal on the previous trial
        reward(scalar): reward on correct response
        costM(scalar): cost of maintaining the active state
        costS(scalar): cost of switching to the active state
        pRes(scalar): probability of responding to cue/default on the A/D states
        
    Returns:
        (scalar): response of the animal at the end of the trial
        (string): internal state chosen by the animal on the current trial
        (float ndarray): posterior probablities infered by the animal through the trial      
    """    
    #0 is default, 1 is cue
    respond = 2; internalState = np.nan; payofftoA = 0; payofftoD = 0
    p = np.full((len(obs)+1,2), np.nan) #array of posterior prob for default, cue
    fs = np.full((len(obs)+1,2), np.nan) #array of scaled f values for default, cue
    
    transition1 = np.array([[1, 0],[0,1]]) #transition probabilities in general
    e = np.array([[n1,1-n1],[1-n2,n2]]) #emission probabilities
    foreperiodSteps = int((6/dt)+1)
    
    
    fs[0,:] = np.array([1,0])
    p[0,:] = fs[0,:]/np.sum(fs[0,:])
    
    #inference process    
    for i in range(len(obs)):
         if i < foreperiodSteps:
             r = 1/(foreperiodSteps-i)
             #print(r, i, sep= ' ')
             transition2 = np.array([[1-pc*r,pc*r],[0,1]])
             #transition probability in foreperiod, before transition
             fs[i+1, :] =  scaling*e[:,int(obs[i])]*np.matmul(fs[i,:], transition2)
             #calculaitng joint probabilities
         else:
             fs[i+1, :] = scaling*e[:,int(obs[i])]*np.matmul(fs[i,:], transition1)
             #calculaitng joint probabilities
             
         p[i+1, :] = fs[i+1,:]/np.sum(fs[i+1,:]) #posterior probabilites
     
    #response process
    
    #calculating payoffs
    if prevInternalState == 'default' :
        payofftoA = p[len(obs),1]*pRes[1,1]*reward + p[len(obs),0]*pRes[0,1]*reward - costS
        payofftoD = p[len(obs),0]*pRes[0,0]*reward + p[len(obs),1]*pRes[1,0]*reward
    elif prevInternalState == 'active' :
        payofftoA = p[len(obs),1]*pRes[1,1]*reward + p[len(obs),0]*pRes[0,1]*reward - costM
        payofftoD = p[len(obs),0]*pRes[0,0]*reward + p[len(obs),1]*pRes[1,0]*reward
    
    
    #deciding internal state based on payoffs
    if payofftoA > payofftoD :
        internalState = 'active'
        k = np.random.binomial(1,pRes[1,1]) #probabilistic response in A
        if k == 1:
            respond = 1
        elif k == 0:
            respond = 0
            
    elif payofftoA < payofftoD :
        internalState = 'default'
        k = np.random.binomial(1,pRes[0,0]) #probabilistic response in D
        if k == 1:
            respond = 0
        elif k == 0:
            respond = 1
        
   
    return respond, internalState, p

#%%
#defining the parameters to be used in the model

scaling = 1.75 #scaling factor to prevent underflows in forward algorithm
dt = 0.005; rounding = 3 #timestep and rounding 
n1 = 0.75; n2 = 0.7; pc = 0.5 #confusability of default, cue; prob of going to cue after transition
window = 0.1;  #window length for stimulus
#prevResponseState, reward, costM, costS,pRes
prevInternalState = 'default'; reward = 1;
costM = 0.01; costS = 0.02; 
pRes = np.array([[0.9, 0.1], [0.1, 0.9]])

#%%

obs, states= hmm(pc,n1, n2, window, dt, rounding)#generating observations
respond, internalState, p = responseProb(obs, dt, n1, n2, pc, scaling, 
                            prevInternalState, reward, costM, costS, pRes)#inference and respons


timeaxis = np.linspace(0, len(obs)*dt, len(obs))
plt.scatter(timeaxis, states); plt.xlabel('time (in sec)'); plt.ylabel('hidden states')   
plt.figure()
plt.scatter(timeaxis, obs); plt.xlabel('time (in sec)'); plt.ylabel('observations')
plt.figure()
plt.plot(timeaxis, p[1:,0]); plt.xlabel('time (in sec)'); plt.ylabel('posterior probability of default state')
plt.figure()
plt.plot(timeaxis, p[1:,1]); plt.xlabel('time (in sec)'); plt.ylabel('posterior probability of cue state')  

#%%
def run_trials(nTrials,pc,n1, n2, window, dt, rounding, scaling, prevInternalState,
               reward, costM, costS, pRes):
    array = np.full((nTrials,3), np.nan) #array of states, responStates and responses for different trials
    array1 = np.full((nTrials,2), 0.0) #array of pCue and pDefault
    for i in range(nTrials):
        obs=[]; states = []; 
        obs, states= hmm(pc,n1, n2, window, dt, rounding)#generating observations
        respond, internalState, p = responseProb(obs, dt, n1, n2, pc, scaling, 
                                prevInternalState, reward, costM, costS, pRes)#inference and respons
        prevInternalState = internalState #IS at the end of current trial is prevIS for the next
        
        
        #collecting array of states, IS and response for each trial
        if states[len(states)-1] == 'default':
            array[i,0] = 0
        elif states[len(states)-1] == 'cue':
            array[i,0] = 1
            
        if internalState == 'default':
            array[i,1] = 0
        elif internalState == 'active':
            array[i,1] = 1
        
        array[i,2] = respond
        array1[i,1] = p[len(p)-1, 1] #pCue
        array1[i,0] = p[len(p)-1, 0] #pDefault
            
    #counting trial types, cue and default trials and avg posteriors (on certain trial types)
    hit = 0; miss = 0; cr = 0; fa = 0; omit = 0; cueTrials= 0; defaultTrials=0;
    pCueHitAvg = 0; pCueFaAvg=0; pCueOmitAvg=0;
    for k in range(len(array)):
        if array[k, 0] == 1:
            cueTrials = cueTrials+1
            if array[k,2] == 1:
                hit = hit +1
                pCueHitAvg = (pCueHitAvg*(hit-1) + array1[k,1])/(hit)
            elif array[k,2] == 0:
                miss = miss +1
            else:
                omit = omit +1
                pCueOmitAvg = (pCueOmitAvg*(omit-1) + array1[k,1])/(omit)
        elif array[k,0] == 0:
            defaultTrials = defaultTrials+1
            if array[k,2] == 0:
                cr = cr +1
            elif array[k,2] == 1:
                fa = fa + 1
                pCueFaAvg = (pCueFaAvg*(fa-1) + array1[k,1])/(fa)
            else:
                omit = omit+1
                pCueOmitAvg = (pCueOmitAvg*(omit-1) + array1[k,1])/(omit)
                
    return array, array1, hit, miss, cr, fa, omit, cueTrials, defaultTrials, pCueHitAvg, pCueFaAvg

#%%
#running nTrials no. of trials

start = time.perf_counter()

scaling = 1.75 #scaling factor to prevent underflows in forward algorithm
dt = 0.005; rounding = 3 #timestep and rounding 
n1 = 0.7; n2 = 0.7; pc = 0.5 #confusability of default, cue; prob of going to cue after transition
window = 0.1;  #window length for stimulus
nTrials = 1000; #no. of trials to simulate
#prevResponseState, reward, costM, costS,pRes
prevInternalState = 'default'; reward = 1;
costM = 0.05; costS = 0.1; 
pRes = np.array([[0.8, 0.2], [0.2, 0.8]])


a, a1,h,m,c,f,o,cT,dT,pCH,pCF = run_trials(nTrials, pc, n1, n2, window, dt, rounding, scaling, 
                                           prevInternalState, reward, costM, costS, pRes)

print(time.perf_counter()-start)

plt.scatter(np.linspace(1,nTrials,nTrials), a[:,0], label = 'state')
plt.scatter(np.linspace(1,nTrials,nTrials), a[:,1], label = 'internal state')
#plt.scatter(np.linspace(1,100,100), a[:,2],  label = 'response')
plt.plot(np.linspace(1,nTrials,nTrials),a1[:,1], label = 'pCue')
plt.title('window = %1.2f , cM = %1.2f , cS = %1.2f, n1 = %1.2f, n2 = %1.2f' %(window, costM, costS, n1, n1))
plt.legend(); plt.figure()

#%%
#sequential effects
def transitions(a, costM, costS):
    #counting IS, signal and trial type transitions
    """
    Returns number of IS transitions across different trials classified according to trial type or signal transitions.
    
    Args:
        a(ndarray): array of signal/hidden states, internal states and responses on each trial
        costM(scalar): cost associated with maintaining active state
        costS(scalar): cost associated with switching to active state
        
    Returns:
        (ndarray): number of different IS transitions across different signal transitions
        (ndarray): number of different IS transitions across different trial type transitions
    """
    
    numA = sum(a[:,1]); a = a.astype(int); nTrials = len(a[:,0])
    transitions = pd.DataFrame(columns = ['AA', 'AD', 'DA', 'DD', 'SS', 'SN', 'NS', 'NN'])
    transitions['AA'] = np.bitwise_and(a[1:,1], a[:-1, 1]); transitions['DA'] = np.bitwise_and(a[1:,1], 1-a[:-1, 1])
    transitions['AD'] = np.bitwise_and(1-a[1:,1], a[:-1, 1]);transitions['DD'] = np.bitwise_and(1-a[1:,1], 1-a[:-1, 1])
    transitions['SS'] = np.bitwise_and(a[1:,0], a[:-1, 0]); transitions['NS'] = np.bitwise_and(a[1:,0], 1-a[:-1, 0])
    transitions['SN'] = np.bitwise_and(1-a[1:,0], a[:-1, 0]);transitions['NN'] = np.bitwise_and(1-a[1:,0], 1-a[:-1, 0])
    
    transition = np.zeros((nTrials,4)) #H,M,FA,CR
    transition[:,0] = np.bitwise_and(a[:,0], a[:, 2]); transition[:,1] = np.bitwise_and(a[:,0], 1-a[:, 2])
    transition[:,2] = np.bitwise_and(1-a[:,0], a[:, 2]);transition[:,3] = np.bitwise_and(1-a[:,0], 1-a[:, 2])
    
    
    #no. of DA, DD, AD, AA transitions on SS, SN, NS, NN trials
    numSignal = np.zeros((4,4)) #DA, DD, AA, AD numbers for SS, SN, NS, NN
    numSignal[0,:] = [sum(np.bitwise_and(transitions['DA'], transitions['SS'])),
    sum(np.bitwise_and(transitions['DA'], transitions['SN'])), 
    sum(np.bitwise_and(transitions['DA'], transitions['NS'])),
    sum(np.bitwise_and(transitions['DA'], transitions['NN']))]
    numSignal[1,:] = [sum(np.bitwise_and(transitions['DD'], transitions['SS'])),
    sum(np.bitwise_and(transitions['DD'], transitions['SN'])), 
    sum(np.bitwise_and(transitions['DD'], transitions['NS'])),
    sum(np.bitwise_and(transitions['DD'], transitions['NN']))]
    numSignal[2,:] = [sum(np.bitwise_and(transitions['AA'], transitions['SS'])),
    sum(np.bitwise_and(transitions['AA'], transitions['SN'])), 
    sum(np.bitwise_and(transitions['AA'], transitions['NS'])),
    sum(np.bitwise_and(transitions['AA'], transitions['NN']))]
    numSignal[3,:] = [sum(np.bitwise_and(transitions['AD'], transitions['SS'])),
    sum(np.bitwise_and(transitions['AD'], transitions['SN'])), 
    sum(np.bitwise_and(transitions['AD'], transitions['NS'])),
    sum(np.bitwise_and(transitions['AD'], transitions['NN']))]
    
    #no. of DA, DD, AD, AA transitions on incongruent and congruent hits and crs
    numTypes = np.zeros((4,4))
    da = np.array(transitions.loc[:,'DA'])
    transition = transition.astype(int)
    for i in range(4):
        for j in range(4): 
            numTypes[i,j] = sum(np.bitwise_and(np.bitwise_and(transition[1:,j], transition[:-1, i]), da))

    return numSignal, numTypes

#%%
#counting transitions

nTrials = 1000; costM = 0.01; costS = 0.02
a, a1,h,m,c,f,o,cT,dT,pCH,pCF = run_trials(nTrials, pc, n1, n2, window, dt, rounding, scaling, prevInternalState, reward, costM, costS, pRes)
#counting no. of AA, AD, DA, DD transitions
a = a.astype(int)
nAA = sum(np.bitwise_and(a[1:,1], a[:-1, 1])); nDA = sum(np.bitwise_and(a[1:,1], 1-a[:-1, 1]))
nAD = sum(np.bitwise_and(1-a[1:,1], a[:-1, 1])); nDD = sum(np.bitwise_and(1-a[1:,1], 1-a[:-1, 1]))
plt.plot(['AA', 'AD','DA', 'DD'],[nAA, nAD,nDA, nDD], marker = 'o', label = 'cm=%1.2f cs=%1.2f' %(costM, costS))
plt.legend(); plt.xlabel('transitions'); plt.ylabel('number'); plt.figure()
numSignal, numTypes = transitions(a, costM, costS)
plt.plot(['SS', 'SN', 'NS', 'NN'],numSignal[0,:], marker ='o', label = 'DA transitions')
plt.plot(['SS', 'SN', 'NS', 'NN'],numSignal[1,:], marker ='o', label = 'DD transitions')
plt.plot(['SS', 'SN', 'NS', 'NN'],numSignal[2,:], marker ='o', label = 'AA transitions')
plt.plot(['SS', 'SN', 'NS', 'NN'],numSignal[3,:], marker ='o', label = 'AD transitions')
plt.xlabel('transitions in signal'); plt.ylabel('number of transitions')
plt.legend(); plt.title('cm=%1.2f cs=%1.2f' %(costM, costS)); plt.figure()

ax = sns.heatmap(numTypes, xticklabels=['H', 'M', 'FA', 'CR'], yticklabels=['H', 'M', 'FA', 'CR'])
plt.xlabel('current trial'); plt.ylabel('prev trial'); 
plt.title('no. of DA transitions, cm=%1.2f cs=%1.2f' %(costM, costS)); plt.show();   

#%%
scaling = 1.75 #scaling factor to prevent underflows in forward algorithm
dt = 0.005; rounding = 3 #timestep and rounding 
n1 = 0.7; n2 = 0.7; pc = 0.5 #confusability of default, cue; prob of going to cue after transition
window = 0.5;  #window length for stimulus
nTrials = 1000 #no. of trials
#prevResponseState, reward, costM, costS,pRes
prevResponseState = 'default'; reward = 1;
costM = 0.05; costS = 0.1; 
pRes = np.array([[0.9, 0.1], [0.1, 0.9]])

costS = [0.05, 0.4, 0.7]; costM = np.array([[0.01, 0.02, 0.04], 
                                             [0.05, 0.1, 0.2], [0.2,0.3, 0.6]])
trialTypeNos = np.full((len(costS),len(costM[0,:]),5), 0)
trialTypePercent = np.full((len(costS),len(costM[0,:]),3), 0.0)
postAvg =  np.full((len(costS),len(costM[0,:]),2), 0.0)
numberA = np.full((len(costS),len(costM[0,:]),1), 0.0)

fig1, ax1 = plt.subplots(); fig2, ax2 = plt.subplots()

for i in range(len(costS)):
    for j in range(len(costM[0,:])):
        array = np.full((nTrials,3), np.nan) #array of states, responStates and responses for different trials
        array1 = np.full((nTrials,2), 0.0) #array of pCue and pDefault
        a, a1,h,m,c,f,o,cT,dT,pCH,pCF = run_trials(nTrials, pc, n1, n2, window, dt, 
                rounding, scaling, prevResponseState, reward, costM[i,j], costS[i], pRes)
        trialTypeNos[i,j,0] = h; trialTypeNos[i,j,1] = m
        trialTypeNos[i,j,2] = c; trialTypeNos[i,j,3] = f;
        trialTypeNos[i,j,4] = o 
        
        trialTypePercent[i,j,0] = h*100/(cT) #percent hits of signal trials
        trialTypePercent[i,j,1] = f*100/(dT) #percent fa of nonsignal trials
        trialTypePercent[i,j,2] = o*100/(cT+dT) #percent omits of all trials
    
        postAvg[i,j,0] = pCH #avg posterior prob of cue on hit
        postAvg[i,j,1] = pCF #avg posterior prob of cue on hit 
        
        nAA = len(np.intersect1d(np.where(a[:,1] == 1),np.where(a[1:,1] == 1)))
        nAD = len(np.intersect1d(np.where(a[:,1] == 1),np.where(a[1:,1] == 0)))
        nDD = len(np.intersect1d(np.where(a[:,1] == 0),np.where(a[1:,1] == 0)))
        nDA = len(np.intersect1d(np.where(a[:,1] == 0),np.where(a[1:,1] == 1)))
    
        ax1.plot(['AA', 'AD','DA', 'DD'],[nAA, nAD,nDA, nDD], marker = 'o',  
                 label = 'cm=%1.2f cs=%1.2f' %(costM[i,j], costS[i]))
        ax2.plot(['A', 'D'], [np.sum(a[:,1]), nTrials-np.sum(a[:,1])],
                  marker = 'o', label = 'cm=%1.2f cs=%1.2f' %(costM[i,j], costS[i]))
                
        
ax1.legend(loc = 'upper left'); ax2.legend(loc = 'upper left'); plt.figure()
ax1.set(xlabel = 'transitions in IS', ylabel = 'number of transitions')
ax2.set(xlabel = 'IS', ylabel = 'number of IS')


fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for i in range( len(costM[0,:])):
    ax1.plot(costM[i,:],trialTypePercent[i,:,0], label = 'hit,costS=%1.2f' %costS[i],
             marker = 'o')
    ax1.plot(costM[i,:],trialTypePercent[i,:,1], label = 'fa,costS=%1.2f' %costS[i],
             marker = 'o')
    ax2.plot(costM[i,:],postAvg[i,:,0], label = 'pCueHit,costS=%1.2f' %costS[i],
             marker = 'o')
    ax2.plot(costM[i,:],postAvg[i,:,1], label = 'pCueFA,costS=%1.2f' %costS[i],
             marker = 'o')
    
ax1.legend(); ax1.set(xlabel = 'costM', ylabel = '%hits/fas of no/signal trials')
ax2.legend(); ax1.set(xlabel = 'costM', ylabel = 'avg posterior probability of cue')

#%%
scaling = 1.75 #scaling factor to prevent underflows in forward algorithm
dt = 0.005; rounding = 3 #timestep and rounding 
n1 = 0.7; n2 = 0.7; pc = 0.5 #confusability of default, cue; prob of going to cue after transition
window = 0.1;  #window length for stimulus
nTrials = 100 #no. of trials
#prevResponseState, reward, costM, costS,pRes
prevResponseState = 'default'; reward = 1;
costM = 0.1; costS = 0.3; 
pRes = np.array([[0.9, 0.1], [0.1, 0.9]])

n1 = [0.6, 0.7, 0.8, 0.9]
n2 = [0.6, 0.7, 0.8, 0.9]

trialTypeNos = np.full((len(n1),len(n2),5), 0)
trialTypePercent = np.full((len(n1),len(n2),3), 0.0)
postAvg =  np.full((len(n1),len(n2),3), 0.0)

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for i in range(len(n1)):
    for j in range(len(n2)):
        array = np.full((nTrials,3), np.nan) #array of states, responStates and responses for different trials
        array1 = np.full((nTrials,2), 0.0) #array of pCue and pDefault
        a, a1,h,m,c,f,o,cT,dT,pCH,pCF = run_trials(nTrials, pc, n1[i], n2[j], window, dt, 
                rounding, scaling, prevResponseState, reward, costM, costS, pRes)
        trialTypeNos[i,j,0] = h; trialTypeNos[i,j,1] = m
        trialTypeNos[i,j,2] = c; trialTypeNos[i,j,3] = f;
        trialTypeNos[i,j,4] = o 
        
        trialTypePercent[i,j,0] = h*100/(cT) #percent hits of signal trials
        trialTypePercent[i,j,1] = f*100/(dT) #percent fa of nonsignal trials
        trialTypePercent[i,j,2] = o*100/(cT+dT) #percent omits of all trials
    
        postAvg[i,j,0] = pCH
        postAvg[i,j,1] = pCF  
        
        
        nAA = len(np.intersect1d(np.where(a[:,1] == 1),np.where(a[1:,1] == 1)))
        nAD = len(np.intersect1d(np.where(a[:,1] == 1),np.where(a[1:,1] == 0)))
        nDD = len(np.intersect1d(np.where(a[:,1] == 0),np.where(a[1:,1] == 0)))
        nDA = len(np.intersect1d(np.where(a[:,1] == 0),np.where(a[1:,1] == 1)))
    
        ax1.plot(['AA', 'AD','DA', 'DD'],[nAA, nAD,nDA, nDD], marker = 'o',  
                 label = 'n2=%1.2f n1=%1.2f' %(n2[j], n1[i]))
        ax2.plot(['A', 'D'], [np.sum(a[:,1]), nTrials-np.sum(a[:,1])],
                  marker = 'o', label = 'n2=%1.2f n1=%1.2f' %(n2[j], n1[i]))
        
        
ax1.legend(loc = 'upper left'); ax2.legend(loc = 'upper left'); plt.figure()
ax1.set(xlabel = 'transitions in IS', ylabel = 'number of transitions')
ax2.set(xlabel = 'IS', ylabel = 'number of IS')

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for i in range(len(n1)):
    ax1.plot(n2,trialTypePercent[i,:,0], label = 'hit,n1=%1.2f' %n1[i],
             marker = 'o')
    ax1.plot(n2,trialTypePercent[i,:,1], label = 'fa,n1=%1.2f' %n1[i],
             marker = 'o')
    ax2.plot(n2,postAvg[i,:,0], label = 'pCueHit,n1=%1.2f' %n1[i],
             marker = 'o')
    ax2.plot(n2,postAvg[i,:,1], label = 'pCueFA,n1=%1.2f' %n1[i],
             marker = 'o')
    
ax1.legend(); ax1.set(xlabel = 'n2', ylabel = '%hits/fas of no/signal trials')
ax2.legend(); ax1.set(xlabel = 'n2', ylabel = 'avg posterior probability of cue')

