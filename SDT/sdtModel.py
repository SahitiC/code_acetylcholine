import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import time

#%%
#function trial inputs p, pRat, meanS, meanN, delta, varS, varN, prevState,r, cs, cm
#function returns outcome of new trial based on input(decision of animal
#on its own state, generation of signal and response)

def trial(p, pRat, meanS, meanN, delta, varS, varN, prevState, r, cs, cm):
    
    #calculate thresholds
    threshold = (1-pRat)/pRat #beta: threshold that ratios of prob of S to 
                              #N must cross to choose signal
    criterion = np.log(threshold)/(meanS - meanN) #the signal value X above which 
                                               #ratios of prob cross the threshold
    #initialize 
    newState = 2 #new state decided by animal, 0 for U and 1 for P
    trialType = 2 #type of new trial, 0 for N and 1 for S
    x = np.nan #observation drawn from gaussian of N/S based on trial type and state(U/P)
    response = 2 #response to x based on criterion on new trial
    payOffToP = 0; payOffToU =0
    #print(newState, trialType, x, response, payOffToP, payOffToU)
    
    #decision about changing state (U/P) based on expected payoff in new state
    if prevState == 0:
        #if prev state is U, expected payoff in new state (U or P)
        payOffToU = r*pRat*(1-scipy.stats.norm.cdf(criterion, meanS, np.sqrt(varS)))+r*(1-pRat)*(
        scipy.stats.norm.cdf(criterion, meanN, np.sqrt(varN)))        
        
        payOffToP = -cs + r*pRat*(1-scipy.stats.norm.cdf(criterion, meanS+delta, 
        np.sqrt(varS)))+r*(1-pRat)*(scipy.stats.norm.cdf(criterion, meanN-delta, np.sqrt(varN)))
                  
        if payOffToU > payOffToP:
            newState = 0
        elif payOffToU < payOffToP:
            newState = 1
            
    elif prevState == 1:
        #if prev state is P, expected payoff in new state (U or P)
        payOffToU = r*pRat*(1-scipy.stats.norm.cdf(criterion, meanS, np.sqrt(varS)))+r*(1-pRat)*(
            scipy.stats.norm.cdf(criterion, meanN, np.sqrt(varN)))
        
        payOffToP = -cm + r*pRat*(1-scipy.stats.norm.cdf(criterion, meanS+delta, np.sqrt(varS)))+r*(
            1-pRat)*(scipy.stats.norm.cdf(criterion, meanN-delta, np.sqrt(varN)))
        
        if payOffToU > payOffToP:
            newState = 0
        elif payOffToU < payOffToP:
            newState = 1  
    else:
        #if prev trial neither signal nor no signal, set next state (U/P) randomly
        #with equal probability
        j = np.random.binomial(1, 0.5)
        if j == 1:
            newState = 1
        else:
            newState = 0
    #print(newState, trialType, x, response, payOffToP, payOffToU)
            
    #generate observation from S or N based on prior and animal's new state
    if newState == 0:
        i = np.random.binomial(1, p)
        if i == 1:
            trialType = 1
            x = np.random.normal(meanS, np.sqrt(varS))
        else:
            trialType = 0
            x = np.random.normal(meanN, np.sqrt(varN))        
    elif newState == 1: 
        i = np.random.binomial(1, p)
        if i == 1:
            trialType = 1
            x = np.random.normal(meanS+delta, np.sqrt(varS))
        else:
            trialType = 0
            x = np.random.normal(meanN+delta, np.sqrt(varN))
            
                
    #generate response:
    if x > criterion:
        response = 1
    elif x < criterion:
        response = 0
        
    print(payOffToP, payOffToU)
    return newState, trialType, x, response

#%%
#running a trial

p = 0.5 #prior probability of signal
pRat = 0.5 #animal's estimate of prior
meanS = 1; meanN = -1; #means of gaussians of S and N in unprepared state
delta = 0.5 #amount by which means of S and N move apart in prepared state
varS = varN = 1; #variance of S and N gaussian
prevState = 1 # prev state of animal; unprepared state = 0; prepared = 1
r = 1; #reward on correct response
cs = 0.5; cm = 0.1; #costs of shifting from U to P and P to P 

n,t,x,re = trial(p, pRat, meanS, meanN, delta, varS, varN, prevState, r, cs, cm)                                    

prevState = 0
array = np.full((100,4),np.nan) #array to store newState, trialType, x and response for diff delta
for i in range(100):    
    array[i,:] = trial(p, pRat, meanS, meanN, delta, varS, varN, 
                        prevState, r, cs, cm)
array = np.append(np.full((100,1),prevState), array, axis = 1)
plt.plot(array[:,2], label = 'trial type'); plt.plot(array[:,3], label = 'signal');
plt.plot(array[:,4], label = 'response'); plt.title('prev State=%d, newState=%d'%(prevState,array[0,1]))
plt.legend(); plt.figure()

#%%
#multiple trials

p = 0.5 #prior probability of signal
pRat = 0.3 #animal's estimate of prior
meanS = 1; meanN = -1; #means of gaussians of S and N in unprepared state
varS = varN = 1; #variance of S and N gaussian
r = 1; #reward on correct response
delta = 0.5 #amount by which means of S and N move apart in prepared state
cs = 0.12; cm = 0.1; #costs of shifting from U to P and P to P 

#multiple deltas
delta = np.linspace(0,1,11) #a range of deltas
array1 = np.full((len(delta),4),np.nan) #array to store newState, trialType, x and response for diff delta
array0 = np.full((len(delta),4),np.nan) #array to store newState, trialType, x and response for diff deltas

for i in range(len(delta)):
    prevState = 1 # prev state of animal; unprepared state = 0; prepared = 1
    array1[i,:] = trial(p, pRat, meanS, meanN, delta[i], varS, varN, 
                        prevState, r, cs, cm)
    prevState = 0 # prev state of animal; unprepared state = 0; prepared = 1
    array0[i,:] = trial(p, pRat, meanS, meanN, delta[i], varS, varN, 
                        prevState, r, cs, cm)
array1 = np.append(np.full((len(delta),1),1), array1, axis = 1) #1st column = prev state
array0 = np.append(np.full((len(delta),1),0), array0, axis = 1) #1st column = prev state

y1 = np.where(array1[:,1] == array1[:,0], 'PP', 'PU')
y2 = np.where(array0[:,1] == array0[:,0], 'UU', 'UP')
plt.plot(delta, y1, marker = 'o', label = 'cs=%1.2f, cm=%1.2f' %(cs, cm)); 
plt.plot(delta, y2, marker = 'o', label = 'cs=%1.2f, cm=%1.2f' %(cs, cm));
plt.xlabel('delta'); plt.ylabel('transitions'); plt.legend()


# multiple deltas, cm, cs
delta = np.linspace(0,1,11) #a range of deltas
cs = np.array([0.02, 0.06, 0.1, 0.13])
cm = np.array([0, 0.02, 0.06, 0.1])

for k in range(len(cs)):
    j = 0
    while j<len(cm) and cm[j] < cs[k]:
        
        array1 = np.full((len(delta),4),np.nan) #array to store newState, trialType, x and response for diff deltas
        array0 = np.full((len(delta),4),np.nan) #array to store newState, trialType, x and response for diff deltas
        for i in range(len(delta)):
            prevState = 1 # prev state of animal; unprepared state = 0; prepared = 1
            array1[i,:] = trial(p, pRat, meanS, meanN, delta[i], varS, varN, 
                                prevState, r, cs[k], cm[j])
            prevState = 0 # prev state of animal; unprepared state = 0; prepared = 1
            array0[i,:] = trial(p, pRat, meanS, meanN, delta[i], varS, varN, 
                                prevState, r, cs[k], cm[j])
        array1 = np.append(np.full((len(delta),1),1), array1, axis = 1) #1st column = prev state
        array0 = np.append(np.full((len(delta),1),0), array0, axis = 1) #1st column = prev state
        
        y1 = np.where(array1[:,1] == array1[:,0], 'PP', 'PU')
        y0 = np.where(array0[:,1] == array0[:,0], 'UU', 'UP')
        #plt.plot(delta, y1, marker = 'o', label = 'cs=%1.2f' %(cs[k])); 
        plt.plot(delta, y0, marker = 'o', label = 'cm=%1.2f' %(cm[j]));
        plt.xlabel('delta'); plt.ylabel('transitions'); plt.legend()
        j +=1
        
        
#changing pRat
cm = 0.1; cs = 0.12; 
pRat = np.linspace(0.3, 0.7, 5)

for j in range(len(pRat)):
    array1 = np.full((len(delta),4),np.nan) #array to store newState, trialType, x and response for diff deltas
    array0 = np.full((len(delta),4),np.nan) #array to store newState, trialType, x and response for diff deltas
    for i in range(len(delta)):
        prevState = 1 # prev state of animal; unprepared state = 0; prepared = 1
        array1[i,:] = trial(p, pRat[j], meanS, meanN, delta[i], varS, varN, 
                            prevState, r, cs, cm)
        prevState = 0 # prev state of animal; unprepared state = 0; prepared = 1
        array0[i,:] = trial(p, pRat[j], meanS, meanN, delta[i], varS, varN, 
                            prevState, r, cs, cm)
    array1 = np.append(np.full((len(delta),1),1), array1, axis = 1) #1st column = prev state
    array0 = np.append(np.full((len(delta),1),0), array0, axis = 1) #1st column = prev state
        
    y1 = np.where(array1[:,1] == array1[:,0], 'PP', 'PU')
    y0 = np.where(array0[:,1] == array0[:,0], 'UU', 'UP')
    plt.plot(delta, y1, marker = 'o', label = 'pRat=%1.2f' %pRat[j]); 
    plt.plot(delta, y0, marker = 'o', label = 'pRat=%1.2f' %pRat[j]);
    plt.xlabel('delta'); plt.ylabel('transitions'); plt.legend()

