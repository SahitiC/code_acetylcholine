#importing required packages
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm

#%%
def generate_trial(trial_length, p_signal, mu_0, mu_1, mu_2, sigma, q,signal_length_type,signal_length):
    n = trial_length
    trial_signal = np.random.binomial(1,p_signal) #trial is signal/non signal with prob 0.5
    
    if signal_length_type == 0:
        start_signal = np.random.randint(0, n) #index of sample when signal begins (on signal trial)
        end_signal = np.random.geometric(q, size=1) #time points for which signal will stay on
    elif signal_length_type == 1:
        start_signal = np.random.randint(0, n-signal_length+1) #index of sample when signal begins (on signal trial)
        end_signal = signal_length #length of signal is fixed
    
    trial = np.full((int(round(n))+1,1),0);#state values within a trial
    if trial_signal == 1: 
        r = min(end_signal, n-start_signal)
        trial[start_signal+1:start_signal+int(r)+1] = np.full((int(r),1),1)
        trial[start_signal+int(r)+1:n+1] = np.full((n-start_signal-int(r),1),2) 
    
    observation = np.full((int(round(n)),1),np.nan)
    for i in range(len(observation)):
        if trial[i+1] == 0:
            observation[i] = np.random.normal(mu_0,sigma)
        elif trial[i+1] == 1:
            observation[i] = np.random.normal(mu_1,sigma)
        elif trial[i+1] == 2:
            observation[i] = np.random.normal(mu_2,sigma)
            
    return trial, observation

def generate_trialHMM(trial_length, p_signal, mu_0, mu_1, mu_2, sigma, q,
                      signal_length_type, signal_length):
    transition_matrix = np.array([[1,0,0],[0,1-q,q],[0,0,1]])
    emission = np.array([mu_0,mu_1,mu_2])
    n = trial_length
    trial = np.full((int(round(n))+1,1),0);#state values within a trial
    observation = np.full((int(round(n)),1),np.nan)
        
    for i in range(n):
        probStart = 1/((n)-i)
        transition_matrix = np.array([[1-(p_signal*probStart),p_signal*probStart,0],
                                      [0,1-q,q],[0,0,1]])
        

        trial[i+1] = np.random.choice([0,1,2], 
                                          p=transition_matrix[trial[i],:][0])
        
        observation[i] = np.random.normal(emission[trial[i+1]],sigma)
        
    return trial, observation

def inference(observation,trial_length,p_signal, mu_0, mu_1, mu_2, sigma, q):
    n = trial_length
    transition_matrix = np.array([[1,0,0],[0,1-q,q],[0,0,1]])
       
    posterior = np.full((trial_length+1,3),0.0) #posterior for states 0,1,2
    posterior[0,:] = [1.0,0.0,0.0] # posterior at j = 0
    posterior[0,:] = posterior[0,:]/(np.sum(posterior[0,:]))
    
    for j in range(n):
        probStart = 1/((n/p_signal)-j)
        transition_matrix = np.array([[1-(probStart),probStart,0],
                                      [0,1-q,q],[0,0,1]])
        emission_probability = [norm.pdf(observation[j],mu_0,sigma)[0],
        norm.pdf(observation[j],mu_1,sigma)[0], norm.pdf(observation[j],mu_2,sigma)[0]]
        
        posterior[j+1,:] = emission_probability*np.matmul(posterior[j,:],transition_matrix)
        posterior[j+1,:] = posterior[j+1,:]/(np.sum(posterior[j+1,:]))
        
    return posterior

def generate_response(trial,posterior):
    response = 10
    hit = 0; miss = 0; cr = 0; fa = 0
    trial_signal = 0
    if sum(trial) > 0: trial_signal = 1
    inferred_state = np.where(posterior[len(trial)-1,:] == max(posterior[len(trial)-1,:]))[0]
    if inferred_state ==0:
        response = 0;
        if trial_signal==0: cr =cr+1
        elif trial_signal==1: miss =miss+1
    elif inferred_state==1:
        response = 1;
        if trial_signal==0: fa=fa+1
        elif trial_signal==1: hit=hit+1
    elif inferred_state==2:
        response = 1;
        if trial_signal==0: fa=fa+1
        elif trial_signal==1: hit=hit+1   
    
    return inferred_state,response, hit, miss, cr, fa

#%%
#simulating a single trial+inference

func = generate_trial #function to use
#parameters:
trial_length = 50 #trial length
p_signal = 0.5 #prob of signal trial
mu_0 = 0; mu_1 = 1; mu_2 = 0 #means of gaussian for observations in states 0,1,2
sigma = 1 #standard deviation of Gaussian
q = 0.1 #constant probability of leaving
signal_length_type = 0; signal_length = 10

start = time.perf_counter()
trial, observation = func(trial_length, p_signal, mu_0, mu_1, mu_2, sigma, q,
                          signal_length_type,signal_length)
posterior = inference(observation,trial_length,p_signal, mu_0, mu_1, 
                      mu_2, sigma, q)
inferred_state,response,hit,miss,cr,fa = generate_response(trial,posterior)
print(time.perf_counter()-start)  

#%%
#plotting the associated signal and inferred posterior
t = np.arange(0,len(observation),1)
plt.plot(t,trial[1:], label='underlying signal')
plt.yticks([0,1,2],labels = ['pre-signal', 'signal', 'post-signal'])
plt.plot(t,observation, label='observations', 
         marker = 'o', linestyle = 'dashed')
plt.legend(); plt.xlabel('timepoint'); 
plt.title('p_signal=%1.2f, q=%1.2f, N=%d'%(p_signal,q,trial_length))
plt.figure()
plt.plot(posterior[1:,0],label='P(X=0|Y{1:t})')
plt.plot(posterior[1:,1],label='P(X=1|Y{1:t})')
plt.plot(posterior[1:,2],label='P(X=2|Y{1:t})')
plt.legend(); plt.ylabel('posterior'); plt.xlabel('time')
plt.title('p_signal=%1.2f, q=%1.2f, N=%d'%(p_signal,q,trial_length))
plt.figure()

#%%
#generating trial(s) - using generate_trial or generate_trialHMM

func = generate_trialHMM #function to use to generate trial
#parameters:
trial_length = 1 #trial length
p_signal = 0.5 #prob of signal trial
mu_0 = 0; mu_1 = 1; mu_2 = 0 #means of gaussian for observations in states 0,1,2
sigma = 1 #standard deviation of Gaussian
q = 0.01 #constant probability of leaving
nTrials = 2000
signal_length_type = 0; signal_length = 10

trial_lengthArr = [1,5,10,25,50,75,100,150]
qArr = [0.0] #0.01,0.2,0.5,0.7
trialTypeRates = np.full((len(qArr),len(trial_lengthArr),4),np.nan)

start = time.perf_counter()

for t in range(len(trial_lengthArr)):
    trial_length= trial_lengthArr[t]
    s=0
    for s in range(len(qArr)):
        q = qArr[s]
        trial_type = np.full((nTrials,3),0) #signal trial or not, start point of signal, signal len
        hit = 0; miss = 0; cr = 0; fa = 0
        #trials:
        for k in range(nTrials):        

            trial, observation = func(trial_length, p_signal, mu_0, mu_1, mu_2,
                                      sigma, q, signal_length_type, signal_length)

            trial, observation = func(trial_length, p_signal, mu_0, mu_1, 
                        mu_2, sigma, q,signal_length_type,signal_length)

            posterior = inference(observation,trial_length,p_signal, mu_0, mu_1, 
                              mu_2, sigma, q)
            inferred_state,response,hit0,miss0,cr0,fa0 = generate_response(trial,posterior)
            
            hit = hit+hit0; miss = miss+miss0; cr = cr+cr0; fa = fa+fa0
            
        trialTypeRates[s,t,0] = hit; trialTypeRates[s,t,1] = miss;
        trialTypeRates[s,t,2] = cr; trialTypeRates[s,t,3] = fa;

print(time.perf_counter()-start) 

#%%
for l in range(len(qArr)):

    a = trialTypeRates[l,:,0]/(trialTypeRates[l,:,0]+trialTypeRates[l,:,1])
    plt.plot(trial_lengthArr,a, marker = 'o', label = 'q=%1.3f'%qArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('trial length')
plt.ylabel('hit rates'); plt.title('using V_j');
plt.figure()

for l in range(len(qArr)):

    a = trialTypeRates[l,:,3]/(trialTypeRates[l,:,2]+trialTypeRates[l,:,3])
    plt.plot(trial_lengthArr,a, marker = 'o', label = 'signal=%1.3f'%qArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('trial length')
plt.ylabel('fa rates'); plt.title('using V_j'); plt.figure()

for l in range(len(trial_lengthArr)):

    a = trialTypeRates[:,l,0]/(trialTypeRates[:,l,0]+trialTypeRates[:,l,1])
    plt.plot(qArr,a, marker = 'o', label = 'trial_length=%d'%trial_lengthArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('signal length')
plt.ylabel('hit rates'); plt.title('using V_j'); plt.figure()

for l in range(len(trial_lengthArr)):

    a = trialTypeRates[:,l,3]/(trialTypeRates[:,l,2]+trialTypeRates[:,l,3])
    plt.plot(qArr,a, marker = 'o', label = 'trial_length=%d'%trial_lengthArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('signal length')
plt.ylabel('fa rates'); plt.title('using V_j'); plt.figure()

       
#%%
#generating trial(s) - using generate_trial or generate_trialHMM

func = generate_trial #function to use to generate trial
#parameters:
trial_length = 50 #trial length
p_signal = 0.5 #prob of signal trial
mu_0 = 0; mu_1 = 1; mu_2 = 0 #means of gaussian for observations in states 0,1,2
sigma = 1 #standard deviation of Gaussian
q = 0.01 #constant probability of leaving
nTrials = 2000
signal_length_type = 0; signal_length = 10

trial_lengthArr = [1,5,10,25,50,75,100,150]
signal_lengthArr = [1,3,5,10,15,25,35,50,75,100,150,200]
trialTypeRates = np.full((len(signal_lengthArr),len(trial_lengthArr),4),np.nan)

start = time.perf_counter()

for t in range(len(trial_lengthArr)):
    trial_length= trial_lengthArr[t]
    s=0
    while signal_lengthArr[s] <= trial_length:
        signal_length = signal_lengthArr[s]
        trial_type = np.full((nTrials,3),0) #signal trial or not, start point of signal, signal len
        hit = 0; miss = 0; cr = 0; fa = 0
        #trials:
        for k in range(nTrials):        

            trial, observation = func(trial_length, p_signal, mu_0, mu_1, mu_2,
                                      sigma, q, signal_length_type, signal_length)

            trial, observation = func(trial_length, p_signal, mu_0, mu_1, 
                        mu_2, sigma, q,signal_length_type,signal_length)

            posterior = inference(observation,trial_length,p_signal, mu_0, mu_1, 
                              mu_2, sigma, q)
            inferred_state,response,hit0,miss0,cr0,fa0 = generate_response(trial,posterior)
            
            hit = hit+hit0; miss = miss+miss0; cr = cr+cr0; fa = fa+fa0
            
        trialTypeRates[s,t,0] = hit; trialTypeRates[s,t,1] = miss;
        trialTypeRates[s,t,2] = cr; trialTypeRates[s,t,3] = fa;
        s = s+1

print(time.perf_counter()-start) 

#%%
for l in range(len(signal_lengthArr)):

    a = trialTypeRates[l,:,0]/(trialTypeRates[l,:,0]+trialTypeRates[l,:,1])
    plt.plot(trial_lengthArr,a, marker = 'o', label = 'signal=%1.3f'%signal_lengthArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('trial length')
plt.ylabel('hit rates'); plt.title('using V_j'); plt.figure()

for l in range(len(signal_lengthArr)):

    a = trialTypeRates[l,:,3]/(trialTypeRates[l,:,2]+trialTypeRates[l,:,3])
    plt.plot(trial_lengthArr,a, marker = 'o', label = 'signal=%1.3f'%signal_lengthArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('trial length')
plt.ylabel('fa rates'); plt.title('using V_j'); plt.figure()

for l in range(len(trial_lengthArr)):

    a = trialTypeRates[:,l,0]/(trialTypeRates[:,l,0]+trialTypeRates[:,l,1])
    plt.plot(signal_lengthArr,a, marker = 'o', label = 'trial_length=%d'%trial_lengthArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('signal length')
plt.ylabel('hit rates'); plt.title('using V_j'); plt.figure()

for l in range(len(trial_lengthArr)):

    a = trialTypeRates[:,l,3]/(trialTypeRates[:,l,2]+trialTypeRates[:,l,3])
    plt.plot(signal_lengthArr,a, marker = 'o', label = 'trial_length=%d'%trial_lengthArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('signal length')
plt.ylabel('fa rates'); plt.title('using V_j'); plt.figure()

#%%
#HMM with discrete & finite observations

def generate_trialDiscrete(trial_length, p_signal, eta0, eta1,eta2, q,signal_length_type,signal_length):
    n = trial_length
    trial_signal = np.random.binomial(1,p_signal) #trial is signal/non signal with prob 0.5
    
    if signal_length_type == 0:
        start_signal = np.random.randint(0, n) #index of sample when signal begins (on signal trial)
        end_signal = np.random.geometric(q, size=1) #time points for which signal will stay on
    elif signal_length_type == 1:
        start_signal = np.random.randint(0, n-signal_length+1) #index of sample when signal begins (on signal trial)
        end_signal = signal_length #length of signal is fixed
    
    trial = np.full((int(round(n))+1,1),0);#state values within a trial
    if trial_signal == 1: 
        r = min(end_signal, n-start_signal)
        trial[start_signal+1:start_signal+int(r)+1] = np.full((int(r),1),1)
        trial[start_signal+int(r)+1:n+1] = np.full((n-start_signal-int(r),1),2) 
    
    observation = np.full((int(round(n)),1),0)
    for i in range(len(observation)):
        if trial[i+1] == 0:
            observation[i] = np.random.binomial(1,1-eta0)
        elif trial[i+1] == 1:
            observation[i] = np.random.binomial(1,eta1)
        elif trial[i+1] == 2:
            observation[i] = np.random.binomial(1,1-eta2)
            
    return trial, observation

def generate_trialHMMDiscrete(trial_length, p_signal, eta0, eta1,eta2, q,
                      signal_length_type, signal_length):
    transition_matrix = np.array([[1,0,0],[0,1-q,q],[0,0,1]])
    emission = np.array([1-eta0,eta1,1-eta2])
    n = trial_length
    trial = np.full((int(round(n))+1,1),0);#state values within a trial
    observation = np.full((int(round(n)),1),0)
        
    for i in range(n):
        probStart = 1/((n/p_signal)-i)
        transition_matrix = np.array([[1-(probStart),probStart,0],
                                      [0,1-q,q],[0,0,1]])
        
        trial[i+1] = np.random.choice([0,1,2], 
                     p=transition_matrix[trial[i],:][0])
        
        observation[i] = np.random.binomial(1,emission[trial[i+1]])
        
    return trial, observation


def inferenceDiscrete(observation,trial_length,p_signal, eta0,eta1,eta2, q):
    n = trial_length
    transition_matrix = np.array([[1,0,0],[0,1-q,q],[0,0,1]])
    emission_matrix = np.array([[eta0,1-eta1,eta2],[1-eta0,eta1,1-eta2]])
        
    posterior = np.full((trial_length+1,3),0.0) #posterior for states 0,1,2
    posterior[0,:] = [1.0,0.0,0.0] # posterior at j = 0
    posterior[0,:] = posterior[0,:]/(np.sum(posterior[0,:]))
    
    for j in range(n):
        probStart = 1/((n/p_signal)-j)
        transition_matrix = np.array([[1-(probStart),probStart,0],
                                      [0,1-q,q],[0,0,1]])
        emission_probability = emission_matrix[observation[j],:]
        
        posterior[j+1,:] = emission_probability*np.matmul(posterior[j,:],transition_matrix)
        posterior[j+1,:] = posterior[j+1,:]/(np.sum(posterior[j+1,:]))
        
    return posterior

def generate_responseDiscrete(trial,posterior):
    response = 10
    hit = 0; miss = 0; cr = 0; fa = 0
    trial_signal = 0
    if sum(trial) > 0: trial_signal = 1
    inferred_state = np.where(posterior[len(trial)-1,:] == max(posterior[len(trial)-1,:]))[0]
    if inferred_state ==0:
        response = 0;
        if trial_signal==0: cr =cr+1
        elif trial_signal==1: miss =miss+1
    elif inferred_state==1:
        response = 1;
        if trial_signal==0: fa=fa+1
        elif trial_signal==1: hit=hit+1
    elif inferred_state==2:
        response = 1;
        if trial_signal==0: fa=fa+1
        elif trial_signal==1: hit=hit+1   
    
    return inferred_state,response, hit, miss, cr, fa

#%%
#simulating a single trial+inference

func = generate_trialHMMDiscrete #function to use
#parameters:
trial_length = 50 #trial length
p_signal = 0.5 #prob of signal trial
eta= 0.7
eta_0 = eta; eta_1 = eta; eta_2 = eta #confusabilities for the 3 states
q = 0.1 #constant probability of leaving
signal_length_type = 0; signal_length = 10

start = time.perf_counter()
trial, observation = func(trial_length, p_signal, eta_0, eta_1, eta_2, q,
                          signal_length_type,signal_length)
posterior = inferenceDiscrete(observation,trial_length,p_signal, eta_0,
                              eta_1, eta_2, q)
inferred_state,response,hit,miss,cr,fa = generate_responseDiscrete(trial,posterior)
print(time.perf_counter()-start)  

#%%
fig,ax = plt.subplots(1,1)
t = np.arange(1,trial_length+1,1)
ax.plot(t,trial[1:], label='underlying signal')
ax.set_yticks([0,1,2])
ax.set_yticklabels(['pre-signal', 'signal', 'post-signal'])
ax.scatter(t,observation, label='observations',color ='green')
ax.legend();
ax.set_title('q=%1.2f,eta=%1.1f,N=%d'%(q,eta,trial_length))

plt.figure()
plt.plot(posterior[1:,0],label='P(X=0|Y{1:t})')
plt.plot(posterior[1:,1],label='P(X=1|Y{1:t})')
plt.plot(posterior[1:,2],label='P(X=2|Y{1:t})')
plt.legend();
plt.ylabel('posterior'); plt.xlabel('time')
plt.title('q=%1.2f,eta=%1.1f,N=%d'%(q,eta,trial_length))
plt.figure()

#%%
#generating trial(s) - using generate_trial or generate_trialHMM

func = generate_trialHMMDiscrete #function to use to generate trial
#parameters:
trial_length = 50 #trial length
p_signal = 0.5 #prob of signal trial
eta_0 = 0.7; eta_1 = 0.7; eta_2 = 0.7 #eta
q = 0.00 #constant probability of leaving
nTrials = 2000
signal_length_type = 0; signal_length = 10

trial_lengthArr = [1,5,10,25,50,75,100,150]
qArr = [0.01,0.2,0.5,0.7] #0.01,0.2,0.5,0.7
trialTypeRates = np.full((len(qArr),len(trial_lengthArr),4),np.nan)

start = time.perf_counter()

for t in range(len(trial_lengthArr)):
    trial_length= trial_lengthArr[t]
    s=0
    for s in range(len(qArr)):
        q = qArr[s]
        trial_type = np.full((nTrials,3),0) #signal trial or not, start point of signal, signal len
        hit = 0; miss = 0; cr = 0; fa = 0
        #trials:
        for k in range(nTrials):        

            trial, observation = func(trial_length, p_signal, eta_0, eta_1, 
                                eta_2, q, signal_length_type, signal_length)


            posterior = inferenceDiscrete(observation,trial_length,p_signal, eta_0, 
                                  eta_1, eta_2, q)
            
            inferred_state,response,hit0,miss0,cr0,fa0 = generate_responseDiscrete(
                trial,posterior)
            
            hit = hit+hit0; miss = miss+miss0; cr = cr+cr0; fa = fa+fa0
            
        trialTypeRates[s,t,0] = hit; trialTypeRates[s,t,1] = miss;
        trialTypeRates[s,t,2] = cr; trialTypeRates[s,t,3] = fa;

print(time.perf_counter()-start) 

#%%
for l in range(len(qArr)):

    a = trialTypeRates[l,:,0]/(trialTypeRates[l,:,0]+trialTypeRates[l,:,1])
    plt.plot(trial_lengthArr,a, marker = 'o', label = 'q=%1.3f'%qArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('trial length')
plt.ylabel('hit rates'); 
plt.title('eta0=%1.1f,eta1=%1.1f,eta2=%1.1f'%(eta_0,eta_1,eta_2)); 
plt.figure()

for l in range(len(qArr)):

    a = trialTypeRates[l,:,3]/(trialTypeRates[l,:,2]+trialTypeRates[l,:,3])
    plt.plot(trial_lengthArr,a, marker = 'o', label = 'q=%1.3f'%qArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('trial length')
plt.ylabel('fa rates'); 
plt.title('eta0=%1.1f,eta1=%1.1f,eta2=%1.1f'%(eta_0,eta_1,eta_2));
plt.figure()

for l in range(len(trial_lengthArr)):

    a = trialTypeRates[:,l,0]/(trialTypeRates[:,l,0]+trialTypeRates[:,l,1])
    plt.plot(qArr,a, marker = 'o', label = 'trial_length=%d'%trial_lengthArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('signal length')
plt.ylabel('hit rates');
plt.title('eta0=%1.1f,eta1=%1.1f,eta2=%1.1f'%(eta_0,eta_1,eta_2)); 
plt.figure()

for l in range(len(trial_lengthArr)):

    a = trialTypeRates[:,l,3]/(trialTypeRates[:,l,2]+trialTypeRates[:,l,3])
    plt.plot(qArr,a, marker = 'o', label = 'trial_length=%d'%trial_lengthArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('signal length')
plt.ylabel('fa rates'); 
plt.title('eta0=%1.1f,eta1=%1.1f,eta2=%1.1f'%(eta_0,eta_1,eta_2));
plt.figure()

#%%
#generating trial(s) - using generate_trial or generate_trialHMM

func = generate_trialDiscrete #function to use to generate trial
#parameters:
trial_length = 50 #trial length
p_signal = 0.5 #prob of signal trial
eta_0 = 0.7; eta_1 = 0.7; eta_2 = 0.7 #means of gaussian for observations in states 0,1,2
sigma = 1 #standard deviation of Gaussian
q = 0.00001 #constant probability of leaving
nTrials = 2000
signal_length_type = 1; signal_length = 10

trial_lengthArr = [1,5,10,25,50,75,100,150]
signal_lengthArr = [1,3,5,10,15,25,35,50,75,100,150,200]
trialTypeRates = np.full((len(signal_lengthArr),len(trial_lengthArr),4),np.nan)

start = time.perf_counter()

for t in range(len(trial_lengthArr)):
    trial_length= trial_lengthArr[t]
    s=0
    while signal_lengthArr[s] <= trial_length:
        signal_length = signal_lengthArr[s]
        trial_type = np.full((nTrials,3),0) #signal trial or not, start point of signal, signal len
        hit = 0; miss = 0; cr = 0; fa = 0
        #trials:
        for k in range(nTrials):        

            trial, observation = func(trial_length, p_signal, eta_0, eta_1, 
                                eta_2, q, signal_length_type, signal_length)


            posterior = inferenceDiscrete(observation,trial_length,p_signal, eta_0, 
                                  eta_1, eta_2, q)
            
            inferred_state,response,hit0,miss0,cr0,fa0 = generate_responseDiscrete(
                trial,posterior)
            
            hit = hit+hit0; miss = miss+miss0; cr = cr+cr0; fa = fa+fa0
            
        trialTypeRates[s,t,0] = hit; trialTypeRates[s,t,1] = miss;
        trialTypeRates[s,t,2] = cr; trialTypeRates[s,t,3] = fa;
            
        s = s+1

print(time.perf_counter()-start) 

#%%
for l in range(len(signal_lengthArr)):

    a = trialTypeRates[l,:,0]/(trialTypeRates[l,:,0]+trialTypeRates[l,:,1])
    plt.plot(trial_lengthArr,a, marker = 'o', label = 'signal=%1.3f'%signal_lengthArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('trial length')
plt.ylabel('hit rates'); 
plt.title('eta0=%1.1f,eta1=%1.1f,eta2=%1.1f'%(eta_0,eta_1,eta_2)); 
plt.figure()

for l in range(len(signal_lengthArr)):

    a = trialTypeRates[l,:,3]/(trialTypeRates[l,:,2]+trialTypeRates[l,:,3])
    plt.plot(trial_lengthArr,a, marker = 'o', label = 'signal=%1.3f'%signal_lengthArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('trial length')
plt.ylabel('fa rates'); 
plt.title('eta0=%1.1f,eta1=%1.1f,eta2=%1.1f'%(eta_0,eta_1,eta_2)); plt.figure()

for l in range(len(trial_lengthArr)):

    a = trialTypeRates[:,l,0]/(trialTypeRates[:,l,0]+trialTypeRates[:,l,1])
    plt.plot(signal_lengthArr,a, marker = 'o', label = 'trial_length=%d'%trial_lengthArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('signal length')
plt.ylabel('hit rates'); 
plt.title('eta0=%1.1f,eta1=%1.1f,eta2=%1.1f'%(eta_0,eta_1,eta_2)); 
plt.figure()

for l in range(len(trial_lengthArr)):

    a = trialTypeRates[:,l,3]/(trialTypeRates[:,l,2]+trialTypeRates[:,l,3])
    plt.plot(signal_lengthArr,a, marker = 'o', label = 'trial_length=%d'%trial_lengthArr[l])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.xlabel('signal length')
plt.ylabel('fa rates'); 
plt.title('eta0=%1.1f,eta1=%1.1f,eta2=%1.1f'%(eta_0,eta_1,eta_2));
plt.figure()


#%%
#old/obsolete code
func = generate_trialDiscrete  
trial_length = 10 #trial length
p_signal = 0.5 #prob of signal trial
eta_0 = 0.7; eta_1 = 0.7; eta_2 = 0.7 #means of gaussian for observations in states 0,1,2
q = 0.01 #constant probability of leaving
nTrials = 3000
signal_length_type = 0; signal_length = 10
hit =0; cr=0; miss=0; fa=0
   
trial_type = np.full((nTrials,3),0)
for k in range(nTrials):
    trial, observation = func(trial_length, p_signal, eta_0, eta_1, 
                    eta_2, q, signal_length_type, signal_length)
    posterior = inferenceDiscrete(observation,trial_length,p_signal, eta_0, 
                          eta_1, eta_2, q)
    
    inferred_state,response,hit0,miss0,cr0,fa0 = generate_responseDiscrete(
    trial,posterior)
    trial_signal = 0
    if sum(trial) > 0: trial_signal =1 
    if trial_signal == 1:
        trial_type[k,0] = 1; 
        trial_type[k,1] = np.intersect1d(np.where(trial[1:] == 1)[0], 
                                     np.where(trial[:-1] == 0)[0]) #start signal
        trial_type[k,2] = len(np.where(trial == 1)[0]) #signal length
    
    hit = hit+hit0; miss = miss+miss0; cr = cr+cr0; fa = fa+fa0

























