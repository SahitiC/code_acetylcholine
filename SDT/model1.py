import numpy as np
import matplotlib.pyplot as plt
import time

#%%
#generating data trial wise
p_signal = 0.5; dt = 0.01
iti_lb = 0; iti_ub = 6; 
n = 0.9; d1 = 0.02; d2 = 0.05; d3 = 0.50; p_ds = 1/3

signal = 0; duration=0; iti=0;
i = np.random.binomial(1,p_signal)
if i == 1: signal=1
elif i == 0: signal=0
iti = np.random.choice(np.arange(iti_lb,iti_ub,dt))
if signal == 0 : dur = 0.0
else:
    j = np.random.choice(3, p = np.array([p_ds,p_ds,p_ds]))
    if j == 0: duration = d1
    elif j == 1: duration = d2
    elif j == 2: duration = d3 

steps = int(iti/dt)+int(duration/dt)+int(2/dt)
state = np.full((steps, 1),0); state[0:int(iti/dt)] = np.full((int(iti/dt), 1),0);
state[int(iti/dt):int(iti/dt)+int(duration/dt)] = np.full((int(duration/dt), 1),1)
state[int(iti/dt)+int(duration/dt):int(iti/dt)+int(duration/dt)+int(2/dt)] =np.full((int(2/dt), 1),0)
obs = np.full((steps, 1),0.0)
for i in range(steps):
    j = np.random.binomial(1,n)
    if j==1: obs[i] = state[i]
    elif j==0: obs[i] = 1-state[i]
    
#%%

