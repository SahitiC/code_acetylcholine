#%%
#importing needed packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import sklearn.linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf

#%%
#convert data into a more usable form

data = pd.read_excel('SAT Choline Behaviorv2.xlsx') #importing data

#separating the data from the 5 rats

ratData1 = pd. DataFrame(data, columns= ['Trial', 'Type', 'Latency(ms)']) 
ratData1 = ratData1.rename(columns={'Latency(ms)': 'Latency'}) 

ratData2 = pd. DataFrame(data, columns= ['Trial.1', 'TYPE', 'Latency'])
ratData2 = ratData2.rename(columns={'Trial.1': 'Trial', 'TYPE': 'Type'})

ratData3 = pd. DataFrame(data, columns= ['Trial.2', 'Type.1', 'Latency.1'])
ratData3 = ratData3.rename(columns={'Trial.2': 'Trial', 'Type.1': 'Type', 'Latency.1':'Latency'})

ratData4 = pd. DataFrame(data, columns= ['Trial.3', 'Type.2', 'latency'])
ratData4 = ratData4.rename(columns={'Trial.3': 'Trial', 'Type.2': 'Type', 'latency': 'Latency'})

ratData5 = pd. DataFrame(data, columns= ['Trial.4', 'Type.3', 'Latency.2'])
ratData5 = ratData5.rename(columns={'Trial.4': 'Trial', 'Type.3': 'Type', 'Latency.2': 'Latency'})

#extracting the data about trials from each rat, convert to np array, string type
ratArr1 = (ratData1.Trial.to_numpy()).astype('U')
ratArr2 = (ratData2.Trial.to_numpy()).astype('U')
ratArr3 = (ratData3.Trial.to_numpy()).astype('U')
ratArr4 = (ratData4.Trial.to_numpy()).astype('U')
ratArr5 = (ratData5.Trial.to_numpy()).astype('U') 

#arrays to hold modified data from each rat
rat1 = np.full((162,6), np.nan, dtype = object) 
rat2 = np.full((162,6), np.nan, dtype = object)
rat3 = np.full((162,6), np.nan, dtype = object)
rat4 = np.full((162,6), np.nan, dtype = object)
rat5 = np.full((162,6), np.nan, dtype = object)

#first column contains trial data for each rat
rat1[:,0] = ratArr1; rat2[:,0] = ratArr2; rat3[:,0] = ratArr3; rat4[:,0] = ratArr4; rat5[:,0] = ratArr5; 

#function to separate data about signal presented, response and trial type from trial data
def change(substr, cue, response, typ):
    rat1[np.char.startswith(ratArr1, substr).nonzero(), 1] = cue #assigning signal presented to next column
    rat1[np.char.startswith(ratArr1, substr).nonzero(), 2] = response #assigning response for cue = 1, and defaullt = 0
    rat1[np.char.startswith(ratArr1, substr).nonzero(), 3] = typ #assigning type of trial to 4th column
    
    rat2[np.char.startswith(ratArr2, substr).nonzero(), 1] = cue
    rat2[np.char.startswith(ratArr2, substr).nonzero(), 2] = response 
    rat2[np.char.startswith(ratArr2, substr).nonzero(), 3] = typ
    
    rat3[np.char.startswith(ratArr3, substr).nonzero(), 1] = cue
    rat3[np.char.startswith(ratArr3, substr).nonzero(), 2] = response 
    rat3[np.char.startswith(ratArr3, substr).nonzero(), 3] = typ

    rat4[np.char.startswith(ratArr4, substr).nonzero(), 1] = cue
    rat4[np.char.startswith(ratArr4, substr).nonzero(), 2] = response 
    rat4[np.char.startswith(ratArr4, substr).nonzero(), 3] = typ

    rat5[np.char.startswith(ratArr5, substr).nonzero(), 1] = cue
    rat5[np.char.startswith(ratArr5, substr).nonzero(), 2] = response 
    rat5[np.char.startswith(ratArr5, substr).nonzero(), 3] = typ    
    return rat1, rat2, rat3, rat4, rat5

#using function change() to take care of various cases
change('M25', 25, 0, 'miss')
change('M50', 50, 0, 'miss')
change('M500', 500, 0, 'miss')
change('LevoffH25', 25, 1, 'hit')
change('LevoffH50', 50, 1, 'hit')
change('LevoffH500', 500, 1, 'hit')
change('H25', 25, 1, 'hit')
change('H50', 50, 1, 'hit')
change('H500', 500, 1, 'hit')
change('h25', 25, 1, 'hit')
change('h50', 50, 1, 'hit')
change('h500', 500, 1, 'hit')
change('CR', 0, 0, 'cr')
change('FA', 0, 1, 'fa')
change('O',3, np.nan, 'omit') #3 because type of signal not known
change('NS',0, np.nan, 'omit')
change('S25', 25, np.nan, 'omit')
change('S50', 50, np.nan, 'omit')
change('S500', 500, np.nan, 'omit')

#extracting the data about response latency from data of eacg rat
ratArr1 = ratData1.Latency.to_numpy()  
ratArr2 = ratData2.Latency.to_numpy() 
ratArr3 = ratData3.Latency.to_numpy() 
ratArr4 = ratData4.Latency.to_numpy() 
ratArr5 = ratData5.Latency.to_numpy() 

#assigning latency to next column
rat1[:,4] = ratArr1; rat2[:,4] = ratArr2; rat3[:,4] = ratArr3; rat4[:,4] = ratArr4; rat5[:,4] = ratArr5;

#extracting the data about type of trials (what type preceded this trial type) 
#from each rat, convert to np array, string type
ratArr1 = (ratData1.Type.to_numpy()).astype('U')
ratArr2 = (ratData2.Type.to_numpy()).astype('U')
ratArr3 = (ratData3.Type.to_numpy()).astype('U')
ratArr4 = (ratData4.Type.to_numpy()).astype('U')
ratArr5 = (ratData5.Type.to_numpy()).astype('U') 

#assigning this to the 6th column
rat1[:,5] = ratArr1; rat2[:,5] = ratArr2; rat3[:,5] = ratArr3; rat4[:,5] = ratArr4; rat5[:,5] = ratArr5;
rat3[:,5] = np.char.lower((rat3[:,5]).astype("U"))

r = [rat1, rat2, rat3, rat4, rat5]

rat = np.vstack((rat1, rat2, rat3,rat4, rat5))
#%%

#some exploratory plotting and stats
missRates = np.full((5,3), 0.0) #percent miss rates for 25, 50, 500 ms signals for each rat
hitRates = np.full((5,3), 0.0) #percent hit rates for 25, 50, 500 ms signals for each rat
crRates = np.full((5,1), 0.0)
faRates = np.full((5,1), 0.0)
omitRates = np.full((5,4), 0.0)
missingTrials = np.full((5,1), 0.0)

missPerc = np.full((5,3), 0.0) #percent miss rates for 25, 50, 500 ms signals for each rat
hitPerc = np.full((5,3), 0.0) #percent hit rates for 25, 50, 500 ms signals for each rat
crPerc = np.full((5,1), 0.0)
faPerc = np.full((5,1), 0.0)
omitPerc = np.full((5,4), 0.0)

#function to count number of trials with a particular signal and type
def countTypes(signal, typ):
    c = np.full((5,1), 0.0)
    c[0,0] = len(np.where(rat1[np.where(rat1[:,1] == signal),3] == typ)[1])
    c[1,0] = len(np.where(rat2[np.where(rat2[:,1] == signal),3] == typ)[1])
    c[2,0] = len(np.where(rat3[np.where(rat3[:,1] == signal),3] == typ)[1])
    c[3,0] = len(np.where(rat4[np.where(rat4[:,1] == signal),3] == typ)[1])
    c[4,0] = len(np.where(rat5[np.where(rat5[:,1] == signal),3] == typ)[1])

    return c[:,0]

missRates[:,0] = countTypes(25, 'miss'); missRates[:,1] = countTypes(50, 'miss'); 
missRates[:,2] = countTypes(500, 'miss')
hitRates[:,0] = countTypes(25, 'hit'); hitRates[:,1] = countTypes(50, 'hit'); 
hitRates[:,2] = countTypes(500, 'hit') 
crRates[:,0] = countTypes(0, 'cr'); faRates[:,0] = countTypes(0, 'fa')
omitRates[:,0] = countTypes(25, 'omit'); omitRates[:,1] = countTypes(50, 'omit'); 
omitRates[:, 2] = countTypes(500, 'omit'); 
omitRates[:,3] = countTypes(0, 'omit'); missingTrials[:,0] = countTypes(3, 'omit') 

missPerc[:,0]=(missRates[:,0]*100)/(missRates[:,0] + hitRates[:,0]); hitPerc[:,0]=100-missPerc[:,0]
missPerc[:,1]=(missRates[:,1]*100)/(missRates[:,1] + hitRates[:,1]); hitPerc[:,1]=100-missPerc[:,1]
missPerc[:,2]=(missRates[:,2]*100)/(missRates[:,2] + hitRates[:,2]); hitPerc[:,2]=100-missPerc[:,2]
crPerc[:,0]=(crRates[:,0]*100)/(crRates[:,0] + faRates[:,0]); faPerc[:,0] = 100-crPerc[:,0]
omitPerc = (np.sum(omitRates,1)*100)/162

mean = [np.mean(hitPerc[:,0]), np.mean(hitPerc[:,1]), np.mean(hitPerc[:,2]), np.mean(crPerc[:,0])]
err = [np.std(hitPerc[:,0]), np.std(hitPerc[:,1]), np.std(hitPerc[:,2]), np.std(crPerc[:,0])]
plt.bar(['25', '50', '500', 'default'], mean, yerr = err)
plt.xlabel('signal duration in ms'); plt.ylabel('mean percent hit/cr rates (of (n)signal trials) over 5 rats')
plt.figure()

#print('Avg cr rate (of nonginal trials) over all rats=%f w std = %1.2f' % (np.mean(crRates[:,0]), np.std(crRates[:,0])))
print('Avg omit rate over all rats=%f w std = %f' %(np.mean(omitRates[:,]), np.std(omitRates[:,])))

# 1-way anova with signal duration as ind variable (and avgs for each rat as sample points)
print('one-way anova across signal durations with avg hit/cr rates from each rat as sample points: (cr excluded/ included)')
print(scipy.stats.f_oneway(hitPerc[:,0], hitPerc[:,1], hitPerc[:,2])) #only hit rates
print(scipy.stats.f_oneway(hitPerc[:,0], hitPerc[:,1], hitPerc[:,2], crPerc[:,0]))#cr rates included

mean1 = [np.mean(hitPerc[0,:]), np.mean(hitPerc[1,:]), np.mean(hitPerc[2,:]), np.mean(hitPerc[3,:]),
         np.mean(hitPerc[4,:])]
err1 =  [np.std(hitPerc[0,:]), np.std(hitPerc[1,:]), np.std(hitPerc[2,:]), np.std(hitPerc[3,:]),
         np.std(hitPerc[4,:])]
plt.bar(['rat1', 'rat2', 'rat3', 'rat4', 'rat5'], mean1, yerr = err1)
plt.xlabel('rat no.'); plt.ylabel('mean percent hit rates (of signal trials) over all signals')
plt.figure()

#anova for ht rates(avg over signals) across rats
print('one-way anova across rats with avg hit rates over signall durations as sample points')
print(scipy.stats.f_oneway(hitRates[0,:], hitRates[1,:], hitRates[2,:], hitRates[3,:],hitRates[4,:]))

mean2 = [np.mean(crRates[0,:]), np.mean(crRates[1,:]), np.mean(crRates[2,:]), np.mean(crRates[3,:]),
         np.mean(crRates[4,:])]
err2 =  [np.std(crRates[0,:]), np.std(crRates[1,:]), np.std(crRates[2,:]), np.std(crRates[3,:]),
         np.std(crRates[4,:])]
plt.bar(['rat1', 'rat2', 'rat3', 'rat4', 'rat5'], mean2, yerr = err2)
plt.xlabel('rat no.'); plt.ylabel('mean percent cr rates (of nonsignal trials) over all signals')
plt.figure()

#%%
#accuracy vs time
accuracy = np.full((5,3),0.0)
for i in range(len(r)):
    accuracy[i,0] = (len(np.where(r[i][:54,3] == 'hit')[0])+len(np.where(r[i][:54,3] == 'cr')[0]))/(
        len(np.where(r[i][:54,3] == 'hit')[0])+len(np.where(r[i][:54,3] == 'cr')[0])+len(np.where(r[i][:54,3] == 'fa')[0])+len(np.where(r[i][:54,3] == 'miss')[0]))
    accuracy[i,1] = (len(np.where(r[i][54:108,3] == 'hit')[0])+len(np.where(r[i][54:108,3] == 'cr')[0]))/(
        len(np.where(r[i][54:108,3] == 'hit')[0])+len(np.where(r[i][54:108,3] == 'cr')[0])+len(np.where(r[i][54:108,3] == 'fa')[0])+len(np.where(r[i][54:108,3] == 'miss')[0]))
    accuracy[i,2] = (len(np.where(r[i][108:162,3] == 'hit')[0])+len(np.where(r[i][108:162,3] == 'cr')[0]))/(
        len(np.where(r[i][108:162,3] == 'hit')[0])+len(np.where(r[i][108:162,3] == 'cr')[0])+len(np.where(r[i][108:162,3] == 'fa')[0])+len(np.where(r[i][108:162,3] == 'miss')[0]))
avg = [np.mean(accuracy[:,0]), np.mean(accuracy[:,1]), np.mean(accuracy[:,2])]
std = [np.std(accuracy[:,0]), np.std(accuracy[:,1]), np.std(accuracy[:,2])]

#VI vs time
hit = np.full((5,3),0.0)
fa = np.full((5,3),0.0)
for i in range(len(r)):
    hit[i,0] = (len(np.where(r[i][:54,3] == 'hit')[0]))/(
        len(np.where(r[i][:54,3] == 'hit')[0])+len(np.where(r[i][:54,3] == 'miss')[0]))
    hit[i,1] = (len(np.where(r[i][54:108,3] == 'hit')[0]))/(
        len(np.where(r[i][54:108,3] == 'hit')[0])+len(np.where(r[i][54:108,3] == 'miss')[0]))
    hit[i,2] = (len(np.where(r[i][108:162,3] == 'hit')[0]))/(
        len(np.where(r[i][108:162,3] == 'hit')[0])+len(np.where(r[i][108:162,3] == 'miss')[0]))
    fa[i,0] = (len(np.where(r[i][:54,3] == 'fa')[0]))/(
        len(np.where(r[i][:54,3] == 'cr')[0])+len(np.where(r[i][:54,3] == 'fa')[0]))
    fa[i,1] = (len(np.where(r[i][54:108,3] == 'fa')[0]))/(
        len(np.where(r[i][54:108,3] == 'cr')[0])+len(np.where(r[i][54:108,3] == 'fa')[0]))
    fa[i,2] = (len(np.where(r[i][108:162,3] == 'fa')[0]))/(
        len(np.where(r[i][108:162,3] == 'cr')[0])+len(np.where(r[i][108:162,3] == 'fa')[0]))

VI = (hit-fa)/(2*(hit+fa)-(hit+fa)*(hit+fa))

avgH = [np.mean(hit[:,0]), np.mean(hit[:,1]), np.mean(hit[:,2])]
stdH = [np.std(hit[:,0]), np.std(hit[:,1]), np.std(hit[:,2])]
avgF = [np.mean(fa[:,0]), np.mean(fa[:,1]), np.mean(fa[:,2])]
stdF = [np.std(fa[:,0]), np.std(fa[:,1]), np.std(fa[:,2])]
avgV = [np.mean(VI[:,0]), np.mean(VI[:,1]), np.mean(VI[:,2])]
stdV = [np.std(VI[:,0]), np.std(VI[:,1]), np.std(VI[:,2])]

hit25 = np.full((5,3),0.0); hit50 = np.full((5,3),0.0); hit500 = np.full((5,3),0.0)
for i in range(len(r)):
    hit25[i,0] = sum(np.bitwise_and(r[i][:54,3] == 'hit',r[i][:54,1] == 25))/(
        sum(np.bitwise_and(r[i][:54,3] == 'hit',r[i][:54,1] == 25))+sum(np.bitwise_and(r[i][:54,3] == 'miss',r[i][:54,1] == 25)))
    hit25[i,1] = sum(np.bitwise_and(r[i][54:108,3] == 'hit',r[i][54:108,1] == 25))/(
        sum(np.bitwise_and(r[i][54:108,3] == 'hit',r[i][54:108,1] == 25))+sum(np.bitwise_and(r[i][54:108,3] == 'miss',r[i][54:108,1] == 25)))
    hit25[i,2] = sum(np.bitwise_and(r[i][108:162,3] == 'hit',r[i][108:162,1] == 25))/(
        sum(np.bitwise_and(r[i][108:162,3] == 'hit',r[i][108:162,1] == 25))+sum(np.bitwise_and(r[i][108:162,3] == 'miss',r[i][108:162,1] == 25)))
    hit50[i,0] = sum(np.bitwise_and(r[i][:54,3] == 'hit',r[i][:54,1] == 50))/(
        sum(np.bitwise_and(r[i][:54,3] == 'hit',r[i][:54,1] == 50))+sum(np.bitwise_and(r[i][:54,3] == 'miss',r[i][:54,1] == 50)))
    hit50[i,1] = sum(np.bitwise_and(r[i][54:108,3] == 'hit',r[i][54:108,1] == 50))/(
        sum(np.bitwise_and(r[i][54:108,3] == 'hit',r[i][54:108,1] == 50))+sum(np.bitwise_and(r[i][54:108,3] == 'miss',r[i][54:108,1] == 50)))
    hit50[i,2] = sum(np.bitwise_and(r[i][108:162,3] == 'hit',r[i][108:162,1] == 50))/(
        sum(np.bitwise_and(r[i][108:162,3] == 'hit',r[i][108:162,1] == 50))+sum(np.bitwise_and(r[i][108:162,3] == 'miss',r[i][108:162,1] == 50)))
    hit500[i,0] = sum(np.bitwise_and(r[i][:54,3] == 'hit',r[i][:54,1] == 500))/(
        sum(np.bitwise_and(r[i][:54,3] == 'hit',r[i][:54,1] == 500))+sum(np.bitwise_and(r[i][:54,3] == 'miss',r[i][:54,1] == 500)))
    hit500[i,1] = sum(np.bitwise_and(r[i][54:108,3] == 'hit',r[i][54:108,1] == 500))/(
        sum(np.bitwise_and(r[i][54:108,3] == 'hit',r[i][54:108,1] == 500))+sum(np.bitwise_and(r[i][54:108,3] == 'miss',r[i][54:108,1] == 500)))
    hit500[i,2] = sum(np.bitwise_and(r[i][108:162,3] == 'hit',r[i][108:162,1] == 500))/(
        sum(np.bitwise_and(r[i][108:162,3] == 'hit',r[i][108:162,1] == 500))+sum(np.bitwise_and(r[i][108:162,3] == 'miss',r[i][108:162,1] == 500)))

VI25 = (hit25-fa)/(2*(hit25+fa)-(hit25+fa)*(hit25+fa)); 
VI50 = (hit50-fa)/(2*(hit50+fa)-(hit50+fa)*(hit50+fa));
VI500 = (hit500-fa)/(2*(hit500+fa)-(hit500+fa)*(hit500+fa));
avgV25 = [np.mean(VI25[:,0]), np.mean(VI25[:,1]), np.mean(VI25[:,2])]
stdV25 = [np.std(VI25[:,0]), np.std(VI25[:,1]), np.std(VI25[:,2])]
avgV50 = [np.mean(VI50[:,0]), np.mean(VI50[:,1]), np.mean(VI50[:,2])]
stdV50 = [np.std(VI50[:,0]), np.std(VI50[:,1]), np.std(VI50[:,2])]
avgV500 = [np.mean(VI500[:,0]), np.mean(VI500[:,1]), np.mean(VI500[:,2])]
stdV500 = [np.std(VI500[:,0]), np.std(VI500[:,1]), np.std(VI500[:,2])]


plt.plot(['block1', 'block2', 'block3'], VI[0,:], marker = 'o', linestyle ='dashed'); plt.plot(['block1', 'block2', 'block3'], VI[1,:], marker = 'o', linestyle ='dashed')
plt.plot(['block1', 'block2', 'block3'], VI[2,:], marker = 'o', linestyle ='dashed'); plt.plot(['block1', 'block2', 'block3'], VI[3,:], marker = 'o', linestyle ='dashed')
plt.plot(['block1', 'block2', 'block3'], VI[4,:], marker = 'o', linestyle ='dashed'); 
plt.errorbar(['block1', 'block2', 'block3'], avgV, marker = 'o', yerr = stdV)
plt.ylabel('VI'); plt.figure()
plt.errorbar(['block1', 'block2', 'block3'], avgV25, marker = 'o', yerr = stdV25, label='VI25')
plt.errorbar(['block1', 'block2', 'block3'], avgV50, marker = 'o', yerr = stdV50, label='VI50')
plt.errorbar(['block1', 'block2', 'block3'], avgV500, marker = 'o', yerr = stdV500, label='VI500')
plt.errorbar(['block1', 'block2', 'block3'], avgV, marker = 'o', yerr = stdV, label = 'VI')
plt.ylabel('VI'); plt.legend(); plt.figure()

#%%
#looking at response latencies

#RL analysis wrt signal duration/signals

latSign = np.full((5,4), 0.0) #avg response latency for each signal type for each rat
latSignStd = np.full((5,4), 0.0)#(within rat)stds associated with the above for each rat 
sign = [25,50,500,0] #4 possible signals

for i in range(len(sign)):
    latSign[0,i] = np.nanmean((rat1[(np.where(rat1[:,1] == sign[i])),4]).astype(float))
    latSignStd[0,i] = np.nanstd((rat1[(np.where(rat1[:,1] == sign[i])),4]).astype(float))
    latSign[1,i] = np.nanmean((rat2[(np.where(rat2[:,1] == sign[i])),4]).astype(float))
    latSignStd[1,i] = np.nanstd((rat2[(np.where(rat2[:,1] == sign[i])),4]).astype(float))
    latSign[2,i] = np.nanmean((rat3[(np.where(rat3[:,1] == sign[i])),4]).astype(float))
    latSignStd[2,i] = np.nanstd((rat3[(np.where(rat3[:,1] == sign[i])),4]).astype(float))
    latSign[3,i] = np.nanmean((rat4[(np.where(rat4[:,1] == sign[i])),4]).astype(float))
    latSignStd[3,i] = np.nanstd((rat4[(np.where(rat4[:,1] == sign[i])),4]).astype(float))
    latSign[4,i] = np.nanmean((rat5[(np.where(rat5[:,1] == sign[i])),4]).astype(float))
    latSignStd[4,i] = np.nanstd((rat5[(np.where(rat5[:,1] == sign[i])),4]).astype(float))
    
meanLatSign = [np.mean(latSign[:,0]), np.mean(latSign[:,1]), np.mean(latSign[:,2]), np.mean(latSign[:,3])]
errLatSign = [np.std(latSign[:,0]), np.std(latSign[:,1]), np.std(latSign[:,2]), np.std(latSign[:,3])]

plt.bar(['25', '50', '500', 'default'], meanLatSign, yerr = errLatSign)
plt.xlabel('signal duration in ms'); plt.ylabel('mean response latency over 5 rats (in ms)')
plt.figure()

#1-way anova with signal duration as ind var, and avg RL from each rat as sample points
print('one-way anova across signal durations with avg RLs from each rat as sample points: (defaults excluded/ included)')
print(scipy.stats.f_oneway(latSign[:,0], latSign[:,1], latSign[:,2]))
print(scipy.stats.f_oneway(latSign[:,0], latSign[:,1], latSign[:,2], latSign[:,3]))

#RL analysis wrt trial types (response types)

latType = np.full((5,4), 0.0) #avg response latency for each response type for each rat
latTypeStd = np.full((5,4), 0.0) #(within rat) std associated with the above for each rat
ty = ['hit', 'miss', 'cr', 'fa'] #4 diff response types

for i in range(len(ty)):
    latType[0,i] = np.nanmean((rat1[(np.where(rat1[:,3] == ty[i])),4]).astype(float))
    latTypeStd[0,i] = np.nanstd((rat1[(np.where(rat1[:,3] == ty[i])),4]).astype(float))
    latType[1,i] = np.nanmean((rat2[(np.where(rat2[:,3] == ty[i])),4]).astype(float))
    latTypeStd[1,i] = np.nanstd((rat2[(np.where(rat2[:,3] == ty[i])),4]).astype(float))
    latType[2,i] = np.nanmean((rat3[(np.where(rat3[:,3] == ty[i])),4]).astype(float))
    latTypeStd[2,i] = np.nanstd((rat3[(np.where(rat3[:,3] == ty[i])),4]).astype(float))
    latType[3,i] = np.nanmean((rat4[(np.where(rat4[:,3] == ty[i])),4]).astype(float))
    latTypeStd[3,i] = np.nanstd((rat4[(np.where(rat4[:,3] == ty[i])),4]).astype(float))
    latType[4,i] = np.nanmean((rat5[(np.where(rat5[:,3] == ty[i])),4]).astype(float))
    latTypeStd[4,i] = np.nanstd((rat5[(np.where(rat5[:,3] == ty[i])),4]).astype(float))

meanLatTy = [np.mean(latType[:,0]), np.mean(latType[:,1]), np.mean(latType[:,2]), np.mean(latType[:,3])]
errLatTy = [np.std(latType[:,0]), np.std(latType[:,1]), np.std(latType[:,2]), np.std(latType[:,3])]

plt.bar(ty, meanLatTy, yerr = errLatTy)
plt.xlabel('trial type'); plt.ylabel('mean response latency over 5 rats (in ms)') 
plt.figure() 

#1-way anova with signal duration as ind var, and avg RL from each rat as sample points
print('one-way anova across trial types with avg RLs from each rat as sample points: ')
print(scipy.stats.f_oneway(latType[:,0], latType[:,1], latType[:,2], latType[:,3]))  

#RL analysis wrt rats

meanLat = [np.nanmean((rat1[:,4]).astype(float)), np.nanmean((rat2[:,4]).astype(float)),
           np.nanmean((rat3[:,4]).astype(float)), np.nanmean((rat4[:,4]).astype(float)),
           np.nanmean((rat5[:,4]).astype(float))]
errLat = [np.nanstd((rat1[:,4]).astype(float)), np.nanstd((rat2[:,4]).astype(float)),
          np.nanstd((rat3[:,4]).astype(float)), np.nanstd((rat4[:,4]).astype(float)),
          np.nanstd((rat5[:,4]).astype(float))]

plt.bar(['rat1', 'rat2', 'rat3', 'rat4', 'rat5'], meanLat, yerr = errLat)
plt.xlabel('rat no.'); plt.ylabel('mean response latency for each rat (in ms)')
plt.figure()

#1-way anova with rat as ind variable and trial-wise RLs as sample points
print('one-way anova across rats with raw RLs from each rat as sample points:')
print(scipy.stats.f_oneway(rat1[~np.isnan((rat1[:,4]).astype(float)),4],
                           rat2[~np.isnan((rat2[:,4]).astype(float)),4],
                           rat3[~np.isnan((rat3[:,4]).astype(float)),4],
                           rat4[~np.isnan((rat4[:,4]).astype(float)),4],
                           rat5[~np.isnan((rat5[:,4]).astype(float)),4])) #remove nan values

#RL wrt signal but for each rat

plt.bar(['25', '50', '500', '0'],latSign[0,:],yerr = latSignStd[0,:], label = 'rat1')
plt.bar(['1-25', '1-50', '1-500', '1-0'],latSign[1,:],yerr = latSignStd[1,:], label = 'rat2')
plt.bar(['2-25', '2-50', '2-500', '2-0'],latSign[2,:],yerr = latSignStd[2,:], label = 'rat3')
plt.bar(['3-25', '3-50', '3-500', '3-0'],latSign[3,:],yerr = latSignStd[3,:], label = 'rat4')
plt.bar(['4-25', '4-50', '4-500', '4-0'],latSign[4,:],yerr = latSignStd[4,:], label = 'rat5')
plt.xlabel('signal duration(in ms)'); plt.ylabel('mean response latency  (in ms)')
plt.legend(); plt.figure()

#1-way anova for diff signal durations, within each rat
r = [rat1, rat2, rat3, rat4, rat5]
a = [rat1[~np.isnan((rat1[:,4]).astype(float)),4],
     rat2[~np.isnan((rat2[:,4]).astype(float)),4],
     rat3[~np.isnan((rat3[:,4]).astype(float)),4],
     rat4[~np.isnan((rat4[:,4]).astype(float)),4],
     rat5[~np.isnan((rat5[:,4]).astype(float)),4]]
b = [rat1[~np.isnan((rat1[:,4]).astype(float)),1],
     rat2[~np.isnan((rat2[:,4]).astype(float)),1],
     rat3[~np.isnan((rat3[:,4]).astype(float)),1],
     rat4[~np.isnan((rat4[:,4]).astype(float)),1],
     rat5[~np.isnan((rat5[:,4]).astype(float)),1]]
print('one-way anova across signal durations for each rat, raw RLs as sample points: (defaults excluded/ included)')
for i in range(len(r)):
    print(scipy.stats.f_oneway(a[i][(np.where(b[i] == 25))],
                           a[i][(np.where(b[i] == 50))],
                           a[i][(np.where(b[i] == 500))],
                           a[i][(np.where(b[i] == 0))]))
    print(scipy.stats.f_oneway(a[i][(np.where(b[i] == 25))],
                           a[i][(np.where(b[i] == 50))],
                           a[i][(np.where(b[i] == 500))]))
    
    
#RL wrt trial types but for each rat

plt.bar(['h', 'm', 'c', 'f'],latType[0,:],yerr = latTypeStd[0,:], label = 'rat1')
plt.bar(['1h', '1m', '1c', '1f'],latType[1,:],yerr = latTypeStd[1,:], label = 'rat2')
plt.bar(['2h', '2m', '2c', '2f'],latType[2,:],yerr = latTypeStd[2,:], label = 'rat3')
plt.bar(['3h', '3m', '3c', '3f'],latType[3,:],yerr = latTypeStd[3,:], label = 'rat4')
plt.bar(['4h', '4m', '4c', '4f'],latType[4,:],yerr = latTypeStd[4,:], label = 'rat5')
plt.xlabel('trial type'); plt.ylabel('mean response latency  (in ms)')
plt.legend(); plt.figure()

#1-way anova for diff signal durations, within each rat
r = [rat1, rat2, rat3, rat4, rat5]
a = [rat1[~np.isnan((rat1[:,4]).astype(float)),4],
     rat2[~np.isnan((rat2[:,4]).astype(float)),4],
     rat3[~np.isnan((rat3[:,4]).astype(float)),4],
     rat4[~np.isnan((rat4[:,4]).astype(float)),4],
     rat5[~np.isnan((rat5[:,4]).astype(float)),4]]
b = [rat1[~np.isnan((rat1[:,4]).astype(float)),3],
     rat2[~np.isnan((rat2[:,4]).astype(float)),3],
     rat3[~np.isnan((rat3[:,4]).astype(float)),3],
     rat4[~np.isnan((rat4[:,4]).astype(float)),3],
     rat5[~np.isnan((rat5[:,4]).astype(float)),3]]
print('one-way anova across trial type for each rat, raw RLs as sample points:')
for i in range(len(r)):
    print(scipy.stats.f_oneway(a[i][(np.where(b[i] == 'hit'))],
                           a[i][(np.where(b[i] == 'miss'))],
                           a[i][(np.where(b[i] == 'cr'))],
                           a[i][(np.where(b[i] == 'fa'))]))


#%%

# a summary


summary = pd.DataFrame(index = ['rat1', 'rat2', 'rat3', 'rat4', 'rat5'])
summary['subject'] = [3, 6, 11, 7, 8]
summary['total trials'] =  [162-sum(missingTrials[0,:]), 
                            162-sum(missingTrials[1,:]), 
                            162-sum(missingTrials[2,:]), 
                            162-sum(missingTrials[3,:]), 
                            162-sum(missingTrials[4,:])]
summary['hit'] = [sum(hitRates[0,:]), sum(hitRates[1,:]), sum(hitRates[2,:]), sum(hitRates[3,:]), 
                   sum(hitRates[4,:])]
summary['miss'] = [sum(missRates[0,:]), sum(missRates[1,:]), sum(missRates[2,:]), sum(missRates[3,:]), 
                   sum(missRates[4,:])]
summary['cr'] = [sum(crRates[0,:]), sum(crRates[1,:]), sum(crRates[2,:]), sum(crRates[3,:]), 
                   sum(crRates[4,:])]
summary['fa'] = [sum(faRates[0,:]), sum(faRates[1,:]), sum(faRates[2,:]), sum(faRates[3,:]), 
                   sum(faRates[4,:])]
summary['omit'] = [sum(omitRates[0,:]), 
                    sum(omitRates[1,:]), 
                    sum(omitRates[2,:]), 
                    sum(omitRates[3,:]), 
                   sum(omitRates[4,:])] 
summary['h25'] = countTypes(25, 'hit') 
summary['m25'] = countTypes(25, 'miss')
summary['h50'] = countTypes(50, 'hit')
summary['m50'] = countTypes(50, 'miss')
summary['h500'] = countTypes(500, 'hit')
summary['m500'] = countTypes(500, 'miss')

summary['%h'] = (summary['hit']*100)/(summary['hit']+summary['miss'])
summary['%fa'] = (summary['fa']*100)/(summary['fa']+summary['cr'])
summary['%m'] =  (summary['miss']*100)/(summary['hit']+summary['miss'])
summary['%cr'] = (summary['cr']*100)/(summary['fa']+summary['cr'])
summary['%omit'] = (summary['omit']*100)/(summary['total trials'])
summary['%h25'] = [hitPerc[0,0], hitPerc[1,0], hitPerc[2,0], hitPerc[3,0], hitPerc[4,0]]
summary['%h50'] = [hitPerc[0,1], hitPerc[1,1], hitPerc[2,1], hitPerc[3,1], hitPerc[4,1]]
summary['%h500'] = [hitPerc[0,2], hitPerc[1,2], hitPerc[2,2], hitPerc[3,2], hitPerc[4,2]]
summary['%m25'] = [missPerc[0,0], missPerc[1,0], missPerc[2,0], missPerc[3,0], missPerc[4,0]]
summary['%m50'] = [missPerc[0,1], missPerc[1,1], missPerc[2,1], missPerc[3,1], missPerc[4,1]]
summary['%m500'] = [missPerc[0,2], missPerc[1,2], missPerc[2,2], missPerc[3,2], missPerc[4,2]]
summary['rt h'] = [latType[0,0], latType[1,0], latType[2,0], latType[3,0], latType[4,0]]
summary['rt m'] = [latType[0,1], latType[1,1], latType[2,1], latType[3,1], latType[4,1]]
summary['rt cr'] = [latType[0,2], latType[1,2], latType[2,2], latType[3,2], latType[4,2]]
summary['rt fa'] = [latType[0,3], latType[1,3], latType[2,3], latType[3,3], latType[4,3]]

def tySig(signal, typ):
    c = np.full((5),0.0)
    c[0] = np.nanmean((rat1[np.intersect1d(np.where(rat1[:,1] == signal), np.where(rat1[:,3] == typ)), 4]).astype(float))
    c[1] = np.nanmean((rat2[np.intersect1d(np.where(rat2[:,1] == signal), np.where(rat2[:,3] == typ)), 4]).astype(float))
    c[2] = np.nanmean((rat3[np.intersect1d(np.where(rat3[:,1] == signal), np.where(rat3[:,3] == typ)), 4]).astype(float))
    c[3] = np.nanmean((rat4[np.intersect1d(np.where(rat4[:,1] == signal), np.where(rat4[:,3] == typ)), 4]).astype(float))
    c[4] = np.nanmean((rat5[np.intersect1d(np.where(rat5[:,1] == signal), np.where(rat5[:,3] == typ)), 4]).astype(float))
    
    return c

summary['rt h25'] = tySig(25, 'hit'); summary['rt m25'] = tySig(25, 'miss')
summary['rt h50'] = tySig(50, 'hit'); summary['rt m50'] = tySig(50, 'miss')
summary['rt h500'] = tySig(500, 'hit');  summary['rt m500'] = tySig(500, 'miss')

summary.to_csv('summaryHoweRaw.csv', index = False)

#%%

#a trial based (time) analysis

#linear regression for RL with prev RL, current trial type, signal, time t (trial no.) and 
# rat no. as predictors

#within a rat : no rat no. predictor

for i in range(len(r)):
    X = np.full((162, 4), np.nan) #predictors
    Y = np.full((162,1), np.nan) #regressor
    
    X[1:,0] = r[i][:-1,4]; #prev RLs
    a = r[i][:,3]; a = np.where(a == 'hit', 0, a);  a = np.where(a == 'miss', 1, a);
    a = np.where(a == 'cr', 2, a);  a = np.where(a == 'fa', 3, a);  a = np.where(a == 'omit', 5, a)
    X[:,1] = a #trial types
    X[:,2] = r[i][:,1] #signal duration
    X[:,3] = np.linspace(0,161,num=162) #time point
    Y[:,0] = r[i][:,4] #RLs as the dependent variable
    w = np.append(X, Y, axis = 1); w = w[~np.isnan(w).any(axis=1)] #removing nans
    X = w[:,:-1]; Y = w[:,4] # predictor and regressors after removing nan rows
    
    x = sm.add_constant(X) #allowing intercept in model
    model = sm.OLS(Y, x) #fitting ordinary least squares
    results = model.fit(); print('results of rat', i); print(results.summary()) #results of fit

#all rats together
rat = np.vstack((rat1, rat2, rat3,rat4, rat5))

X = np.full((810, 5), np.nan) #predictors
Y = np.full((810, 1), np.nan) #regressor
X[:162,4] = 0; X[162:324,4] = 1; X[324:486,4] = 2; X[486:648,4] = 3; X[648:810,4] = 4; 
X[:162,3] = np.linspace(0,161,num=162); #rat nos.
X[162:324,3] = np.linspace(0,161,num=162) ; X[324:486,3] = np.linspace(0,161,num=162); 
X[486:648,3] = np.linspace(0,161,num=162); X[648:810,3] = np.linspace(0,161,num=162) #time points
X[1:,0] = rat[:-1,4]; #prev RLs
a = rat[:,3]; a = np.where(a == 'hit', 0, a);  a = np.where(a == 'miss', 1, a);
a = np.where(a == 'cr', 2, a);  a = np.where(a == 'fa', 3, a);  a = np.where(a == 'omit', 5, a)
X[:,1] = a; #trial types
X[:,2] = rat[:,1]; #signal duration
Y[:,0] = rat[:,4] #RLs as dependent variable
w = np.append(X, Y, axis = 1); w = w[~np.isnan(w).any(axis=1)] #removing nans
np.random.shuffle(w) #shuffle rows
X = w[:,:-1]; Y = w[:,5] #predictors and regressor after removing nans

x = sm.add_constant(X) #allowing intercept in  model
model = sm.OLS(Y, x) #fitting ordinary least squares
results = model.fit(); print('results of rat'); print(results.summary()) #result of fit


#linear regression for RLs (as above) but with one hot coding for trial types
for i in range(len(r)):
    X = np.full((162, 7), np.nan) #predictors
    Y = np.full((162,1), np.nan) #regressor
    
    X[1:,0] = r[i][:-1,4]; #prev RLs
    X[:,1] = r[i][:,1] #signal duration
    X[:,2] = np.linspace(0,161,num=162) #time point
    #one hot coding for hit, miss, cr and fa
    a = r[i][:,3]; a = np.where(a == 'hit', 1, 0); X[:,3] = a
    a = r[i][:,3]; a = np.where(a == 'miss', 1, 0); X[:,4] = a
    a = r[i][:,3]; a = np.where(a == 'cr', 1, 0); X[:,5] = a
    a = r[i][:,3]; a = np.where(a == 'fa', 1, 0); X[:,6] = a
    
    Y[:,0] = r[i][:,4] #RLs as the dependent variable
    w = np.append(X, Y, axis = 1); w = w[~np.isnan(w).any(axis=1)] #removing nans
    X = w[:,:-1]; Y = w[:,7] # predictor and regressors after removing nan rows
    
    x = sm.add_constant(X) #allowing intercept in model
    model = sm.OLS(Y, x) #fitting ordinary least squares
    results = model.fit(); print('results of rat', i); print(results.summary()) #results of fit


#all rats together
rat = np.vstack((rat1, rat2, rat3,rat4, rat5))
X = np.full((810, 8), np.nan) #predictors
Y = np.full((810,1), np.nan) #regressor

X[1:,0] = rat[:-1,4]; #prev RLs
X[:,1] = rat[:,1] #signal duration
X[:162,2] = 0; X[162:324,2] = 1; X[324:486,2] = 2; X[486:648,2] = 3; X[648:810,2] = 4;  #rat nos.
X[:162,3] = np.linspace(0,161,num=162);
X[162:324,3] = np.linspace(0,161,num=162) ; X[324:486,3] = np.linspace(0,161,num=162); 
X[486:648,3] = np.linspace(0,161,num=162); X[648:810,3] = np.linspace(0,161,num=162) #time points
a = rat[:,3]; a = np.where(a == 'hit', 1, 0); X[:,4] = a
a = rat[:,3]; a = np.where(a == 'miss', 1, 0); X[:,5] = a
a = rat[:,3]; a = np.where(a == 'cr', 1, 0); X[:,6] = a
a = rat[:,3]; a = np.where(a == 'fa', 1, 0); X[:,7] = a
Y[:,0] = rat[:,4] #RLs as dependent variable
w = np.append(X, Y, axis = 1); w = w[~np.isnan(w).any(axis=1)] #removing nans
#np.random.shuffle(w) #shuffle rows
X = w[:,:-1]; Y = w[:,8]
x = sm.add_constant(X) #allowing intercept in  model
model = sm.OLS(Y, x) #fitting ordinary least squares
results = model.fit(); print('results of rat'); print(results.summary()) #result of fit

plt.scatter(X[:,0],Y); plt.xlabel('Prev RL (in ms)'); plt.ylabel('RL (in ms)'); plt.figure() #RLs vs prev RLs
plt.scatter(X[:,2],Y); plt.xlabel('Rat no.'); plt.ylabel('RL (in ms)'); plt.figure() #RLs vs rat nos. 
f = Y-results.fittedvalues; #diff bettwen predicted and observed Ys (residual)
print(scipy.stats.ttest_1samp(f, 0)); plt.plot(f); plt.figure() #checking if residuals are distributed about 0
#%%
#multinomial logit regression for trial type with prev trial type, current RL, signal, 
#time t (trial no.)and rat no. as predictors

#within a rat : no rat no. predictor

for i in range(len(r)):

    X = np.full((162, 4), np.nan) #predictors
    Y = np.full((162, 1), np.nan) #regressor
    
    a = r[i][:,3]; a = np.where(a == 'hit', 0, a);  a = np.where(a == 'miss', 1, a);
    a = np.where(a == 'cr', 2, a);  a = np.where(a == 'fa', 3, a);  a = np.where(a == 'omit', np.nan, a)
    Y[:,0] = a #trial type as dependent variable
    X[1:,0] = a[:-1]; #prev trial type
    X[:,1] = r[i][:,4]; #RLs
    X[:,2] = (r[i][:,1])/1000; #signal duration
    X[:,3] = np.linspace(0,161,num=162) #timepoints
    w = np.append(X, Y, axis = 1); w = w[~np.isnan(w).any(axis=1)] #removing nans
    X = w[:,:-1]; Y = w[:,4] #predictors and regressor after removing nans
    x = sm.add_constant(X[:,np.array([0,1,3])]) #allowing intercept in model
    model_logit = sm.MNLogit(Y, x) #fitting multinomial logit
    results_logit = model_logit.fit(); print('results of rat', i); print(results_logit.summary())
    #results of fit
    
#all rat data
rat = np.vstack((rat1, rat2, rat3,rat4, rat5)) #all rat data

X = np.full((810, 5), np.nan) #predictors
Y = np.full((810, 1), np.nan) #regressor
a = rat[:,3]; a = np.where(a == 'hit', 0, a);  a = np.where(a == 'miss', 1, a);
a = np.where(a == 'cr', 2, a);  a = np.where(a == 'fa', 3, a);  a = np.where(a == 'omit', np.nan, a)
Y[:,0] = a #trial types as dependent variable
X[1:,0] = a[:-1]; #prev trial types 
X[:,1] = rat[:,4]; #RLs
X[:,2] = (rat[:,1])/1000; #signal duration
X[:162,4] = 0; X[162:324,4] = 1; X[324:486,4] = 2; X[486:648,4] = 3; X[648:810,4] = 4; #rat nos. 
X[:162,3] = np.linspace(0,161,num=162); X[162:324,3] = np.linspace(0,161,num=162) ;  
X[324:486,3] = np.linspace(0,161,num=162); X[486:648,3] = np.linspace(0,161,num=162);
X[648:810,3] = np.linspace(0,161,num=162) #time points

w = np.append(X, Y, axis = 1); w = w[~np.isnan(w).any(axis=1)] #removing rows with nans
np.random.shuffle(w) #shuffling rows
X = w[:,:-1]; Y = w[:,5] #predictors and regressors after removing rows with nans
x = sm.add_constant(X[:,np.array([0,1,3,4])]) #allowing intercept in the model
model_logit = sm.MNLogit(Y, x) #fitting multinomial logit
results_logit = model_logit.fit();  print(results_logit.summary()) #results

#%%
# no. accurate responses in bins across time
#correct response=1, wrong=0, om = nan
c = np.full((5,8), 0); #array for no. of accurate responses for 5 rats in each bin
t = [1,2,3,4,5,6,7,8] #time bin nos.
for i in range(len(r)):
    #getting correct responses as 1 and wrong as 0
    a = r[i][:,3]; a = np.where(a == 'hit', 1, a);  a = np.where(a == 'miss', 0, a);
    a = np.where(a == 'cr', 1, a);  a = np.where(a == 'fa', 0, a);  a = np.where(a == 'omit', np.nan, a)
    a = a.astype(float) #trial types for rat i
    b = [] #list with no. of accurate response in each bin for each rat
    for j in range(0, len(a)-2, 20):
        b = b + [np.nansum(a[j:j+20])] #summing no. of correct responses in a bin
    c[i,:] = b #assigning b to c for each rat
    plt.plot(t, b, marker='o', label = 'rat no.%d'%(i+1)); #plotting the nos. for each rat across bins
    plt.xlabel('bin nos.'); plt.ylabel('no. of correct responses');
plt.legend()
avg = np.mean(c, axis = 0); std = np.std(c, axis = 0); plt.figure() #avg and std types nos. across 5 rats for each bbin
plt.errorbar(t, avg, yerr = std, marker = 'o', label ='all rats'); #plotting avg and std across bins
plt.xlabel('bin nos.'); plt.ylabel('no. of correct responses');
plt.legend(); plt.figure()
print('oneway anova for # of correct trials for 8 time bins (across rats)')
print(scipy.stats.f_oneway(c[:,0], c[:,1], c[:,2], c[:,3], c[:,4], c[:,5], c[:,6], c[:,7]))


#avg RLs in bins across time
c = np.full((5,8), 0);# array for avg RLs in each bin
t = [1,2,3,4,5,6,7,8] #time bin nos.
for i in range(len(r)):
    a = (r[i][:,4]).astype(float); #RLs for rat i
    b = []; d=[] #arrays for avg and std RLs in each bin for each rat
    for j in range(0, len(a)-2, 20):
        b = b + [np.nanmean(a[j:j+20])] #avg RLs for a bin
        d = d + [np.nanstd(a[j:j+20])] #std RLs for a bin
    c[i,:] = b; #assigning b to c for each rat 
    plt.errorbar(t, b, yerr = d, marker='o', label = 'rat no.%d'%(i+1)); #plotting RL avg and std for each in a rat
    plt.xlabel('bin nos.'); plt.ylabel('avg RLs');
plt.legend()
avg = np.mean(c, axis = 0); std = np.std(c, axis = 0); plt.figure() #avg and std RL across 5 rats for each bin
plt.errorbar(t, avg, yerr = std, marker = 'o', label ='all rats'); #plotting avg and std
plt.xlabel('bin nos.'); plt.ylabel('avg RLs');
plt.legend(); plt.figure()
print('oneway anova for avg RL for 8 time bins (across rats)')
print(scipy.stats.f_oneway(c[:,0], c[:,1], c[:,2], c[:,3], c[:,4], c[:,5], c[:,6], c[:,7]))

#%%

#from pymer4.models import Lmer

#all data - RLs, prev RLs, signal duran, response, prev response,
#trial type, rat nos., time points, congruent/incongruent 

X = np.full((810, 13), np.nan) #predictors
X[:,0] = rat[:,4]*0.001 #RLs as dependent variable
X[1:,1] = rat[:-1,4]*0.001 #lagged RLs
X[:,2] = rat[:,1]*0.001 #signal duration
X[:,3] = rat[:,2] #response
X[1:,4] = rat[:-1,2] #lagged response
a = rat[:,3]; a = np.where(a == 'hit', 1, 0); X[:,5] = a
a = rat[:,3]; a = np.where(a == 'miss', 1, 0); X[:,6] = a
a = rat[:,3]; a = np.where(a == 'cr', 1, 0); X[:,7] = a
a = rat[:,3]; a = np.where(a == 'fa', 1, 0); X[:,8] = a #trial types
X[:162,9] = 0; X[162:324,9] = 1; X[324:486,9] = 2; X[486:648,9] = 3; X[648:810,9] = 4; #rat nos.
X[:162,10] = np.linspace(0,161,num=162); X[162:324,10] = np.linspace(0,161,num=162) ;
X[324:486,10] = np.linspace(0,161,num=162); X[486:648,10] = np.linspace(0,161,num=162);
X[648:810,10] = np.linspace(0,161,num=162) #time points
X[:,10] = X[:,10]*0.001
a = rat[:,5]; b = (a == 'hpcr') + (a == 'hpm'); a = np.where(b, 1, 0 ); X[:,11] = a; #incogruent hit or not
a = rat[:,5];  a = np.where(a == 'hph', 1, 0 ); X[:,12] = a;
X = X[~np.isnan(X).any(axis=1)] #removing nans
#np.random.shuffle(X) #shuffle rows

ratDF = pd.DataFrame({'RL': X[:,0], 'laggedRL': X[:,1], 'signalDuration': X[:,2],
                      'response': X[:,3],'laggedResponse': X[:,4], 'hit': X[:,5], 
                      'miss': X[:,6], 'cr': X[:,7], 'fa': X[:,8], 'rats': X[:,9], 
                      'time': X[:,10], 'incongHit': X[:,11], 'congHit': X[:,12]})

model = smf.mixedlm("RL ~ laggedRL+time+hit+miss+cr+signalDuration+congHit+incongHit", ratDF, groups = ratDF["rats"], 
                    re_formula = "~laggedRL+time+hit+miss+cr+signalDuration+congHit+incongHit")
result = model.fit()
print(result.summary())

model = smf.mixedlm("RL ~ incongHit+laggedRL+time+signalDuration+response+congHit", ratDF, groups = ratDF["rats"], 
                    re_formula = "~incongHit+laggedRL+time+signalDuration+response+congHit")
result = model.fit()
print(result.summary())

model = smf.mixedlm("RL ~ hit+miss+cr", ratDF, groups = ratDF["rats"], 
                    re_formula = "~hit+miss+cr")
result = model.fit()
print(result.summary())

#incongHit+laggedRL+time+signalDuration+response+
#%%

#congruent and incongruent trials
rlType = np.full((5,5),np.nan) #avg RL for incong and cong hits and crs, everything else
rlTystd = np.full((5,5),np.nan) #corresponding stds
ty = ['incong hits', 'cong hits', 'incong crs', 'cong crs',
      '!incong hits']

for i in range(len(r)):
    
    rlType[i,0] = np.nanmean((r[i][np.union1d(np.where(r[i][:,5] == 'hpcr'), 
                                  np.where(r[i][:,5] == 'hpm')), 4]).astype(float))
    rlType[i,1] = np.nanmean((r[i][np.where(r[i][:,5] == 'hph'), 4]).astype(float))
    rlType[i,2] = np.nanmean((r[i][np.where(r[i][:,5] == 'crph'), 4]).astype(float))
    rlType[i,3] = np.nanmean((r[i][np.where(r[i][:,5] == 'crpcr'), 4]).astype(float))
    rlType[i,4] = np.nanmean((r[i][np.intersect1d(np.where(~(r[i][:,5] == 'hpcr')), 
                                  np.where(~(r[i][:,5] == 'hpm'))), 4]).astype(float))
    
    
    rlTystd[i,0] = np.nanstd((r[i][np.union1d(np.where(r[i][:,5] == 'hpcr'), 
                                  np.where(r[i][:,5] == 'hpm')), 4]).astype(float))
    rlTystd[i,1] = np.nanstd((r[i][np.where(r[i][:,5] == 'hph'), 4]).astype(float))
    rlTystd[i,2] = np.nanstd((r[i][np.where(r[i][:,5] == 'crph'), 4]).astype(float))
    rlTystd[i,3] = np.nanstd((r[i][np.where(r[i][:,5] == 'crpcr'), 4]).astype(float))
    rlTystd[i,4] = np.nanstd((r[i][np.intersect1d(np.where(~(r[i][:,5] == 'hpcr')), 
                                  np.where(~(r[i][:,5] == 'hpm'))), 4]).astype(float))
    
    plt.errorbar(ty, rlType[i,:], yerr = rlTystd[i,:], marker='o', label = 'rat no.%d'%(i+1))
    plt.xlabel('type'); plt.ylabel('Avg RL for each rat in ms'); plt.legend()
   

rlAll = [np.mean(rlType[:,0]), np.mean(rlType[:,1]), np.mean(rlType[:,2]), np.mean(rlType[:,3]),
         np.mean(rlType[:,4])]
rlstdAll =  [np.std(rlType[:,0]), np.std(rlType[:,1]), np.std(rlType[:,2]), np.std(rlType[:,3]),
         np.std(rlType[:,4])]
plt.figure()
plt.errorbar(ty, rlAll, yerr = rlTystd[i,:], marker='o');
plt.xlabel('type'); plt.ylabel('Avg Rl across rats in ms'); 

print('oneway anova avgRL (across rats)')
print(scipy.stats.f_oneway(rlType[:,0], rlType[:,1], rlType[:,2], rlType[:,3],
                           rlType[:,4]))
