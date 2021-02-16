#analysis of SAT data from  Eryn Donovan ofAaron kucinski's lab

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from contextlib import suppress

#%%
#load data

data = pd.read_excel('more trial by trial.xlsx', header = None)

data = pd.read_excel('trial by trial so many.xlsx', header = None)

#%%
#convert data into a more tractable form

nSessions = data.shape[1]-1 #no. of sessions
l = []#list for holding empty session nos.
for i in range(nSessions):
    if sum(data.iloc[4:4020,i+1]) == 0:
        l = l+[i+1] #collecting empty sessions to remove
data = data.drop(data.columns[l], axis=1) #remove empty sessions
nSessions = data.shape[1]-1 #no. of sessions, new

ratno = (np.linspace(1,nSessions , nSessions)).astype(int) #index of rats
time = (np.linspace(0,250, 251)).astype(int) #data for the 251 trials for each rat
idx = pd.MultiIndex.from_product([ratno, time], names = ('session no.', 'trial no.'))
#heirarchical indexing in dataframe
cols = ['rat','date','FA', 'RTFA', 'H500', 'RTH500', 'H25', 'RTH25', 'H50', 'M500', 'RTM500',
        'RTH50', 'CR', 'RTCR', 'M25', 'RTM25', 'M50', 'RTM50', 'X'] #the columns in the dataframe
rat = pd.DataFrame(index = idx, columns = cols) #dataframe to hold all data from 
# all the rats

for i in range(len(ratno)):
    y = np.array(data.iloc[:, i+1]) #stacked data of i+1th rat
    z = np.transpose(np.reshape(y[4:], (17, 251))).astype(int) #reshaped with diff columns for
                                                     #diff data
    x1 = np.full((251,1), data.iloc[1,i+1]); x2 = np.full((251,1), data.iloc[0,i+1]) 
    s = np.concatenate((x1,x2,z), axis = 1)
    rat.loc[i+1,0:251] = s #assigning to (i+1)th rat index in the combined data frame
    
#%%
#some summary stats
def get_summary(ratno, rat):
    summary = pd.DataFrame(index = ratno, columns = 
                           ['subject', 'total trials', 'date', 'hit', 'miss', 'cr', 'fa',
                            'omit'])
    summary.index.name = 'No.'
    summary['subject'] = data.loc[1, 1:].array
    summary['total trials'] = data.loc[3,1:].array
    summary['date'] = data.loc[0,1:].array
    
    for i in range(len(ratno)):
        summary.loc[i+1,'hit'] =  np.sum(rat.loc[i+1,'H500']) + np.sum(
            rat.loc[i+1,'H50'])+np.sum(rat.loc[i+1,'H25'])
        summary.loc[i+1,'miss'] =  np.sum(rat.loc[i+1,'M500']) + np.sum(
            rat.loc[i+1,'M50'])+np.sum(rat.loc[i+1,'M25'])
        summary.loc[i+1, 'cr'] = np.sum(rat.loc[i+1, 'CR'])
        summary.loc[i+1, 'fa'] = np.sum(rat.loc[i+1, 'FA'])
        summary.loc[i+1, 'omit'] = summary.loc[i+1, 'total trials'] - np.sum(summary.loc[i+1, 'hit' : 'fa'])
        summary.loc[i+1, 'h25'] = np.sum(rat.loc[i+1, 'H25']); summary.loc[i+1, 'm25'] = np.sum(rat.loc[i+1, 'M25'])
        summary.loc[i+1, 'h50'] = np.sum(rat.loc[i+1, 'H50']); summary.loc[i+1, 'm50'] = np.sum(rat.loc[i+1, 'M50'])
        summary.loc[i+1, 'h500'] = np.sum(rat.loc[i+1, 'H500']); summary.loc[i+1, 'm500'] = np.sum(rat.loc[i+1, 'M500'])
        
        with suppress(ZeroDivisionError): #ignore zerodivisionerror
            summary.loc[i+1, '%h'] = (summary.loc[i+1, 'hit']*100)/(summary.loc[i+1,'hit']+summary.loc[i+1,'miss'])
            summary.loc[i+1, '%fa'] = (summary.loc[i+1, 'fa']*100)/(summary.loc[i+1,'cr']+summary.loc[i+1,'fa'])    
            summary.loc[i+1, '%m'] = (summary.loc[i+1, 'miss']*100)/(summary.loc[i+1,'hit']+summary.loc[i+1,'miss'])
            summary.loc[i+1, '%cr'] = (summary.loc[i+1, 'cr']*100)/(summary.loc[i+1,'cr']+summary.loc[i+1,'fa'])    
            summary.loc[i+1, '%omit'] = (summary.loc[i+1,'omit']*100)/summary.loc[i+1, 'total trials']
            summary.loc[i+1, '%h25'] = np.sum(rat.loc[i+1,'H25']*100)/(np.sum(rat.loc[i+1,'H25'])+np.sum(rat.loc[i+1,'M25']))
            summary.loc[i+1, '%h50'] = np.sum(rat.loc[i+1, 'H50']*100)/(np.sum(rat.loc[i+1,'H50'])+np.sum(rat.loc[i+1,'M50']))
            summary.loc[i+1, '%h500'] = np.sum(rat.loc[i+1, 'H500']*100)/(np.sum(rat.loc[i+1,'H500'])+np.sum(rat.loc[i+1,'M500']))
            summary.loc[i+1, '%m25'] = np.sum(rat.loc[i+1,'M25']*100)/(np.sum(rat.loc[i+1,'H25'])+np.sum(rat.loc[i+1,'M25']))
            summary.loc[i+1, '%m50'] = np.sum(rat.loc[i+1, 'M50']*100)/(np.sum(rat.loc[i+1,'H50'])+np.sum(rat.loc[i+1,'M50']))
            summary.loc[i+1, '%m500'] = np.sum(rat.loc[i+1, 'M500']*100)/(np.sum(rat.loc[i+1,'H500'])+np.sum(rat.loc[i+1,'M500']))
            
            summary.loc[i+1, 'rt h'] = (np.sum(rat.loc[i+1,'RTH500']) + np.sum(
                rat.loc[i+1,'RTH50'])+np.sum(rat.loc[i+1,'RTH25']))/(summary.loc[i+1,'hit'])
            summary.loc[i+1, 'rt m'] = (np.sum(rat.loc[i+1,'RTM500']) + np.sum(
                rat.loc[i+1,'RTM50'])+np.sum(rat.loc[i+1,'RTM25']))/(summary.loc[i+1,'miss'])
            summary.loc[i+1, 'rt cr'] = (np.sum(rat.loc[i+1, 'RTCR']))/(summary.loc[i+1,'cr'])
            summary.loc[i+1, 'rt fa'] = (np.sum(rat.loc[i+1, 'RTFA']))/(summary.loc[i+1,'fa'])
            summary.loc[i+1, 'rt h25'] = (np.sum(rat.loc[i+1, 'RTH25']))/(np.sum(rat.loc[i+1, 'H25']))
            summary.loc[i+1, 'rt m25'] = (np.sum(rat.loc[i+1, 'RTM25']))/(np.sum(rat.loc[i+1, 'M25']))
            summary.loc[i+1, 'rt h50'] = (np.sum(rat.loc[i+1, 'RTH50']))/(np.sum(rat.loc[i+1, 'H50']))
            summary.loc[i+1, 'rt m50'] = (np.sum(rat.loc[i+1, 'RTM50']))/(np.sum(rat.loc[i+1, 'M50']))
            summary.loc[i+1, 'rt h500'] = (np.sum(rat.loc[i+1, 'RTH500']))/(np.sum(rat.loc[i+1, 'H500']))
            summary.loc[i+1, 'rt m500'] = (np.sum(rat.loc[i+1, 'RTM500']))/(np.sum(rat.loc[i+1, 'M500']))
    return summary

summary = get_summary(ratno, rat)
#%%
#manually assign to columns which were ignored due to supress 0divisionerror

summary.loc[54, 'rt m25'] = (np.sum(rat.loc[54, 'RTM25']))/(np.sum(rat.loc[54, 'M25']))
summary.loc[54, 'rt h50'] = (np.sum(rat.loc[54, 'RTH50']))/(np.sum(rat.loc[54, 'H50']))
summary.loc[54, 'rt h500'] = (np.sum(rat.loc[54, 'RTH500']))/(np.sum(rat.loc[54, 'H500']))
summary.loc[54, 'rt m500'] = (np.sum(rat.loc[54, 'RTM500']))/(np.sum(rat.loc[54, 'M500']))

summary.loc[62, 'rt m50'] = (np.sum(rat.loc[62, 'RTM50']))/(np.sum(rat.loc[62, 'M50']))
summary.loc[62, 'rt h500'] = (np.sum(rat.loc[62, 'RTH500']))/(np.sum(rat.loc[62, 'H500']))
summary.loc[62, 'rt m500'] = (np.sum(rat.loc[62, 'RTM500']))/(np.sum(rat.loc[62, 'M500']))

#%%
#learning in rats
x = np.unique(data.loc[1,:].array[1:].astype(int))
for i in range(len(x)):
    isub = np.where(summary['subject'].array == x[i])[0] #choose ith subject, get indices 
    date = data.iloc[0,isub+1].array.astype('U') #date of occurence
    plt.figure()
    for j in range(len(isub)):
        plt.plot(['25', '50', '500', '0'], [summary.iloc[isub[j],19],
        summary.iloc[isub[j],20],summary.iloc[isub[j],21],
        summary.iloc[isub[j],17]], label = '%s'%date[j], marker = 'o', linestyle = 'dashed')
    plt.legend(); plt.title('rat%d'%x[i]); plt.xlabel('signal duration')
    plt.ylabel('accuracy')

#%%
#removing below-criterion sessions

a = summary.loc[:,'%h500'].array; x = np.where(a<70); #session nos. with %hits less than 70
a = summary.loc[:,'%cr'].array; y = np.where(a<70); #session nos. with %hits less than 70
a = summary.loc[:,'%omit'].array; z = np.where(a>10);#session nos. with #of omits greater than 10

l = np.union1d(np.union1d(x,y), z)#union of the two

data = data.drop(data.columns[l+1], axis=1) #drop columns according to l
nSessions = data.shape[1]-1 #no. of sessions, new

ratno = (np.linspace(1,nSessions , nSessions)).astype(int) #index of rats
time = (np.linspace(0,250, 251)).astype(int) #data for the 251 trials for each rat
idx = pd.MultiIndex.from_product([ratno, time], names = ('session no.', 'trial no.'))#heirarchical indexing in dataframe
cols = ['rat','date','FA', 'RTFA', 'H500', 'RTH500', 'H25', 'RTH25', 'H50', 'M500', 'RTM500',
        'RTH50', 'CR', 'RTCR', 'M25', 'RTM25', 'M50', 'RTM50', 'X']#the columns in the dataframe

rat1 = pd.DataFrame(index = idx, columns = cols) #dataframe to hold all data from 
# the remaining rats

for i in range(len(ratno)):
    y = np.array(data.iloc[:, i+1]) #stacked data of ith rat
    z = np.transpose(np.reshape(y[4:], (17, 251))).astype(int) #reshaped with diff columns for
                                                     #diff data
    x1 = np.full((251,1), data.iloc[1,i+1]); x2 = np.full((251,1), data.iloc[0,i+1])  
    s = np.concatenate((x1,x2,z), axis = 1)
    rat1.loc[i+1,0:251] = s #assigning to (i+1)th rat index in the combined data frame
      
summary1 = get_summary(ratno, rat1) #new summary    
subject = np.unique(summary1['subject']) #list of subjects that pass criterion
colorNo = np.full((len(summary1),1),0) #assigning no. to each unique rat
for i in range(len(subject)):
    colorNo[np.where(summary1['subject'] == subject[i])] = i
color = {0:"purple", 1:"slateblue", 2:"teal", 3:"limegreen", 4:"yellow"} 

#%%
#moving average of RLs
t = np.linspace(0,250, 251)
rlOverMeanStd = np.full((nSessions,10), 0.0) #proportion of RLs from each type above mean+std

for i in range(len(ratno)):
    
    RL = rat1.loc[i+1, 'RTH25']+rat1.loc[i+1, 'RTH50']+rat1.loc[i+1, 'RTH500']+rat1.loc[i+1, 'RTM25'
            ]+rat1.loc[i+1, 'RTM50']+rat1.loc[i+1, 'RTM500']+rat1.loc[i+1, 'RTCR']+rat1.loc[i+1, 'RTFA']
    RL = RL[1:(data.iloc[3,i+1])+1]; RL =  RL[RL>0] #compiling RTs of diff trial types
    #removing tail end of empty trials and RTs = 0 (omissions)
    rolling = RL.rolling(window = 20); RLavg = rolling.mean(); RL.plot(color='cyan',label ='RLs'); RLavg.plot(label='RL Smooth');
    plt.scatter(t,rat1.loc[i+1, 'RTH25']+rat1.loc[i+1, 'RTH50']+rat1.loc[i+1, 'RTH500'],color = 'red', label = 'hits' )
    plt.scatter(t,rat1.loc[i+1, 'RTM25']+rat1.loc[i+1, 'RTM50']+rat1.loc[i+1, 'RTM500'],color = 'orange', label = 'miss' )
    plt.scatter(t,rat1.loc[i+1, 'RTCR'],color = 'brown', label = 'cr' )
    plt.scatter(t,rat1.loc[i+1, 'RTFA'],color = 'yellow', label = 'fa' )
    plt.xlim(0, 210); plt.title('rat%d'%summary1.iloc[i,0]) 
    plt.legend(); plt.figure()
    r = (rat1.loc[i+1, 'RTH25']+rat1.loc[i+1, 'RTH50']+rat1.loc[i+1, 'RTH500']).array
    rlOverMeanStd[i,0] =  len(r[np.where(r>np.mean(RLavg)+np.std(RLavg))])/sum(rat1.loc[i+1, 'H25']+rat1.loc[i+1, 'H50']+rat1.loc[i+1, 'H500'])
    r = (rat1.loc[i+1, 'RTM25']+rat1.loc[i+1, 'RTM50']+rat1.loc[i+1, 'RTM500']).array
    rlOverMeanStd[i,1] =  len(r[np.where(r>np.mean(RLavg)+np.std(RLavg))])/sum(rat1.loc[i+1, 'M25']+rat1.loc[i+1, 'M50']+rat1.loc[i+1, 'M500'])
    r = rat1.loc[i+1, 'RTCR'].array;  rlOverMeanStd[i,2] =  len(r[np.where(r>np.mean(RLavg)+np.std(RLavg))])/sum(rat1.loc[i+1, 'CR'])
    r = rat1.loc[i+1, 'RTFA'].array;  rlOverMeanStd[i,3] =  len(r[np.where(r>np.mean(RLavg)+np.std(RLavg))])/sum(rat1.loc[i+1, 'FA'])
    
    r = rat1.loc[i+1, 'RTH25'].array;  rlOverMeanStd[i,4] =  len(r[np.where(r>np.mean(RLavg)+np.std(RLavg))])/sum(rat1.loc[i+1, 'H25'])
    r = rat1.loc[i+1, 'RTH50'].array;  rlOverMeanStd[i,5] =  len(r[np.where(r>np.mean(RLavg)+np.std(RLavg))])/sum(rat1.loc[i+1, 'H50'])
    r = rat1.loc[i+1, 'RTH500'].array;  rlOverMeanStd[i,6] =  len(r[np.where(r>np.mean(RLavg)+np.std(RLavg))])/sum(rat1.loc[i+1, 'H500'])
    r = rat1.loc[i+1, 'RTM25'].array;  rlOverMeanStd[i,7] =  len(r[np.where(r>np.mean(RLavg)+np.std(RLavg))])/sum(rat1.loc[i+1, 'M25'])
    r = rat1.loc[i+1, 'RTM50'].array;  rlOverMeanStd[i,8] =  len(r[np.where(r>np.mean(RLavg)+np.std(RLavg))])/sum(rat1.loc[i+1, 'M50'])
    r = rat1.loc[i+1, 'RTM500'].array;  rlOverMeanStd[i,9] =  len(r[np.where(r>np.mean(RLavg)+np.std(RLavg))])/sum(rat1.loc[i+1, 'M500'])
       
    
fig1, ax1 = plt.subplots(2,2)
for i in range(len(ratno)):
    ax1[0,0].plot(['h','m', 'c', 'f'], rlOverMeanStd[i,0:4], marker = 'o', linestyle = 'dashed', color = color[int(colorNo[i])]) 
    ax1[0,1].plot(['25','50', '500'], rlOverMeanStd[i,4:7], marker = 'o', linestyle = 'dashed', color = color[int(colorNo[i])] ) 
    ax1[1,0].plot(['25','50', '500'], rlOverMeanStd[i,7:10], marker = 'o', linestyle = 'dashed', color = color[int(colorNo[i])] )

ax1[0,0].plot(['h','m', 'c', 'f'], np.mean(rlOverMeanStd[:,0:4], axis = 0), marker = 'o', linestyle = 'dashed', color = 'red' ) 
ax1[0,1].plot(['25','50', '500'], np.mean(rlOverMeanStd[:,4:7], axis = 0), marker = 'o', linestyle = 'dashed', color = 'red' ) 
ax1[1,0].plot(['25','50', '500'], np.mean(rlOverMeanStd[:,7:10], axis = 0), marker = 'o', linestyle = 'dashed', color = 'red' )
ax1[0,1].set_title('hits'); ax1[1,0].set_title('miss')
fig1.text(0.06, 0.5, '% trials w RLs above mean+std', ha='center', va='center', rotation='vertical')
    
#%%
#avgs across groups
summ = summary; ra = rat

rlSig = np.full((len(summ)+1,4), np.nan) #avg rl across signal duraion
rlCSig = np.full((len(summ)+1,4), np.nan); rlWSig = np.full((len(summ)+1,4), np.nan) 
#avg rl vs signal duraion (for correct and wrong response trials)
rlType = np.full((len(summ)+1,4), np.nan) #avg rl across trial types
sig = ['25', '50', '500', '0']; typ = ['h', 'm', 'c', 'f']

fig1, ax1 = plt.subplots(); fig3, ax3 = plt.subplots()
fig2, ax2 = plt.subplots(); fig4, ax4 = plt.subplots()
for i in range(len(summ)):
    rlSig[i,3]=(summ.loc[i+1, 'cr']*summ.loc[i+1, 'rt cr']+ summ.loc[i+1, 'fa']*
                summ.loc[i+1, 'rt fa'])/(summ.loc[i+1, 'cr']+summ.loc[i+1, 'fa'])

    rlSig[i,0]=(summ.loc[i+1, 'h25']*summ.loc[i+1, 'rt h25']+ summ.loc[i+1, 'm25']*
                summ.loc[i+1, 'rt m25'])/(summ.loc[i+1, 'h25']+summ.loc[i+1, 'm25'])
    rlSig[i,1] = (summ.loc[i+1, 'h50']*summ.loc[i+1, 'rt h50']+ summ.loc[i+1, 'm50']*
                summ.loc[i+1, 'rt m50'])/(summ.loc[i+1, 'h50']+summ.loc[i+1, 'm50'])
    rlSig[i,2] = (summ.loc[i+1, 'h500']*summ.loc[i+1, 'rt h500']+ summ.loc[i+1, 'm500']*
                summ.loc[i+1, 'rt m500'])/(summ.loc[i+1, 'h500']+summ.loc[i+1, 'm500'])
    rlCSig[i,0] = summ.loc[i+1, 'rt h25']; rlCSig[i,1] = summ.loc[i+1, 'rt h50']
    rlCSig[i,2] = summ.loc[i+1, 'rt h500']; rlCSig[i,3] = summ.loc[i+1, 'rt cr']
    rlWSig[i,0] = summ.loc[i+1, 'rt m25']; rlWSig[i,1] = summ.loc[i+1, 'rt m50']
    rlWSig[i,2] = summ.loc[i+1, 'rt m500']; rlWSig[i,3] = summ.loc[i+1, 'rt fa']
    
    
    a = (ra.loc[i+1, 'RTH25']+ ra.loc[i+1, 'RTH50']+ ra.loc[i+1, 'RTH500']).to_numpy(dtype = int); 
    a = a[a>0]; rlType[i,0] = np.nanmean(a)
    a = (ra.loc[i+1, 'RTM25']+ ra.loc[i+1, 'RTM50']+ ra.loc[i+1, 'RTM500']).to_numpy(dtype = int); 
    a = a[a>0]; rlType[i,1] = np.nanmean(a)
    a = (ra.loc[i+1, 'RTCR']).to_numpy(dtype = int); a = a[a>0]
    rlType[i,2] = np.nanmean(a)
    a = (ra.loc[i+1, 'RTFA']).to_numpy(dtype = int); a = a[a>0]
    rlType[i,3] = np.nanmean(a)  

    ax1.plot(sig, rlSig[i,:], marker = 'o', linestyle = 'dashed')
    ax2.plot(typ, rlType[i,:], marker = 'o', linestyle = 'dashed')
    ax3.plot(sig, rlCSig[i,:], marker = 'o', linestyle = 'dashed')
    ax4.plot(sig, rlWSig[i,:], marker = 'o', linestyle = 'dashed')
    
rlSig[len(summ), :] = np.nanmean(rlSig[:-1,:], axis =0)
ax1.plot(sig, rlSig[len(summ),:], label = 'all ras', color = 'red'); ax1.set(xlabel='signal',ylabel='avg RL')
ax1.legend(); 
   
rlType[len(summ), :] = np.nanmean(rlType[:-1,:], axis =0)
ax2.plot(typ, rlType[len(summ),:], label = 'all ras', color = 'red'); ax2.set(xlabel='type',ylabel='avg RL')
ax2.legend();

rlCSig[len(summ), :] = np.nanmean(rlCSig[:-1,:], axis =0)
ax3.plot(sig, rlCSig[len(summ),:], label = 'all ras', color = 'red'); ax3.set(xlabel='signal',ylabel='avg RL on correct responses')
ax3.legend(); 
   
rlWSig[len(summ), :] = np.nanmean(rlWSig[:-1,:], axis =0)
ax4.plot(sig, rlWSig[len(summ),:], label = 'all ras', color = 'red'); ax4.set(xlabel='signal',ylabel='avg RL on wrong responses')
ax4.legend(); 
   
#hit and cr rates vs sig duraion
tySig = np.full((len(summ)+1, 4), np.nan); 
fig1, ax1 = plt.subplots()
for i in range(len(summ)):
    tySig[i,0] = summ.loc[i+1, '%h25']
    tySig[i,1] = summ.loc[i+1, '%h50']
    tySig[i,2] = summ.loc[i+1, '%h500']
    tySig[i,3] = summ.loc[i+1, '%cr']
    sno = np.where(subject == summ.loc[i+1,'subject'])[0]  
    ax1.plot(sig, tySig[i,:], marker = 'o', linestyle = 'dashed')

tySig[len(summ), :] = np.nanmean(tySig[:-1,:], axis =0)
ax1.plot(sig, tySig[len(summ),:], label = 'all ras', color = 'red'); ax1.set(xlabel ='signal',ylabel='correct reponse rae')
ax1.legend(); 


#%%
#some plots to explore speed-accuracy trade-off
fig1, ax1 = plt.subplots(2,2, sharex=True, sharey=True)

scatter = ax1[0,0].scatter(100-tySig[:-1,0], rlWSig[:-1,0], c = colorNo); 
ax1[0,0].set_title('25ms'); ax1[0,0].legend(*scatter.legend_elements()) 
scatter = ax1[0,1].scatter(100-tySig[:-1,1], rlWSig[:-1,1], c = colorNo); 
ax1[0,1].set_title('50ms'); ax1[0,0].legend(*scatter.legend_elements()) 
scatter = ax1[1,0].scatter(100-tySig[:-1,2], rlWSig[:-1,2], c = colorNo); 
ax1[1,0].set_title('500ms'); ax1[0,0].legend(*scatter.legend_elements()) 
scatter = ax1[1,1].scatter(100-tySig[:-1,3], rlWSig[:-1,3], c = colorNo);
ax1[1,1].set_title('0ms'); ax1[0,0].legend(*scatter.legend_elements()) 

fig1.text(0.5, 0.04, '% wrong responses', ha='center', va='center')
fig1.text(0.06, 0.5, 'Avg RL on wrong response', ha='center', va='center', rotation='vertical')


fig1, ax1 = plt.subplots(2,2, sharex=True, sharey=True)

scatter = ax1[0,0].scatter(tySig[:-1,0], rlCSig[:-1,0], c = colorNo); 
ax1[0,0].set_title('25ms'); ax1[0,0].legend(*scatter.legend_elements()) 
scatter = ax1[0,1].scatter(tySig[:-1,1], rlCSig[:-1,1], c = colorNo); 
ax1[0,1].set_title('50ms'); ax1[0,0].legend(*scatter.legend_elements()) 
scatter = ax1[1,0].scatter(tySig[:-1,2], rlCSig[:-1,2], c = colorNo); 
ax1[1,0].set_title('500ms'); ax1[0,0].legend(*scatter.legend_elements()) 
scatter = ax1[1,1].scatter(tySig[:-1,3], rlCSig[:-1,3], c = colorNo);
ax1[1,1].set_title('0ms'); 
fig1.text(0.5, 0.04, '% correct responses', ha='center', va='center')
fig1.text(0.06, 0.5, 'Avg RL on correct response', ha='center', va='center', rotation='vertical')

fig1, ax1 = plt.subplots(2,2, sharex=True, sharey=True)

scatter = ax1[0,0].scatter(tySig[:-1,0], rlSig[:-1,0], c = colorNo); 
ax1[0,0].set_title('25ms'); ax1[0,0].legend(*scatter.legend_elements()) 
scatter = ax1[0,1].scatter(tySig[:-1,1], rlSig[:-1,1], c = colorNo); 
ax1[0,1].set_title('50ms'); ax1[0,0].legend(*scatter.legend_elements()) 
scatter = ax1[1,0].scatter(tySig[:-1,2], rlSig[:-1,2], c = colorNo); 
ax1[1,0].set_title('500ms'); ax1[0,0].legend(*scatter.legend_elements()) 
scatter = ax1[1,1].scatter(tySig[:-1,3], rlSig[:-1,3], c = colorNo);
ax1[1,1].set_title('0ms'); 

fig1.text(0.5, 0.04, 'accuracy', ha='center', va='center')
fig1.text(0.06, 0.5, 'Avg RL', ha='center', va='center', rotation='vertical')


#%%
#comparing prev data to this data

#summary1.to_csv('summary_data2.csv', index = False)

summaryData1 = pd.read_csv('summary_data1.csv')#Howe data
summaryData2 = pd.read_csv('summary_data2.csv')#Eryn's data set 1
sig = ['25', '50', '500', '0'];
plt.errorbar(sig, [np.mean(summary1.loc[:,'%h25']), np.mean(summary1.loc[:,'%h50']),
              np.mean(summary1.loc[:,'%h500']), np.mean(summary1.loc[:,'%cr'])],
         yerr=[np.std(summary1.loc[:,'%h25']), np.std(summary1.loc[:,'%h50']),
               np.std(summary1.loc[:,'%h500']), np.std(summary1.loc[:,'%cr'])],
         marker = 'o', linestyle = 'dashed', label = '%correctResponse3')
plt.errorbar(sig, [np.mean(summaryData1.loc[:,'%h25']), np.mean(summaryData1.loc[:,'%h50']),
               np.mean(summaryData1.loc[:,'%h500']), np.mean(summaryData1.loc[:,'%cr'])],
         yerr=[np.std(summaryData1.loc[:,'%h25']), np.std(summaryData1.loc[:,'%h50']),
               np.std(summaryData1.loc[:,'%h500']), np.std(summaryData1.loc[:,'%cr'])],
         marker = 'o', linestyle = 'dashed', label = '%correctResponse1')
plt.errorbar(sig, [np.mean(summaryData2.loc[:,'%h25']), np.mean(summaryData2.loc[:,'%h50']),
               np.mean(summaryData2.loc[:,'%h500']), np.mean(summaryData2.loc[:,'%cr'])],
         yerr=[np.std(summaryData2.loc[:,'%h25']), np.std(summaryData2.loc[:,'%h50']),
               np.std(summaryData2.loc[:,'%h500']), np.std(summaryData2.loc[:,'%cr'])],
         marker = 'o', linestyle = 'dashed', label = '%correctResponse2')

plt.errorbar(sig, [np.mean(summary1.loc[:,'%m25']), np.mean(summary1.loc[:,'%m50']),
               np.mean(summary1.loc[:,'%m500']), np.mean(summary1.loc[:,'%fa'])],
         yerr=[np.std(summary1.loc[:,'%m25']), np.std(summary1.loc[:,'%m50']),
               np.std(summary1.loc[:,'%m500']), np.std(summary1.loc[:,'%fa'])],
         marker = 'o', linestyle = 'dashed', label = '%wrongResponse3')
plt.errorbar(sig, [np.mean(summaryData1.loc[:,'%h25']), np.mean(summaryData1.loc[:,'%m50']),
               np.mean(summaryData1.loc[:,'%m500']), np.mean(summaryData1.loc[:,'%fa'])],
         yerr=[np.std(summaryData1.loc[:,'%m25']), np.std(summaryData1.loc[:,'%m50']),
               np.std(summaryData1.loc[:,'%m500']), np.std(summaryData1.loc[:,'%fa'])],
         marker = 'o', linestyle = 'dashed', label = '%wrongResponse1')
plt.errorbar(sig, [np.mean(summaryData2.loc[:,'%h25']), np.mean(summaryData2.loc[:,'%m50']),
               np.mean(summaryData2.loc[:,'%m500']), np.mean(summaryData2.loc[:,'%fa'])],
         yerr=[np.std(summaryData2.loc[:,'%m25']), np.std(summaryData2.loc[:,'%m50']),
               np.std(summaryData2.loc[:,'%m500']), np.std(summaryData2.loc[:,'%fa'])],
         marker = 'o', linestyle = 'dashed', label = '%wrongResponse2')

plt.ylabel('% correct/ wrong responses'); plt.xlabel('signal duration')
plt.legend(); plt.figure()

plt.errorbar(sig, [np.mean(summary1.loc[:,'rt h25']), np.mean(summary1.loc[:,'rt h50']),
               np.mean(summary1.loc[:,'rt h500']), np.mean(summary1.loc[:,'rt cr'])],
         yerr=[np.std(summary1.loc[:,'rt h25']), np.std(summary1.loc[:,'rt h50']),
               np.std(summary1.loc[:,'rt h500']), np.std(summary1.loc[:,'rt cr'])],
         marker = 'o', linestyle = 'dashed', label = 'correctResponse3')
plt.errorbar(sig, [np.mean(summaryData1.loc[:,'rt h25']/10), np.mean(summaryData1.loc[:,'rt h50']/10),
               np.mean(summaryData1.loc[:,'rt h500']/10), np.mean(summaryData1.loc[:,'rt cr']/10)],
         yerr=[np.std(summaryData1.loc[:,'rt h25']/10), np.std(summaryData1.loc[:,'rt h50']/10),
               np.std(summaryData1.loc[:,'rt h500']/10), np.std(summaryData1.loc[:,'rt cr']/10)],
         marker = 'o', linestyle = 'dashed', label = 'correctResponse1') 
plt.errorbar(sig, [np.mean(summaryData2.loc[:,'rt h25']), np.mean(summaryData2.loc[:,'rt h50']),
               np.mean(summaryData2.loc[:,'rt h500']), np.mean(summaryData2.loc[:,'rt cr'])],
         yerr=[np.std(summaryData2.loc[:,'rt h25']), np.std(summaryData2.loc[:,'rt h50']),
               np.std(summaryData2.loc[:,'rt h500']), np.std(summaryData2.loc[:,'rt cr'])],
         marker = 'o', linestyle = 'dashed', label = 'correctResponse2') 


plt.errorbar(sig, [np.mean(summary1.loc[:,'rt m25']), np.mean(summary1.loc[:,'rt m50']),
               np.mean(summary1.loc[:,'rt m500']), np.mean(summary1.loc[:,'rt fa'])],
         yerr=[np.std(summary1.loc[:,'rt m25']), np.std(summary1.loc[:,'rt m50']),
               np.std(summary1.loc[:,'rt m500']), np.std(summary1.loc[:,'rt fa'])],
         marker = 'o', linestyle = 'dashed', label = 'wrongResponse3')
plt.errorbar(sig, [np.mean(summaryData1.loc[:,'rt m25']/10), np.mean(summaryData1.loc[:,'rt m50']/10),
               np.mean(summaryData1.loc[:,'rt m500']/10), np.mean(summaryData1.loc[:,'rt fa']/10)],
         yerr=[np.std(summaryData1.loc[:,'rt m25']/10), np.std(summaryData1.loc[:,'rt m50']/10),
               np.std(summaryData1.loc[:,'rt m500']/10), np.std(summaryData1.loc[:,'rt fa']/10)],
         marker = 'o', linestyle = 'dashed', label = 'wrongResponse1')
plt.errorbar(sig, [np.mean(summaryData2.loc[:,'rt m25']), np.mean(summaryData2.loc[:,'rt m50']),
               np.mean(summaryData2.loc[:,'rt m500']), np.mean(summaryData2.loc[:,'rt fa'])],
         yerr=[np.std(summaryData2.loc[:,'rt m25']), np.std(summaryData2.loc[:,'rt m50']),
               np.std(summaryData2.loc[:,'rt m500']), np.std(summaryData2.loc[:,'rt fa'])],
         marker = 'o', linestyle = 'dashed', label = 'wrongResponse2')

plt.ylabel('Avg RTs (ms)'); plt.xlabel('signal duration')
plt.legend(); plt.figure()

#%%
#accuracy vs time
n = 5 #no. of blocks
hitTime = np.full((nSessions,n),0.0); crTime = np.full((nSessions,n),0.0);
hit25Time = np.full((nSessions,n),0.0); hit50Time= np.full((nSessions,n),0.0); 
hit500Time = np.full((nSessions,n),0.0)
block = np.linspace(1,n,n)

for i in range(nSessions):
    s = int(summary1.iloc[i,1]/n); #split array into n blocks of size s
    for j in range(n):#accuracy on signal (diff durations) and non-signals for each block
        with suppress(ZeroDivisionError): #ignore zerodivisionerror
            crTime[i,j] = sum(rat1.loc[i+1,'CR'].array[j*s:(j+1)*s])/(sum(rat1.loc[i+1,'CR'].array[j*s:(j+1)*s])+ sum(rat1.loc[i+1,'FA'].array[j*s:(j+1)*s]))
            hitTime[i,j] = (sum(rat1.loc[i+1,'H25'].array[j*s:(j+1)*s])+sum(rat1.loc[i+1,'H50'].array[j*s:(j+1)*s])+sum(rat1.loc[i+1,'H500'].array[j*s:(j+1)*s]))/(
                sum(rat1.loc[i+1,'H25'].array[j*s:(j+1)*s])+sum(rat1.loc[i+1,'H50'].array[j*s:(j+1)*s])+sum(rat1.loc[i+1,'H500'].array[j*s:(j+1)*s])+
                sum(rat1.loc[i+1,'M25'].array[j*s:(j+1)*s])+sum(rat1.loc[i+1,'M50'].array[j*s:(j+1)*s])+sum(rat1.loc[i+1,'M500'].array[j*s:(j+1)*s]))
            hit500Time[i,j] = sum(rat1.loc[i+1,'H500'].array[j*s:(j+1)*s])/(sum(rat1.loc[i+1,'H500'].array[j*s:(j+1)*s])+ sum(rat1.loc[i+1,'M500'].array[j*s:(j+1)*s]))
            hit50Time[i,j] = sum(rat1.loc[i+1,'H50'].array[j*s:(j+1)*s])/(sum(rat1.loc[i+1,'H50'].array[j*s:(j+1)*s])+ sum(rat1.loc[i+1,'M50'].array[j*s:(j+1)*s]))
            hit25Time[i,j] = sum(rat1.loc[i+1,'H25'].array[j*s:(j+1)*s])/(sum(rat1.loc[i+1,'H25'].array[j*s:(j+1)*s])+ sum(rat1.loc[i+1,'M25'].array[j*s:(j+1)*s]))

#plots
fig1, ax1 = plt.subplots() 
ax1.errorbar(block, np.nanmean(hit25Time, axis =0), yerr = np.nanstd(hit25Time, axis =0), marker = 'o', label = 'h25 rate' )    
ax1.errorbar(block, np.nanmean(hit50Time, axis =0), yerr = np.nanstd(hit50Time, axis =0), marker = 'o', label = 'h50 rate' )    
ax1.errorbar(block, np.nanmean(hit500Time, axis =0), yerr = np.nanstd(hit500Time, axis =0), marker = 'o', label = 'h500 rate' )    
ax1.errorbar(block, np.nanmean(hitTime, axis =0), yerr = np.nanstd(hitTime, axis =0), marker = 'o', label = 'h rate')
ax1.errorbar(block, np.nanmean(crTime, axis =0), yerr = np.nanstd(crTime, axis =0), marker = 'o', label = 'cr rate' )  
ax1.legend(); ax1.set(xlabel = 'block', ylabel = 'rate of correct response')

#average within each rats
fig = plt.figure() 
for i in range(len(subject)):
    ax = fig.add_subplot(3,2,i+1)
    plt.errorbar(block, np.nanmean(hit25Time[np.where(colorNo == i)[0], :], axis =0), yerr = np.nanstd(hit25Time[np.where(color == 0)[0], :], axis =0), marker = 'o', label = 'h25 rate' )    
    plt.errorbar(block, np.nanmean(hit50Time[np.where(colorNo == i)[0], :], axis =0), yerr = np.nanstd(hit50Time[np.where(color == i)[0], :], axis =0), marker = 'o', label = 'h50 rate' )    
    plt.errorbar(block, np.nanmean(hit500Time[np.where(colorNo == i)[0], :], axis =0), yerr = np.nanstd(hit500Time[np.where(color == i)[0], :], axis =0), marker = 'o', label = 'h500 rate' )    
    plt.errorbar(block, np.nanmean(hitTime[np.where(colorNo == i)[0], :], axis =0), yerr = np.nanstd(hitTime[np.where(color == i)[0], :], axis =0), marker = 'o', label = 'h rate')
    plt.errorbar(block, np.nanmean(crTime[np.where(colorNo == i)[0], :], axis =0), yerr = np.nanstd(crTime[np.where(color == i)[0], :], axis =0), marker = 'o', label = 'cr rate' )  
lines, labels = ax.get_legend_handles_labels()
fig.legend(lines, labels, loc='lower right'); ax.set(xlabel = 'block', ylabel = 'rate of correct response')

#%%
#sequential effects (for 6 or less rats)
hx = np.full((nSessions,10),0.0); mx = np.full((nSessions,10),0.0) 
cx = np.full((nSessions,10),0.0); fx = np.full((nSessions,10),0.0)
#relative no. of h/m/c/f's preceded by x (can be h,m,c,f, h25/50/500, m25/50/500) for each rat


for i in range(nSessions):
    h = rat1.loc[i+1,'H25']+rat1.loc[i+1,'H50']+rat1.loc[i+1,'H500']; c = rat1.loc[i+1,'CR']
    m = rat1.loc[i+1,'M25']+rat1.loc[i+1,'M50']+rat1.loc[i+1,'M500']; f = rat1.loc[i+1,'FA']
    h25 = rat1.loc[i+1,'H25']; h50 = rat1.loc[i+1,'H50']; h500 = rat1.loc[i+1,'H500']
    m25 = rat1.loc[i+1,'M25']; m50 = rat1.loc[i+1,'M50']; m500 = rat1.loc[i+1,'M500'] 
    x = [h,m,c,f, h25, h50, h500, m25, m50, m500]
    for j in range(len(x)):
        hx[i,j] = len(np.intersect1d(np.where(h[1:] == 1),np.where(x[j][:-1] == 1)))/sum(x[j])    
        mx[i,j] = len(np.intersect1d(np.where(m[1:] == 1),np.where(x[j][:-1] == 1)))/sum(x[j])
        cx[i,j] = len(np.intersect1d(np.where(c[1:] == 1),np.where(x[j][:-1] == 1)))/sum(x[j])
        fx[i,j] = len(np.intersect1d(np.where(f[1:] == 1),np.where(x[j][:-1] == 1)))/sum(x[j])

fig1, ax1 = plt.subplots() 
x = ['h','m', 'c','f','h25', 'h50', 'h500', 'm25', 'm50', 'm500']
ax1.errorbar(x, np.mean(hx, axis = 0), yerr = np.std(hx, axis = 0), marker = 'o', label = 'hit' )
ax1.errorbar(x, np.mean(mx, axis = 0), yerr = np.std(mx, axis = 0), marker = 'o', label = 'miss' )
ax1.errorbar(x, np.mean(cx, axis = 0), yerr = np.std(cx, axis = 0), marker = 'o', label = 'cr' )
ax1.errorbar(x, np.mean(fx, axis = 0), yerr = np.std(fx, axis = 0), marker = 'o', label = 'fa' )
ax1.legend(); ax1.set(xlabel = 'type on previous trial', ylabel = 'prob y precedes x')

#average within each rat
fig = plt.figure() 
x = ['h','m', 'c','f']
for i in range(len(subject)):
    ax = fig.add_subplot(3,2,i+1)
    plt.errorbar(x, np.mean(hx[np.where(colorNo == i)[0]][:,0:4], axis = 0), yerr = np.std(hx[np.where(color == i)[0]][:,0:4], axis = 0), marker = 'o', label = 'hit' )    
    plt.errorbar(x, np.mean(mx[np.where(colorNo == i)[0]][:,0:4], axis = 0), yerr = np.std(mx[np.where(color == i)[0]][:,0:4], axis = 0), marker = 'o', label = 'miss')
    plt.errorbar(x, np.mean(cx[np.where(colorNo == i)[0]][:,0:4], axis = 0), yerr = np.std(cx[np.where(color == i)[0]][:,0:4], axis = 0), marker = 'o', label = 'cr' )
    plt.errorbar(x, np.mean(fx[np.where(colorNo == i)[0]][:,0:4], axis = 0), yerr = np.std(fx[np.where(color == i)[0]][:,0:4], axis = 0), marker = 'o', label = 'fa' )
lines, labels = ax.get_legend_handles_labels()
fig.legend(lines, labels, loc='lower right');
fig.text(0.5, 0.04, 'type on previous trial', ha='center', va='center')
fig.text(0.06, 0.5, 'conditional prob x precedes y', ha='center', va='center', rotation='vertical')

fig = plt.figure() 
x = ['h25', 'h50', 'h500', 'm25', 'm50', 'm500']
for i in range(len(subject)):
    ax = fig.add_subplot(3,2,i+1)
    plt.errorbar(x, np.mean(hx[np.where(colorNo == i)[0]][:,4:10], axis = 0), yerr = np.std(hx[np.where(color == i)[0]][:,4:10], axis = 0), marker = 'o', label = 'hit' )    
    plt.errorbar(x, np.mean(mx[np.where(colorNo == i)[0]][:,4:10], axis = 0), yerr = np.std(mx[np.where(color == i)[0]][:,4:10], axis = 0), marker = 'o', label = 'miss')
    plt.errorbar(x, np.mean(cx[np.where(colorNo == i)[0]][:,4:10], axis = 0), yerr = np.std(cx[np.where(color == i)[0]][:,4:10], axis = 0), marker = 'o', label = 'cr' )
    plt.errorbar(x, np.mean(fx[np.where(colorNo == i)[0]][:,4:10], axis = 0), yerr = np.std(fx[np.where(color == i)[0]][:,4:10], axis = 0), marker = 'o', label = 'fa' )
lines, labels = ax.get_legend_handles_labels()
fig.legend(lines, labels, loc='lower right'); 
fig.text(0.5, 0.04, 'type on previous trial', ha='center', va='center')
fig.text(0.06, 0.5, 'conditional prob x precedes y', ha='center', va='center', rotation='vertical')

#sequential effects in responses (response to signal/ non-signal)
sx = np.full((nSessions,2),0.0); nsx = np.full((nSessions,2),0.0)

#relative no. of s/ns responsess preceded by x (can be s/ns response) for each rat
for i in range(nSessions):
    s = rat1.loc[i+1, 'H25']+rat1.loc[i+1, 'H50']+rat1.loc[i+1, 'H500']+rat1.loc[i+1, 'FA']
    ns = rat1.loc[i+1, 'M25']+rat1.loc[i+1, 'M50']+rat1.loc[i+1, 'M500']+rat1.loc[i+1, 'CR']
    x = [s,ns]
    for j in range(len(x)):
        sx[i,j] = len(np.intersect1d(np.where(s[1:] == 1),np.where(x[j][:-1] == 1)))/sum(x[j])
        nsx[i,j] = len(np.intersect1d(np.where(ns[1:] == 1),np.where(x[j][:-1] == 1)))/sum(x[j])
        
fig1, ax1 = plt.subplots() 
x = ['signal', 'no signal']
ax1.errorbar(x, np.mean(sx, axis = 0), yerr = np.std(sx, axis = 0), marker = 'o', label = 'signal response' )
ax1.errorbar(x, np.mean(nsx, axis = 0), yerr = np.std(nsx, axis = 0), marker = 'o', label = 'no signal response' )
ax1.legend(); ax1.set(xlabel = 'response on previous trial', ylabel = 'prob y precedes x')

#avg for each rat
fig = plt.figure() 
for i in range(len(subject)):
    ax = fig.add_subplot(3,2,i+1)
    plt.errorbar(x, np.mean(sx[np.where(colorNo == i)[0]], axis = 0), yerr = np.std(sx[np.where(color == i)[0]], axis = 0), marker = 'o', label = 'signal response' )    
    plt.errorbar(x, np.mean(nsx[np.where(colorNo == i)[0]], axis = 0), yerr = np.std(nsx[np.where(color == i)[0]], axis = 0), marker = 'o', label = 'no signal response' )
lines, labels = ax.get_legend_handles_labels()
fig.legend(lines, labels, loc='lower right'); ax.set(xlabel = 'response on previous trial', ylabel = 'prob y precedes x')

#%%
#response bias vs time
n = 5 #no. of blocks
bias = np.full((nSessions,n),0.0); bias25 = np.full((nSessions,n),0.0)
bias50 = np.full((nSessions,n),0.0); bias500 = np.full((nSessions,n),0.0)
block = np.linspace(1,n,n)

for i in range(nSessions):
    s = int(summary1.iloc[i,1]/n); #split array into n blocks of size s
    for j in range(n):#accuracy on signal (diff durations) and non-signals for each block
        with suppress(ZeroDivisionError): #ignore zerodivisionerror
            c = rat1.loc[i+1,'CR'].array[j*s:(j+1)*s]; f = rat1.loc[i+1,'FA'].array[j*s:(j+1)*s] 
            h = rat1.loc[i+1,'H25'].array[j*s:(j+1)*s]+rat1.loc[i+1,'H50'].array[j*s:(j+1)*s]+rat1.loc[i+1,'H500'].array[j*s:(j+1)*s]
            m = rat1.loc[i+1,'M25'].array[j*s:(j+1)*s]+rat1.loc[i+1,'M50'].array[j*s:(j+1)*s]+rat1.loc[i+1,'M500'].array[j*s:(j+1)*s]
            bias[i,j] = (sum(h)+sum(f))/(sum(h)+sum(c)+sum(m)+sum(f))
            bias25[i,j] = (sum(rat1.loc[i+1,'H25'])+sum(f))/(
            sum(rat1.loc[i+1,'H25'])+sum(c)+sum(rat1.loc[i+1,'M25'])+sum(f))
            bias50[i,j] = (sum(rat1.loc[i+1,'H50'])+sum(f))/(
            sum(rat1.loc[i+1,'H50'])+sum(c)+sum(rat1.loc[i+1,'M50'])+sum(f))
            bias500[i,j] = (sum(rat1.loc[i+1,'H500'])+sum(f))/(
            sum(rat1.loc[i+1,'H500'])+sum(c)+sum(rat1.loc[i+1,'M500'])+sum(f))

#plots
fig1, ax1 = plt.subplots() 
ax1.errorbar(block, np.nanmean(bias, axis =0), yerr = np.nanstd(bias, axis =0), marker = 'o', label = 'all signals' )    
ax1.errorbar(block, np.nanmean(bias25, axis =0), yerr = np.nanstd(bias, axis =0), marker = 'o', label = '25' )    
ax1.errorbar(block, np.nanmean(bias50, axis =0), yerr = np.nanstd(bias, axis =0), marker = 'o', label = '50' )    
ax1.errorbar(block, np.nanmean(bias500, axis =0), yerr = np.nanstd(bias, axis =0), marker = 'o', label = '500' )    
ax1.legend(); ax1.set(xlabel = 'block', ylabel = 'response bias (to signal)')


#%%

#dataframe for regression analysis

#'rat','trial no.', 'RL', 'laggedRL', 'signalDuration', 'response','laggedResponse', 
#'hit', 'miss', 'cr', 'fa', 'incongHit', 'congHit'
r = rat

X = np.full((len(r), 15), np.nan) #array to hold the the different predictors

ratno = (np.linspace(1,int(len(r)/251) , int(len(r)/251))).astype(int) #index of rats
time = (np.linspace(0,250, 251)).astype(int) #data for the 251 trials for each rat
idx = pd.MultiIndex.from_product([ratno, time], names = ('session no.', 'trial no.'))#heirarchical indexing in dataframe

X[:,0] = (idx.get_level_values('session no.')).to_numpy()
X[:,1] = (idx.get_level_values('trial no.')).to_numpy()
X[:,2] = r['rat'].array
X[:,3]= (r['RTH25']).to_numpy(dtype = int) + (r['RTH50']).to_numpy(
    dtype = int)+(r['RTH500']).to_numpy(dtype = int) + (r['RTM25']).to_numpy(
    dtype = int)+(r['RTM50']).to_numpy(dtype = int)+(r['RTM500']).to_numpy(
    dtype = int)+(r['RTFA']).to_numpy(dtype = int)+(r['RTCR']).to_numpy(dtype = int)
a = X[:,3]; a[a == 0] = np.nan; X[:, 3] = a
X[1:,4] = X[:-1,3]; X[:,3] = X[:,3]*0.001; X[:,4] = X[:,4]*0.001 #RLs and lagged RLs

X[:,5] =  25*(r['H25']).to_numpy(dtype = int) + 50*(r['H50']).to_numpy(
    dtype = int)+ 500*(r['H500']).to_numpy(dtype = int) + 25* (r['M25']).to_numpy(
    dtype = int)+ 50*(r['M50']).to_numpy(dtype = int)+ 500*(r['M500']).to_numpy(
    dtype = int)+(r['FA']).to_numpy(dtype = int)+(r['CR']).to_numpy(dtype = int) 
a = X[:,5]; a[a == 0] = np.nan; X[:, 5] = a; X[:, 5] = np.where(a == 1, 0, a) 
X[:,5] = X[:,5]*0.001#signal durat1ion
X[1:,6] = X[:-1,5] #lagged sig duration

X[:,7] = 1*(r['H25']).to_numpy(dtype = int) + 1*(r['H50']).to_numpy(
    dtype = int)+ 1*(r['H500']).to_numpy(dtype = int) + 2*(r['M25']).to_numpy(
    dtype = int)+ 2*(r['M50']).to_numpy(dtype = int)+ 2*(r['M500']).to_numpy(
    dtype = int)+ 1*(r['FA']).to_numpy(dtype = int)+ 2*(r['CR']).to_numpy(dtype = int)
a = X[:,7]; a[a == 0] = np.nan; X[:, 7] = a; X[:,7] = np.where(a == 2, 0, a)
X[1:,8] = X[:-1,7] #response and lagged response

X[:,9] = (r['H25']).to_numpy(dtype = int) + (r['H50']).to_numpy(
    dtype = int)+(r['H500']).to_numpy(dtype = int)
X[:,10] = (r['M25']).to_numpy(dtype = int) + (r['M50']).to_numpy(
    dtype = int)+(r['M500']).to_numpy(dtype = int)
X[:,11] = (r['CR']).to_numpy(dtype = int)
X[:,12] = (r['FA']).to_numpy(dtype = int) #hit, miss, cr, fa

a = X[:,9]; b = np.full((len(r),),0)
b[1:] = a[1:] + a[:-1]; b = np.where(b == 2, 1, 0);
X[:,13] = a - b #incong hit
X[:,14] = b #cong hit

Y =   r['date'].array #date of session


#dataSDT = X[:, [5,7]];  #data for SDT data analysis
#np.savetxt("dataSDTFull.csv", dataSDT)
Y = Y[~np.isnan(X).any(axis=1)]
X = X[~np.isnan(X).any(axis=1)] #removing nans

ratDF = pd.DataFrame({'session': X[:,0], 'trial': X[:,1], 'rat': X[:,2], 
                      'RL': X[:,3],'laggedRL': X[:,4],'signalDuration': X[:,5], 
                      'laggedsignalDuration': X[:,6],'response': X[:,7], 
                      'laggedResponse': X[:,8], 'hit': X[:,9], 'miss': X[:,10], 
                      'cr': X[:,11], 'fa': X[:,12], 'incongHit': X[:,13], 
                      'congHit': X[:,14]})

#ratDF.to_csv('ratDF3_Full.csv', index=False)
#%%
#logistic model

def sigmoid(x,param):
    c = param[0]; m = param[1]
    temp = -(m*x+c)
    y = 1/(1+np.exp(temp))
    return y

import statsmodels.api as sm

l = int(len(rat)/251)
paramArr = np.full((l, 2), 0.0)

for i in range(l):
    r = rat.loc[i+1]    
    
    dat = np.full((251,2), np.nan)
    dat[:,0] = 25*(r['H25']).to_numpy(dtype = float) + 50*(r['H50']).to_numpy(
    dtype = float)+ 500*(r['H500']).to_numpy(dtype = float) + 25* (r['M25']).to_numpy(
    dtype = float)+ 50*(r['M50']).to_numpy(dtype = float)+ 500*(r['M500']).to_numpy(
    dtype = float)+(r['FA']).to_numpy(dtype = float)+(r['CR']).to_numpy(dtype = float) 
    a = dat[:,0]; a[a == 0] = np.nan; dat[:,0] = a; dat[:,0] = np.where(a == 1, 0, a);
    dat[:,0] = dat[:,0]*0.001 
    
    dat[:,1] = 1*(r['H25']).to_numpy(dtype =float) + 1*(r['H50']).to_numpy(
    dtype = float)+ 1*(r['H500']).to_numpy(dtype =float) + 2*(r['M25']).to_numpy(
    dtype = float)+ 2*(r['M50']).to_numpy(dtype = float)+ 2*(r['M500']).to_numpy(
    dtype = float)+ 1*(r['FA']).to_numpy(dtype = float)+ 2*(r['CR']).to_numpy(dtype = float)
    a = dat[:,1]; a[a == 0] = np.nan; dat[:,1]= a; dat[:,1]= np.where(a == 2, 0, a); 
    
    dat = dat[~np.isnan(dat).any(axis=1)] #removing nans
    
    x = dat[:,0]; y = dat[:,1]
    x = sm.add_constant(x)
    model = sm.Logit(y, x); fits = model.fit()
    paramArr[i,:] =  fits.params

subj = summary['subject'].array.astype(int)
uniqSubj = np.unique(subj)
date = summary['date'].array.astype('U')

for i in range(len(uniqSubj)):
    s = np.where(subj == uniqSubj[i])[0]
    for j in range(len(s)):
        xval = np.linspace(-1,1,50)
        yval = sigmoid(xval, paramArr[s[j]])
        plt.plot(xval, yval, label = date[s[j]])
    plt.legend(); plt.ylabel('prob of responding 1'); plt.xlabel('signal duration')
    plt.figure()


