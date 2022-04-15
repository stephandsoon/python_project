#Evaluation of cell voltage data
#%%
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt



#%%
#Import selected data

#df = pd.read_pickle('C:/Users/steph/Documents/Coding/Python/Virtual_Env_Private_Project/testcase_01.pickle')
df = pd.read_csv('C:/Users/steph/Documents/Coding/Python/Virtual_Env_Private_Project/example.csv')

df.loc[df['category'] == 'balancing', 'category'] = 'sleeping'
cols = list(df.columns)
# df = df.loc[df['t'] > 3*3600*24-200]
#df = df.loc[df['t'] <  0.03 * 3600]
# df[['t']] = df[['t']] - df[['t']].iloc[0]


#%%
#Batterie Status über der Zeit

time = df[['t']]


lower_border = time['t'].iloc[0] #((tag-1)*(3600*24)-100) #3600 #0 # 54000
upper_border = time['t'].iloc[-1] #((tag)*(3600*24)+1200)#


plt.figure(figsize=(16,7))
plt.plot(df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], df['category'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)])
plt.title('Lastprofil')
#plt.vlines((tag*3600*24), ymin = 'wait', ymax = 'sleeping', colors='Magenta', linestyle='dotted', label='Start new day \n-> driving')
plt.xlabel('Zeit in s')
plt.ylabel('Status')
plt.show()

#%%

print('Zellspannung in Abhängigkeit der Zeit')


bms_u = df.loc[:, df.columns.str.contains('bms.u')]
cell_voltage = bms_u.iloc[:,0:bms_u.shape[1]-3]


plt.figure(figsize=(16,7))
plt.plot(time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage['bms.u0'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage['bms.u1'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage['bms.u2'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage['bms.u3'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], 'r',
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage['bms.u4'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage['bms.u5'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage['bms.u6'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage['bms.u7'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage['bms.u8'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage['bms.u9'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage['bms.u10'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage['bms.u11'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], linestyle='dotted')
plt.xlabel('Zeit in s')
plt.ylabel('U in V')
plt.title('Measured cell voltage')
plt.legend(['bms.u0', 'bms.u1', 'bms.u2', 'bms.u3', 'bms.u4', 'bms.u5', 'bms.u6', 'bms.u7', 'bms.u8', 'bms.u9', 'bms.u10', 'bms.u11'], loc='best')
plt.grid(True)
plt.show()
