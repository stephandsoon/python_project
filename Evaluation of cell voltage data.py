#Evaluation of cell voltage data
#%%
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from Functions import *


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


bms_u = df.loc[:, df.columns.str.contains('bms.u')]
cell_voltage = bms_u.iloc[:,0:bms_u.shape[1]-3]

cvd = calculate_cell_voltage_deviation(df)

#Cell voltage as matrix
[cv_rows, cv_columns] = cell_voltage.shape
cols_cell = list(cell_voltage.columns)
time = df[['t']]

time_in_hours = df[['t']]/3600

#Use median voltage as reference and calculate voltage deviation
median_voltage = cell_voltage.iloc[:, 0:cell_voltage.shape[1]].median(axis=1)
cell_voltage_deviation = cell_voltage.sub(median_voltage, axis = 0).add_prefix('delta_')
vd_num = pd.DataFrame(cell_voltage_deviation).to_numpy()


#Create dataframe of all the cell except of cell 3
threshold = 0.0085
#all_cell_voltage_deviation = cell_voltage_deviation['bms.u0'].append(cell_voltage_deviation['bms.u1']).append(cell_voltage_deviation['bms.u2']).append(cell_voltage_deviation['bms.u3']).append(cell_voltage_deviation['bms.u4']).append(cell_voltage_deviation['bms.u5']).append(cell_voltage_deviation['bms.u6']).append(cell_voltage_deviation['bms.u7']).append(cell_voltage_deviation['bms.u8']).append(cell_voltage_deviation['bms.u9']).append(cell_voltage_deviation['bms.u10']).append(cell_voltage_deviation['bms.u11'])
#all_cell_under_threshold = pd.DataFrame(all_cell_voltage_deviation).to_numpy()
#mask_all_cells = abs(all_cell_under_threshold) < -threshold
#mask_all_cells_bi = np.array(mask_all_cells, dtype=int)

#Evaluate Risc
risc_data = df.loc[:, df.columns.str.contains('.risc')]
print('ISC in :', risc_data.loc[:, risc_data.iloc[1] < 1000000].columns)

#Drop all positive voltage deviations
negative_cell_voltage_deviation = cell_voltage.sub(median_voltage, axis = 0)
nvd_num = pd.DataFrame(negative_cell_voltage_deviation).to_numpy()
mask_neg = nvd_num > 0
nvd_num[mask_neg] = 0

#Calculate number or times cell voltage deviation surpasses a certain threshold
cnd_num_vd = np.zeros(cv_columns, dtype=int)
mask_vd = abs(vd_num) > threshold
mask_vd_bi = np.array(mask_vd, dtype=int)
cnd_num_vd = abs(mask_vd_bi).sum(axis=0)


cnd_num_nvd = np.zeros(cv_columns, dtype=int)
mask_nvd = abs(nvd_num) > threshold
mask_nvd_bi = np.array(mask_nvd, dtype=int)
cnd_num_nvd = abs(mask_nvd_bi).sum(axis=0)


#Calculate sum and min of (negative) voltage deviation
sum_vd = cell_voltage_deviation.sum(axis=0)
sum_nvd = negative_cell_voltage_deviation.sum(axis=0)

max_vd = abs(cell_voltage_deviation).max(axis=0)
max_nvd = abs(negative_cell_voltage_deviation).max(axis=0)


#Extract balancing data
balancing_data = df.loc[:, df.columns.str.contains('.bal')]
bal_cycles = balancing_data.sum(axis=0)/270

#Showing the voltage difference to cell with lowest voltage
real_voltage = df.loc[:, df.columns.str.contains('.ucell')]
real_median_voltage = real_voltage.iloc[:, 0:real_voltage.shape[1]].median(axis=1)
real_cell_voltage_deviation = real_voltage.sub(real_median_voltage, axis = 0)
min_voltage = real_voltage.min(axis=1)

voltage_surplus = real_voltage.subtract(min_voltage, axis=0)
surplus_margin = voltage_surplus - 0.015

voltage_bias = cell_voltage.subtract(cell_voltage['bms.u3'], axis=0)#.drop(index=0)
measured_min = cell_voltage.min(axis=1)
measured_voltage_surplus = cell_voltage.subtract(measured_min, axis=0)

offset = 0.01
balancing_data_offset = balancing_data.copy(deep=True)
for i in range(balancing_data_offset.shape[1]):
    balancing_data_offset.iloc[:,i] = balancing_data_offset.iloc[:,i]+i*offset

#Use Functions-Skript
data = pd.concat([cell_voltage_deviation, balancing_data, df['bms.soc.soc'], df['bms.i'], df['t'], df['category']], axis=1)
list_of_features, list_of_charging_times = get_list_of_features_transposed(data, ave_values=5)

charging_times = get_list_of_charging_times(df)

tag = 1
lower_border = time['t'].iloc[0] #((tag-1)*(3600*24)-100) #3600 #0 # 54000
upper_border = time['t'].iloc[-1] #((tag)*(3600*24)+1200)#

#Batterie Status über der Zeit
plt.figure(figsize=(16,7))
plt.plot(df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], df['category'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)])
plt.title('Lastprofil')
#plt.vlines((tag*3600*24), ymin = 'wait', ymax = 'sleeping', colors='Magenta', linestyle='dotted', label='Start new day \n-> driving')
plt.xlabel('Zeit in s')
plt.ylabel('Status')
plt.show()


# time_in_hours = df[['t']]/3600

# plt.figure(figsize=(16,7))
# plt.plot(time_in_hours, df['category'])
# plt.title('Lastprofil des Arbeitstages')
# plt.xticks(np.arange(0,25,1))
# plt.xlim(0, 24)
# #plt.vlines((tag*3600*24), ymin = 'wait', ymax = 'sleeping', colors='Magenta', linestyle='dotted', label='Start new day \n-> driving')
# plt.xlabel('Zeit in h')
# plt.ylabel('Status')
# plt.grid(axis='x')
# plt.savefig('D:\Hoetger_Stephan\Grafiken\Profile/Arbeitstag.svg')
# plt.show()


#%%
#Balancing Status über der Zeit
plt.figure(figsize=(16,7))
plt.plot(df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], balancing_data_offset.loc[(df['t'] > lower_border) & (df['t'] < upper_border)], linestyle='dotted')
plt.xlabel('Zeit in s')
plt.ylabel('Balancing Status mit leichtem Offset')
plt.ylim(0,1.3)
plt.title('Balancing Status')
plt.legend(['cell0', 'cell1', 'cell2', 'cell3', 'cell4', 'cell5', 'cell6', 'cell7', 'cell8', 'cell9', 'cell10', 'cell11'], loc='best')
plt.show()

#%%
#Darstellung der durchgeführten Balancing Cycles
plt.figure(figsize=(30,7))
plt.bar(bal_cycles.index, bal_cycles)
plt.xlabel('Zelle')
plt.ylabel('Anzahl der durchgeführten Balancing Cycles')
plt.title('Durchgeführte Balancing Cycles')


# %%
print('Zellspannung in Abhängigkeit der Zeit')

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



#%%
lower_border = 0 # 54000
upper_border = time['t'].iloc[-1]
plt.figure(figsize=(16,7))
plt.plot(df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], df['cell0.ucell'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)],
        df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], df['cell1.ucell'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)],
        df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], df['cell2.ucell'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)],
        df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], df['cell3.ucell'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], 'r',
        df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], df['cell4.ucell'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)],
        df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], df['cell5.ucell'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)],
        df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], df['cell6.ucell'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)],
        df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], df['cell7.ucell'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)],
        df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], df['cell8.ucell'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)],
        df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], df['cell9.ucell'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)],
        df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], df['cell10.ucell'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)],
        df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], df['cell11.ucell'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], linestyle='dotted')
plt.xlabel('Zeit in s')
plt.ylabel('Ucell in V')
plt.title('Real cell voltage')
plt.legend(['cell0', 'cell1', 'cell2', 'cell3', 'cell4', 'cell5', 'cell6', 'cell7', 'cell8', 'cell9', 'cell10', 'cell11'], loc='best')
plt.grid(True)
#plt.savefig('D:\Hoetger_Stephan\Grafiken\Cell_Balancing/cv_distribution.svg')
plt.show()

#start_u = df.iloc[1][['cell0.ucell', 'cell1.ucell', 'cell2.ucell','cell3.ucell', 'cell4.ucell', 'cell5.ucell','cell6.ucell', 'cell7.ucell', 'cell8.ucell', 'cell9.ucell', 'cell10.ucell', 'cell11.ucell']]
#end_u = df.iloc[4032][['cell0.ucell', 'cell1.ucell', 'cell2.ucell','cell3.ucell', 'cell4.ucell', 'cell5.ucell','cell6.ucell', 'cell7.ucell', 'cell8.ucell', 'cell9.ucell', 'cell10.ucell', 'cell11.ucell']]
#diff_u = start_u.subtract(end_u, axis=0)
#print('Zellspannungsabnahme:')
#print(diff_u)

#%%
# für 1 Zyklus
bal = df.loc[df['category']=='balancing']

start_soc = bal.iloc[1][['cell0.soc.soc', 'cell1.soc.soc', 'cell2.soc.soc','cell3.soc.soc', 'cell4.soc.soc', 'cell5.soc.soc','cell6.soc.soc', 'cell7.soc.soc', 'cell8.soc.soc', 'cell9.soc.soc', 'cell10.soc.soc', 'cell11.soc.soc']]
end_soc = bal.iloc[-1][['cell0.soc.soc', 'cell1.soc.soc', 'cell2.soc.soc','cell3.soc.soc', 'cell4.soc.soc', 'cell5.soc.soc','cell6.soc.soc', 'cell7.soc.soc', 'cell8.soc.soc', 'cell9.soc.soc', 'cell10.soc.soc', 'cell11.soc.soc']]
diff_soc = start_soc.subtract(end_soc, axis=0)



# start_soc = df.iloc[1][['cell0.soc.soc', 'cell1.soc.soc', 'cell2.soc.soc','cell3.soc.soc', 'cell4.soc.soc', 'cell5.soc.soc','cell6.soc.soc', 'cell7.soc.soc', 'cell8.soc.soc', 'cell9.soc.soc', 'cell10.soc.soc', 'cell11.soc.soc']]
# end_soc = df.iloc[4032][['cell0.soc.soc', 'cell1.soc.soc', 'cell2.soc.soc','cell3.soc.soc', 'cell4.soc.soc', 'cell5.soc.soc','cell6.soc.soc', 'cell7.soc.soc', 'cell8.soc.soc', 'cell9.soc.soc', 'cell10.soc.soc', 'cell11.soc.soc']]
# diff_soc = start_soc.subtract(end_soc, axis=0)

#diff_soc.axes = [['cell1', 'cell2', 'cell3', 'cell4', 'cell5', 'cell6', 'cell7', 'cell8', 'cell9', 'cell10', 'cell1']]
#print('SOC-Abnahme:')
#print(diff_soc)

print('SOC-Abnahme der verschiedenen Zellen beim Balancing')
plt.figure(figsize=(16,7))
plt.plot(diff_soc)
plt.xlabel('Zelle')
plt.ylabel('SOC-Abnahme')
plt.title('SOC change during balancing')
plt.show()

#%%

plt.figure(figsize=(16,7))
plt.plot(time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],real_voltage.loc[(time['t'] > lower_border) & (time['t'] < upper_border)])
plt.xlabel('Zeit in s')
plt.ylabel('Reale Zellspannung in V')
plt.legend(real_voltage.columns, loc='best')
plt.title('Real cell voltage')
plt.show()


plt.figure(figsize=(16,7))
plt.plot(time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],voltage_surplus.loc[(time['t'] > lower_border) & (time['t'] < upper_border)])
plt.xlabel('Zeit in s')
plt.ylabel('Spannungsüberschuss in V')
plt.hlines(0.015, lower_border, upper_border, colors='Magenta', linestyle='dotted', label='Threshold')
plt.legend(voltage_surplus.columns, loc='best')
plt.title('Real cell voltage surplus')
plt.show()



plt.figure(figsize=(16,7))
plt.plot(time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],voltage_bias.loc[(time['t'] > lower_border) & (time['t'] < upper_border)])
plt.xlabel('Zeit in s')
plt.ylabel('Spannungs bias zur Zelle 3 in V')
plt.hlines(0.015, lower_border, upper_border, colors='Magenta', linestyle='dotted', label='Threshold')
plt.legend(voltage_bias.columns, loc='best')
plt.title('Measured cell voltage difference towards cell 3')
plt.show()

#%%
plt.figure(figsize=(16,7))
plt.plot(time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],measured_voltage_surplus.loc[(time['t'] > lower_border) & (time['t'] < upper_border)])
plt.xlabel('Zeit in s')
plt.ylabel('Spannungs bias zur geringsten gemessenen Spannung')
plt.hlines(0.015, lower_border, upper_border, colors='Magenta', linestyle='dotted', label='Threshold')
plt.legend(voltage_bias.columns, loc='best')
plt.title('Measured cell voltage difference towards lowest cell voltage')
plt.show()


#%%
#Gemessene Zellspannungsabweichung
plt.figure(figsize=(16,7))
plt.plot(time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage_deviation['bms.u0'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage_deviation['bms.u1'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage_deviation['bms.u2'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage_deviation['bms.u3'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], 'r--',
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage_deviation['bms.u4'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage_deviation['bms.u5'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage_deviation['bms.u6'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage_deviation['bms.u7'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage_deviation['bms.u8'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage_deviation['bms.u9'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage_deviation['bms.u10'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage_deviation['bms.u11'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], linestyle='dotted')
plt.xlabel('Zeit')
plt.ylabel('Delta_u in V')
plt.grid(True)
plt.show()

# %%
#Nur negative Zellspannungsabweichung
plt.figure(figsize=(16,7))
plt.plot(time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], negative_cell_voltage_deviation['bms.u0'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], negative_cell_voltage_deviation['bms.u1'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], negative_cell_voltage_deviation['bms.u2'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], negative_cell_voltage_deviation['bms.u3'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], 'r--',
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], negative_cell_voltage_deviation['bms.u4'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], negative_cell_voltage_deviation['bms.u5'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], negative_cell_voltage_deviation['bms.u6'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], negative_cell_voltage_deviation['bms.u7'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], negative_cell_voltage_deviation['bms.u8'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], negative_cell_voltage_deviation['bms.u9'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], negative_cell_voltage_deviation['bms.u10'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], negative_cell_voltage_deviation['bms.u11'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], linestyle='dotted')
plt.xlabel('Zeit')
plt.ylabel('Delta_u in V')
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(16,7))
plt.plot(time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage_deviation['bms.u3'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], 'r--',
        time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], cell_voltage_deviation['bms.u0'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)])
plt.xlabel('Zeit')
plt.ylabel('Delta_u in V')
plt.grid(True)
plt.show()# %%

# %%
#Plotte den Strom
plt.figure(figsize=(16,7))
plt.plot(time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)], df['i'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)])
plt.xlabel('Zeit in s')
plt.ylabel('I in A')
plt.grid(True)
plt.show()


#%%
#Stelle realen SOC aller Zellen dar
SOC = df[['cell0.soc.soc', 'cell1.soc.soc', 'cell2.soc.soc', 'cell3.soc.soc', 'cell4.soc.soc', 'cell5.soc.soc', 'cell6.soc.soc', 'cell7.soc.soc', 'cell8.soc.soc', 'cell9.soc.soc', 'cell10.soc.soc', 'cell11.soc.soc']]
plt.figure(figsize=(16,7))
plt.plot(df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], SOC.loc[(df['t'] > lower_border) & (df['t'] < upper_border)], linestyle='dotted')
plt.xlabel('Zeit in s')
plt.ylabel('Realer SOC')
plt.title('Real SOC during balancing process')
plt.legend(['cell0', 'cell1', 'cell2', 'cell3', 'cell4', 'cell5', 'cell6', 'cell7', 'cell8', 'cell9', 'cell10', 'cell11'], loc='best')
plt.grid(True)
#plt.savefig('D:\Hoetger_Stephan\Grafiken\Cell_Balancing/cv_distribution.svg')
plt.show()

#%%
#SOC mit aktuellem Status
fig, ax = plt.subplots()
ax.plot(df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], SOC.loc[(df['t'] > lower_border) & (df['t'] < upper_border)], linestyle='dotted')
ax.set_xlabel('Zeit in s')
ax.set_ylabel('Realer SOC')
plt.legend(['cell0', 'cell1', 'cell2', 'cell3', 'cell4', 'cell5', 'cell6', 'cell7', 'cell8', 'cell9', 'cell10', 'cell11'], loc='upper left')
ax2= ax.twinx()
ax2.plot(df['t'], df['category'], linestyle='dotted')
ax2.set_ylabel('Status')
#plt.grid(True)
fig.set_figheight(8)
fig.set_figwidth(10)
plt.title('SOC and Status')
plt.show()
# %%
lower_border1 = 0#15000# 5000 # #lower_border
upper_border1 = df['t'].iloc[-1] #25000 #8000 # #upper_border
plt.figure(figsize=(16,7))
plt.plot(time['t'].loc[(time['t'] > lower_border1) & (time['t'] < upper_border1)],measured_voltage_surplus.loc[(time['t'] > lower_border1) & (time['t'] < upper_border1)])
plt.xlabel('Zeit in s')
plt.ylabel('Spannungüberschuss in V')
plt.hlines(0.015, lower_border1, upper_border1, colors='Magenta', linestyle='dotted', label='Threshold')
plt.legend(voltage_bias.columns, loc='best')
plt.title('Measured cell voltage difference towards lowest cell voltage')
plt.show()

#%%
lower_border1 = lower_border
upper_border1 = upper_border
soc = df[['cell0.soc.soc', 'cell1.soc.soc', 'cell2.soc.soc', 'cell3.soc.soc', 'cell4.soc.soc', 'cell5.soc.soc', 'cell6.soc.soc', 'cell7.soc.soc', 'cell8.soc.soc', 'cell9.soc.soc', 'cell10.soc.soc', 'cell11.soc.soc']]
soc_min = soc.min(axis=1)
soc_diff = soc.subtract(soc_min, axis=0)

plt.figure(figsize=(16,7))
plt.plot(df['t'].loc[(df['t'] > lower_border1) & (df['t'] < upper_border1)], soc_diff.loc[(df['t'] > lower_border1) & (df['t'] < upper_border1)], linestyle='dotted')
plt.xlabel('Zeit in s')
plt.ylabel('SOC surplus')
plt.title('Real SOC surplus')
plt.legend(soc_diff.columns, loc='best')
plt.grid(True)
#plt.savefig('D:\Hoetger_Stephan\Grafiken\Cell_Balancing/cv_distribution.svg')
plt.show()
# %%
#Reale Zellspannung einer normalen Zelle, einer mit geringer Kapazität und einer defekten Zelle
#lower_border = 172000
plt.figure(figsize=(16,7))
plt.plot(df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], df['cell1.ocv.ocv'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], 'g:',
        df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], df['cell3.ocv.ocv'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)],'-',
        df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], df['cell11.ocv.ocv'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], 'r:')
plt.xlabel('Zeit in s')
plt.ylabel('Uocv in V')
plt.title('Zelle mit geringer Kapazität (grün), Zelle 3 mir Risc = 100 Ohm (blau) und normale Zelle (rot)')
plt.legend(['Defekte Zelle', 'Normale Zelle', 'Zelle mit geringer Kapazität'], loc='upper left')
#plt.grid(True)
#plt.savefig('D:\Hoetger_Stephan\Grafiken\Cell_Balancing/cv_distribution.svg')
plt.show()

# %%
#Plot cell voltage deviation
lower_border = 0
upper_border = df['t'].iloc[-1] #300000

plt.figure(figsize=(16,7))
plt.plot(time['t'].loc[(time['t'] > lower_border) & (time['t'] < upper_border)],cell_voltage_deviation.loc[(time['t'] > lower_border) & (time['t'] < upper_border)], linestyle ='dashed')
plt.xlabel('Zeit in s')
plt.ylabel('Gemessene Spannungsabweichung in V')
#plt.hlines(0.015, lower_border, upper_border, colors='Magenta', linestyle='dotted', label='Threshold')
plt.legend(cell_voltage_deviation.columns, loc='best')
plt.title('Measured cell voltage deviation towards median value')
plt.show()
# %%
plt.figure(figsize=(16,7))
plt.plot(df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)],real_voltage.loc[(df['t'] > lower_border) & (df['t'] < upper_border)])
plt.xlabel('Zeit in s')
plt.ylabel('Reale Zellpannung in V')
#plt.hlines(0.015, lower_border, upper_border, colors='Magenta', linestyle='dotted', label='Threshold')
plt.legend(cell_voltage_deviation.columns, loc='best')
plt.title('Real cell voltage')
plt.show()
# %%
plt.figure(figsize=(16,7))
plt.plot(df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)],real_cell_voltage_deviation.loc[(df['t'] > lower_border) & (df['t'] < upper_border)])
plt.xlabel('Zeit in s')
plt.ylabel('Reale Zellpannungsabweichung in V')
#plt.hlines(0.015, lower_border, upper_border, colors='Magenta', linestyle='dotted', label='Threshold')
plt.legend(real_cell_voltage_deviation.columns, loc='best')
plt.title('Reale Zellspannungsabweichung zum Medianwert')
plt.show()
# %%

#Cell voltage deviation with current status
lower_border = 0 # 54000
upper_border = df['t'].iloc[df.shape[0]-1] #time.size
height = 10
width = 12
Zellen = ['cell0.ucell', 'cell1.ucell', 'cell2.ucell', 'cell4.ucell', 'cell5.ucell', 'cell6.ucell', 'cell7.ucell', 'cell8.ucell', 'cell9.ucell', 'cell10.ucell', 'cell11.ucell']

fig, ax = plt.subplots()
ax.plot(df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], real_cell_voltage_deviation[Zellen].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], linestyle='dotted', label='Normale Zellen')
ax.plot(df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], real_cell_voltage_deviation['cell3.ucell'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], 'r--', label='PTRC')
ax.set_xlabel('Zeit in s')
ax.set_ylabel('Zellspannungsabweichung in V')
plt.legend(loc='best')
for i in range(0, 1): #len(list_of_charging_times)):
    plt.vlines(list_of_charging_times[i], real_cell_voltage_deviation.iloc[:,0:12].min().min(), real_cell_voltage_deviation.iloc[:,0:12].max().max(), colors='Magenta', linestyle='dotted')
ax2= ax.twinx()
ax2.plot(df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], df['category'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], 'y', linestyle='dotted')
ax2.set_ylabel('Status')
#plt.grid(True)
fig.set_figheight(height)
fig.set_figwidth(width)
plt.title('Zellspannungsabweichungen')
plt.show()

#%%
#Cell voltage deviation with current status
lower_border = 0
upper_border = df['t'].iloc[-1] #time.size
height = 10
width = 12
Zellen = ['cell0.ucell', 'cell15.ucell', 'cell47.ucell', 'cell28.ucell', 'cell32.ucell'] #real_cell_voltage_deviation.columns#['cell0.ucell', 'cell1.ucell', 'cell2.ucell', 'cell4.ucell', 'cell5.ucell', 'cell6.ucell', 'cell7.ucell', 'cell8.ucell', 'cell9.ucell', 'cell10.ucell', 'cell11.ucell']

plt.figure(figsize=(width,height))
plt.plot(df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], real_cell_voltage_deviation[Zellen].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], linestyle='dotted', label='Normale Zellen')
plt.plot(df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], real_cell_voltage_deviation['cell3.ucell'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], 'r--', label='PTRC')
plt.xlabel('Zeit in s')
plt.ylabel('Zellspannungsabweichung in V')
plt.legend(Zellen, loc='best')
#for i in range(0, len(list_of_charging_times)):
#    plt.vlines(list_of_charging_times[i], real_cell_voltage_deviation.iloc[:,0:12].min().min(), real_cell_voltage_deviation.iloc[:,0:12].max().max(), colors='Magenta', linestyle='dotted')
plt.title('Zellspannungsabweichungen')
plt.show()

# %%
lower_border = 400000#0
upper_border = 460000#df['t'].iloc[-1] #time.size
height = 10
width = 12
Zellen = ['cell0.bal', 'cell15.bal', 'cell47.bal', 'cell28.bal', 'cell32.bal'] #real_cell_voltage_deviation.columns#['cell0.ucell', 'cell1.ucell', 'cell2.ucell', 'cell4.ucell', 'cell5.ucell', 'cell6.ucell', 'cell7.ucell', 'cell8.ucell', 'cell9.ucell', 'cell10.ucell', 'cell11.ucell']

plt.figure(figsize=(width,height))
plt.plot(df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], balancing_data_offset[Zellen].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], linestyle='dotted', label='Normale Zellen')
#plt.plot(df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], real_cell_voltage_deviation['cell3.ucell'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], 'r--', label='PTRC')
plt.xlabel('Zeit in s')
plt.ylabel('Zellspannungsabweichung in V')
plt.legend(Zellen, loc='best')
#for i in range(0, len(list_of_charging_times)):
#    plt.vlines(list_of_charging_times[i], real_cell_voltage_deviation.iloc[:,0:12].min().min(), real_cell_voltage_deviation.iloc[:,0:12].max().max(), colors='Magenta', linestyle='dotted')
plt.title('Zellspannungsabweichungen')
plt.show()
# %%
BMS_Zellen = ['bms.u0', 'bms.u15', 'bms.u47', 'bms.u28', 'bms.u32']
# %%

for i in range(0, len(charging_times)):

    lower_border = charging_times[i][1,0]-300
    upper_border = charging_times[i][1,1]+100
    time_bc = df['t'].loc[(df['t'] > lower_border) & (df['t'] < lower_border+55)]
    time_ac = df['t'].loc[(df['t'] > upper_border-55) & (df['t'] < upper_border)]
    time_b_and_a_c = pd.concat([time_bc, time_ac], axis=0)
    real_cell_voltage_deviation_bc = real_cell_voltage_deviation.loc[(df['t'] > lower_border) & (df['t'] < lower_border+55)]
    real_cell_voltage_deviation_ac = real_cell_voltage_deviation.loc[(df['t'] > upper_border-55) & (df['t'] < upper_border)]
    real_cell_voltage_deviation_b_and_a_c = pd.concat([real_cell_voltage_deviation_bc, real_cell_voltage_deviation_ac], axis=0)


    height = 10
    width = 15
    #Zellen = ['cell0.ucell', 'cell1.ucell', 'cell2.ucell', 'cell4.ucell', 'cell5.ucell', 'cell6.ucell', 'cell7.ucell', 'cell8.ucell', 'cell9.ucell', 'cell10.ucell', 'cell11.ucell']

    plt.figure(figsize=(width,height))
    plt.plot(time_b_and_a_c, real_cell_voltage_deviation_b_and_a_c[Zellen], linestyle='dotted', label='Normale Zellen')
    plt.xlabel('Zeit in s')
    plt.ylabel('Zellspannungsabweichung in V')
    plt.legend(Zellen, loc='best')
    # for i in range(i,i+1):
    #     plt.vlines(charging_times[i][1,0], real_cell_voltage_deviation.iloc[:,0:12].min().min(), real_cell_voltage_deviation.iloc[:,0:12].max().max(), colors='Magenta', linestyle='dotted')
    #     plt.vlines(charging_times[i][1,1], real_cell_voltage_deviation.iloc[:,0:12].min().min(), real_cell_voltage_deviation.iloc[:,0:12].max().max(), colors='Magenta', linestyle='dotted')

    plt.title('Zellspannungsabweichungen')
    plt.show()
# %%
lower_border = 0
upper_border = df['t'].iloc[-1] #time.size
height = 10
width = 15
Zellen = ['cell0.ucell', 'cell1.ucell', 'cell2.ucell', 'cell3.ucell', 'cell4.ucell', 'cell5.ucell', 'cell6.ucell', 'cell7.ucell', 'cell8.ucell', 'cell9.ucell', 'cell10.ucell', 'cell11.ucell']

plt.figure(figsize=(width,height))
plt.plot(df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], real_cell_voltage_deviation.loc[(df['t'] > lower_border) & (df['t'] < upper_border)], linestyle='dotted', label='Normale Zellen')
plt.xlabel('Zeit in s')
plt.ylabel('Zellspannungsabweichung in V')
plt.legend(real_cell_voltage_deviation.columns, loc='upper left')
for i in range(0,len(charging_times)):
    plt.vlines(charging_times[i][1,0], real_cell_voltage_deviation[Zellen].min().min(), real_cell_voltage_deviation[Zellen].max().max(), colors='Magenta', linestyle='dotted')
    plt.vlines(charging_times[i][1,1], real_cell_voltage_deviation[Zellen].min().min(), real_cell_voltage_deviation[Zellen].max().max(), colors='Magenta', linestyle='dotted')
#plt.vlines(284406, real_cell_voltage_deviation[Zellen].min().min(), real_cell_voltage_deviation[Zellen].max().max(), colors='Cyan', linestyle='dashed')
plt.vlines(6*3600*24, real_cell_voltage_deviation[Zellen].min().min(), real_cell_voltage_deviation[Zellen].max().max(), colors='Cyan', linestyle='dashed')
plt.title('Zellspannungsabweichungen')
plt.show()
# %%


lower_border = 0
upper_border = df['t'].iloc[-1] #time.size
height = 10
width = 15
Zellen = ['cell0.ucell', 'cell1.ucell', 'cell2.ucell', 'cell3.ucell', 'cell4.ucell', 'cell5.ucell']

plt.figure(figsize=(width,height))
plt.plot(df['t'].loc[(df['t'] > lower_border) & (df['t'] < upper_border)], real_cell_voltage_deviation.loc[(df['t'] > lower_border) & (df['t'] < upper_border)], linestyle='dotted', label='Normale Zellen')
plt.xlabel('Zeit in s')
plt.ylabel('Zellspannungsabweichung in V')
#plt.legend(real_cell_voltage_deviation.columns, loc='upper left')
for i in range(0,len(charging_times)):
    plt.vlines(charging_times[i][1,0], real_cell_voltage_deviation.min().min(), real_cell_voltage_deviation.max().max(), colors='Magenta', linestyle='dotted')
    plt.vlines(charging_times[i][1,1], real_cell_voltage_deviation.min().min(), real_cell_voltage_deviation.max().max(), colors='Magenta', linestyle='dotted')
#plt.vlines(284406, real_cell_voltage_deviation[Zellen].min().min(), real_cell_voltage_deviation[Zellen].max().max(), colors='Cyan', linestyle='dashed')
plt.vlines(96*3600, real_cell_voltage_deviation.min().min(), real_cell_voltage_deviation[Zellen].max().max(), colors='Cyan', linestyle='dashed')
plt.vlines(82.49*3600, real_cell_voltage_deviation.min().min(), real_cell_voltage_deviation[Zellen].max().max(), colors='Blue', linestyle='dashed')
plt.title('Zellspannungsabweichungen')
plt.show()
# %%
