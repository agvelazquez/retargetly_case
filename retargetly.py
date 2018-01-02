# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 17:18:33 2017

@author: Usr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.metrics         import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.externals       import joblib 
import time 
import pickle

#==============================================================================
# data = pd.read_csv(r'busqueda_rely_final.csv', usecols = [1], index_col = False, header = None, dtype = { 0:'category', 1:'str', 2:'str', 3:'str'})
# 
# #%% Sites separation
# sites = data[2].str.split('/',expand = True)
# sites = sites[0]
# sites = sites.astype(str)
# sites.to_csv('sites.csv')
# 
# #%% Intereses separation
# data = data[1].str.split(',',expand = True)
# data.to_csv('interes.csv')
#==============================================================================
#%%
data = pd.read_csv(r'busqueda_rely_final.csv', usecols = [2], index_col = False, header = None, dtype = { 0:'category'})

sites = pd.read_csv(r'sites.csv', header = None)
sites = sites.drop([0], axis = 1)
unique_sites = sites[1].value_counts()
unique_sites = unique_sites.to_frame()

#%% pareto graph
unique_sites['cum_sum'] = unique_sites[1].cumsum()
unique_sites['cum_perc']  = 100*unique_sites.cum_sum/unique_sites[1].sum()

#%% graph
plt.figure()
plt.title('Most popular sites', y=1.05, size=15)
g = sns.barplot(unique_sites.index.values[0:40], unique_sites['cum_perc'][0:40], palette="BuGn_d")
plt.xticks(rotation=75)
g.set_ylabel('Cumulated Percentage', fontsize=10)
plt.tight_layout()

#%% graph mobile and web

device = {'device': [514768*100/1000000, 485232*100/1000000]}
device = pd.DataFrame(device, index = ['webm','web'])

plt.figure()
plt.title('Device used', y=1.05, size=15)
a = sns.barplot(x = 'device', y= device.index.values , data = device, palette="BuGn_d")
a.set_xlabel('Percentage')

#%% device in each site
data = pd.concat([data,sites], axis = 1)
heraldo = data[data[1] == 'mangafox.la']

mobile_web = {'elheraldo.co':   [66139*100/83191, 17052*100/83191], 
              'mangafox.la':    [75*100/56717, 56642*100/56717],
              'tvnotas.com.mx': [40827*100/43916, 3089*100/43916],
              'vagas.com.br':   [23840*100/42089, 18249*100/42089],
               'allcalidad.com':[9236*100/39289, 30053*100/39289],
               'Factor': ['webm','web']}

device = pd.DataFrame(mobile_web)

tidy = (
        device.set_index('Factor')
      .stack()  # un-pivots the data 
      .reset_index()  # moves all data out of the index
      .rename(columns={'level_1': 'Variable', 0: 'Value'})
)

plt.figure()
fig, ax1 = plt.subplots(figsize=(10, 10))
plt.title('Device used', y=1.05, size=15)
sns.barplot(x = 'Value', y= 'Factor', hue='Variable', data = tidy, palette="Set2" ,ax = ax1)
ax1.set_xlabel('Percentage')
ax1.set_ylabel('Device')
sns.despine(fig)
#%%
df = pd.DataFrame({
    'Factor': ['Growth', 'Value'],
    'Weight': [0.10, 0.20],
    'Variance': [0.15, 0.35]})
fig, ax1 = plt.subplots(figsize=(10, 10))
tidy = (
    df.set_index('Factor')
      .stack()  # un-pivots the data 
      .reset_index()  # moves all data out of the index
      .rename(columns={'level_1': 'Variable', 0: 'Value'})
)
sns.barplot(x='Factor', y='Value', hue='Variable', data=tidy, ax=ax1)
sns.despine(fig)

#%% interes analisis

sites = pd.read_csv('sites.csv', header = None, index_col = False)
heraldo =  sites[sites[1] == 'elheraldo.co']
mangafox = sites[sites[1] == 'mangafox.la']
tvnotas =  sites[sites[1] == 'tvnotas.com.mx']
vagas = sites[sites[1] == 'vagas.com.br']
allcalidad = sites[sites[1] == 'allcalidad.com']

indexes = pd.concat([heraldo[0], mangafox[0], tvnotas[0], vagas[0], allcalidad[0]], axis = 0)
indexes.to_csv('indexes.csv')

data = pd.read_csv(r'busqueda_rely_final.csv', index_col = False, header = None, dtype = { 0:'category'})
indexes = pd.read_csv(r'indexes.csv',index_col = False, header = None)
indexes =indexes.drop([1], axis = 1)
sites= sites[1].rename(index = 'str', columns={'str':'sites'})
data = pd.concat([data, sites], axis = 1)
del sites
data.to_csv('data.csv')

#%% analysis by site (change the site)
data = pd.read_csv(r'data.csv')
heraldo = data[data['str'] == 'allcalidad.com']
del data

heraldo = heraldo['1'].str.split(',',expand = True)

for i in range(0,137):
    heraldo[i] = heraldo[i].str.extract('(\d+)')
heraldo.to_csv('allcalidad_interes.csv')

heraldo = heraldo.fillna(value = 0)

heraldo = heraldo.apply(pd.to_numeric)
unique_audience = np.unique(heraldo[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136]].values)
unique_audience = pd.Series(unique_audience)

b = heraldo[0]
for i in range(1,137):
    b = b.append(heraldo[i]).reset_index(drop=True)
    
records_per_audience = b.value_counts()
records_per_audience.to_csv('allcalidad_audience.csv')

#%%
heraldo = pd.read_csv(r'heraldo_audience.csv', header = None)
heraldo_15 = heraldo.iloc[1:16]
heraldo_15 = heraldo_15.rename(columns = {0:'Audience', 1:'Number of users'})
heraldo_15.Audience = pd.to_numeric(heraldo_15.Audience, downcast = 'integer')
heraldo_15['Audience']=heraldo_15['Audience'].astype('category')
del heraldo
heraldo_15 = heraldo_15[0:3]

mangafox = pd.read_csv(r'mangafox_audience.csv', header = None)
mangafox_15 = mangafox.iloc[1:16]
mangafox_15 = mangafox_15.rename(columns = {0:'Audience', 1:'Number of users'})
mangafox_15.Audience = pd.to_numeric(mangafox_15.Audience, downcast = 'integer')
mangafox_15['Audience']=mangafox_15['Audience'].astype('category')
del mangafox
mangafox_15 = mangafox_15[0:3]

tvnotas = pd.read_csv(r'tvnotas_audience.csv', header = None)
tvnotas_15 = tvnotas.iloc[1:16]
tvnotas_15 = tvnotas_15.rename(columns = {0:'Audience', 1:'Number of users'})
tvnotas_15.Audience = pd.to_numeric(tvnotas_15.Audience, downcast = 'integer')
tvnotas_15['Audience']=tvnotas_15['Audience'].astype('category')
del tvnotas
tvnotas_15 = tvnotas_15[0:3]

vagas = pd.read_csv(r'vagas_audience.csv', header = None)
vagas_15 = vagas.iloc[1:16]
vagas_15 = vagas_15.rename(columns = {0:'Audience', 1:'Number of users'})
vagas_15.Audience = pd.to_numeric(vagas_15.Audience, downcast = 'integer')
vagas_15['Audience']=vagas_15['Audience'].astype('category')
del vagas
vagas_15 = vagas_15.iloc[0:3]

allcalidad = pd.read_csv(r'allcalidad_audience.csv', header = None)
allcalidad_15 = allcalidad.iloc[1:16]
allcalidad_15 = allcalidad_15.rename(columns = {0:'Audience', 1:'Number of users'})
allcalidad_15.Audience = pd.to_numeric(allcalidad_15.Audience, downcast = 'integer')
allcalidad_15['Audience']=allcalidad_15['Audience'].astype('category')
del allcalidad
allcalidad_15 = allcalidad_15.iloc[0:3]

fig, ax1 = plt.subplots(figsize=(10, 10))
plt.title('Audience allcalidad.com', y=1.05, size=15)
sns.barplot(x = 'Number of users', y= 'Audience', order=allcalidad_15['Audience'], data = allcalidad_15, palette="Set2" ,ax = ax1)
ax1.set_xlabel('Number of users')
ax1.set_ylabel('Audience')

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(8, 6), sharex=True)
sns.barplot(x = 'Number of users', y= 'Audience', order=heraldo_15['Audience'], data = heraldo_15, palette="Set2" ,ax = ax1)
sns.barplot(x = 'Number of users', y= 'Audience', order=mangafox_15['Audience'], data = mangafox_15, palette="Set2" ,ax = ax2)
sns.barplot(x = 'Number of users', y= 'Audience', order=tvnotas_15['Audience'], data = tvnotas_15, palette="Set2" ,ax = ax3)
sns.barplot(x = 'Number of users', y= 'Audience', order=vagas_15['Audience'], data = vagas_15, palette="Set2" ,ax = ax4)
sns.barplot(x = 'Number of users', y= 'Audience', order=allcalidad_15['Audience'], data = allcalidad_15, palette="Set2" ,ax = ax5)
sns.despine(bottom=True)
ax1.set_xlabel("")
ax1.set_ylabel("heraldo.co")
ax2.set_xlabel("")
ax2.set_ylabel("mangafox.la")
ax3.set_xlabel("")
ax3.set_ylabel("tvnotas.com.mx")
ax4.set_xlabel("")
ax4.set_ylabel("vagas.com.br")
ax5.set_xlabel("Number of users")
ax5.set_ylabel("allcalidad.com")
plt.tight_layout(h_pad=1)
ax1.set_title('Top audiences per site')

#==============================================================================
# #%% three audiences for each site
# 
# string1 = ['heraldo.co','heraldo.co','heraldo.co','heraldo.co','heraldo.co',
#            'heraldo.co','heraldo.co','heraldo.co','heraldo.co','heraldo.co',
#            'heraldo.co','heraldo.co','heraldo.co','heraldo.co','heraldo.co',
#            'heraldo.co']
# string2 = ['mangafox.la','mangafox.la','mangafox.la','mangafox.la','mangafox.la',
#            'mangafox.la','mangafox.la','mangafox.la','mangafox.la','mangafox.la',
#            'mangafox.la','mangafox.la','mangafox.la','mangafox.la','mangafox.la',
#            'mangafox.la']
# 
# string3 = ['tvnotas.com.mx','tvnotas.com.mx','tvnotas.com.mx','tvnotas.com.mx',
#            'tvnotas.com.mx','tvnotas.com.mx','tvnotas.com.mx','tvnotas.com.mx',
#            'tvnotas.com.mx','tvnotas.com.mx','tvnotas.com.mx','tvnotas.com.mx',
#            'tvnotas.com.mx','tvnotas.com.mx','tvnotas.com.mx','tvnotas.com.mx']
# 
# string4 = ['vagas.com.br','vagas.com.br','vagas.com.br','vagas.com.br','vagas.com.br',
#            'vagas.com.br','vagas.com.br','vagas.com.br','vagas.com.br','vagas.com.br',
#            'vagas.com.br','vagas.com.br','vagas.com.br','vagas.com.br','vagas.com.br',
#            'vagas.com.br']
# 
# string5 = ['allcalidad.com', 'allcalidad.com', 'allcalidad.com', 'allcalidad.com',
#            'allcalidad.com', 'allcalidad.com', 'allcalidad.com', 'allcalidad.com',
#            'allcalidad.com', 'allcalidad.com', 'allcalidad.com', 'allcalidad.com',
#            'allcalidad.com', 'allcalidad.com', 'allcalidad.com', 'allcalidad.com']
# 
# string1 = pd.Series(string1)
# heraldo_15 = pd.concat([heraldo_15, string1], axis = 1, ignore_index = True, join = 'inner')
# heraldo_15 = heraldo_15.iloc[0:3]
# 
# string5 = pd.Series(string5)
# allcalidad_15 = pd.concat([allcalidad_15, string5], axis = 1, ignore_index = True, join = 'inner')
# allcalidad_15 = allcalidad_15.iloc[0:3]
# 
# ensemble = pd.concat([heraldo_15,mangafox_15,tvnotas_15,vagas_15,allcalidad_15], axis = 0)
# ensemble[0]=ensemble[0].astype('category')
# 
# ensemble = ensemble.rename(columns = {0:'Audience', 1:'Number of users', 2:'Site'})
# 
# 
# g = sns.factorplot(x='Audience', y='Number of users', hue='Site', data=ensemble,
#                    size=6, kind="bar", palette="muted")
#==============================================================================
#%% Pie chart 
labels = ['elheraldo.co', 'mangafox.la', 'tvnotas.com.mx', 'vagas.com.br', 'allcalidad.com', 'Otros']
values = [83191, 56717, 43916, 42089, 39289,734798]
explode=(0.05, 0.05, 0.05, 0.05, 0.05, 0)

fig, ax1 = plt.subplots()
plt.pie(values, labels=labels, autopct='%1.1f%%', shadow=False, explode = explode, startangle=90)
ax1.axis('equal')  
ax1.set_title('Accounts for the visits', size = 16)

#%% Most popular audiences
heraldo = pd.read_csv(r'heraldo_audience.csv', header = None)
heraldo= heraldo.iloc[1:-1]
heraldo= heraldo.rename(columns = {0:'Audience', 1:'Number of users'})
heraldo.Audience = pd.to_numeric(heraldo.Audience, downcast = 'integer')
heraldo['Audience']=heraldo['Audience'].astype('category')
heraldo.set_index('Audience', inplace = True)

mangafox = pd.read_csv(r'mangafox_audience.csv', header = None)
mangafox = mangafox.iloc[1:-1]
mangafox = mangafox.rename(columns = {0:'Audience', 1:'Number of users'})
mangafox.Audience = pd.to_numeric(mangafox.Audience, downcast = 'integer')
mangafox['Audience']=mangafox['Audience'].astype('category')
mangafox.set_index('Audience', inplace = True)

tvnotas = pd.read_csv(r'tvnotas_audience.csv', header = None)
tvnotas= tvnotas.iloc[1:-1]
tvnotas= tvnotas.rename(columns = {0:'Audience', 1:'Number of users'})
tvnotas.Audience = pd.to_numeric(tvnotas.Audience, downcast = 'integer')
tvnotas['Audience']=tvnotas['Audience'].astype('category')
tvnotas.set_index('Audience', inplace = True)

vagas = pd.read_csv(r'vagas_audience.csv', header = None)
vagas = vagas.iloc[1:-1]
vagas = vagas.rename(columns = {0:'Audience', 1:'Number of users'})
vagas.Audience = pd.to_numeric(vagas.Audience, downcast = 'integer')
vagas['Audience']=vagas['Audience'].astype('category')
vagas.set_index('Audience', inplace = True)

allcalidad = pd.read_csv(r'allcalidad_audience.csv', header = None)
allcalidad = allcalidad.iloc[1:-1]
allcalidad = allcalidad.rename(columns = {0:'Audience', 1:'Number of users'})
allcalidad.Audience = pd.to_numeric(allcalidad.Audience, downcast = 'integer')
allcalidad['Audience']=allcalidad['Audience'].astype('category')
allcalidad.set_index('Audience', inplace = True)

ensemble = pd.concat([heraldo, tvnotas, mangafox, vagas, allcalidad], axis = 1, join = 'outer')
ensemble = ensemble.fillna(value = 0)
ensemble = ensemble.apply(pd.to_numeric)
audience_sum = ensemble.sum(axis =1).sort_values(ascending = False)
audience_sum = pd.to_numeric(audience_sum, downcast = 'integer')
audience_30000 = audience_sum[audience_sum > 30000]
audience_30000 = audience_30000.rename('a')
audience_30000 = audience_30000.reset_index()

fig, ax1 = plt.subplots(figsize=(10, 10))
plt.title('Most popular audiences', y=1.05, size=15)
sns.barplot(x = 'a', y= 'Audience', data = audience_30000, order = audience_30000.Audience, palette="hls" ,ax = ax1)
ax1.set_xlabel('Number of users')
ax1.set_ylabel('Audience')

audience_30000['cum_sum']   = audience_30000['a'].cumsum()
audience_30000['cum_perc']  = 100*audience_30000.cum_sum/audience_30000['a'].sum()
audience_30000_drop = audience_30000.drop(['a','cum_sum'], axis = 1)
audience_30000_drop['Audience'] = audience_30000_drop['Audience'].astype(str)
audience_30000_drop.set_index('Audience', inplace = True)

#%% graph
plt.figure()
plt.title('Most popular audiences', y=1.05, size=15)
g = sns.barplot(x = audience_30000_drop.index.values, y = audience_30000_drop.cum_perc,  data = audience_30000_drop, palette="GnBu_d", x_order=audience_30000_drop.index.values)
plt.xticks(rotation=75)
g.set_ylabel('Cumulated Percentage', fontsize=10)
plt.tight_layout()
#%% graph

g = heraldo_unique.groupby('section').sum()
g.sort_values(['sum'], ascending=False, inplace=True)

 
plt.figure()
plt.title('Most popular sections', y=1.05, size=15)
graph = sns.barplot(x = 'sum', y = 'section',  data = g.iloc[0:10], palette="GnBu_d")
plt.xticks(rotation=75)
graph.set_ylabel('Sections', fontsize=10)
graph.set_xlabel('Users', fontsize=10)
plt.tight_layout()

