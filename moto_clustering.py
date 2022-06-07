#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import requests
import io

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import sklearn
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale

import sklearn.metrics as sm
from sklearn import datasets
from sklearn.metrics import confusion_matrix,classification_report

from sklearn import preprocessing 


from sklearn.metrics import silhouette_score

r = requests.get('https://docs.google.com/spreadsheets/d/11NZe3svUyQrhyysvXEbySTu-6qkwcrsD3AFCIETZe1o/export?format=csv&gid=0')
data = r.content

raw_data = pd.read_csv(io.StringIO(data.decode('utf-8')))

df_moto = pd.DataFrame(raw_data)

print(df_moto['drive_train'].value_counts())
print(df_moto['wheel_type'].value_counts())

df_moto_1 = pd.get_dummies(df_moto, prefix = '', prefix_sep = '', columns = ['drive_train'])
df_moto_1 = pd.get_dummies(df_moto_1, prefix = '', prefix_sep = '', columns = ['wheel_type'])

df_moto_1["fuel_efficiency"]=df_moto_1["fuel_efficiency"].str.replace(',','.')
df_moto_1["fuel_tank"]=df_moto_1["fuel_tank"].str.replace(',','.')
df_moto_1["pw_ratio"]=df_moto_1["pw_ratio"].str.replace(',','.')

df_moto_1["fuel_efficiency"] = df_moto_1["fuel_efficiency"].astype(str).astype(float)
df_moto_1["fuel_tank"] = df_moto_1["fuel_tank"].astype(str).astype(float)
df_moto_1["pw_ratio"] = df_moto_1["pw_ratio"].astype(str).astype(float)

df_moto_1.describe().T

corr = pd.DataFrame(df_moto_1, columns = ['displacement', 'power_hp','torque_nm','net_weight',
                                         'gross_weight', 'load_capacity','seat_height','fuel_efficiency',
                                         'fuel_tank','tank_range', 'wheel_base','ground_clear','front_wheel',
                                          'rear_wheel','front_travel','rear_travel','hip_angle','knee_angle',
                                          'pw_ratio','chain','shaft','cast','spoke'])
matrix = corr.corr()
sns.heatmap(matrix, annot=True, annot_kws={"size": 6})
plt.savefig('matrix.png')

df_moto_2 = df_moto_1.loc[:, ['power_hp','net_weight','load_capacity','seat_height','fuel_tank','tank_range', 
                              'wheel_base','ground_clear','front_travel','rear_wheel','rear_travel','hip_angle','chain']].values

scaler = StandardScaler()
df_moto_scaled_features = scaler.fit_transform(df_moto_2)

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12)).fit(df_moto_scaled_features)
visualizer.show()
plt.savefig('elbow.png')

clustering = KMeans(n_clusters = 4, random_state = 5)
clustering.fit(df_moto_scaled_features)
predict=clustering.predict(df_moto_scaled_features)

plt.scatter(df_moto_scaled_features[:,0], 
            df_moto_scaled_features[:,1])
plt.scatter(clustering.cluster_centers_[:, 0], 
            clustering.cluster_centers_[:, 1], 
            s=200,                             
            c='red')  
plt.title('K-means clustering')
plt.xlabel('power_hp')
plt.ylabel('net_weight')
plt.show()
plt.savefig('k-means.png')

print(Counter(clustering.labels_))
clustering.inertia_

df_moto_1['clusters'] = pd.Series(predict, index=df_moto_1.index)

group = df_moto_1.groupby(["clusters", "category"])["clusters"].count()
group

df_moto_1['new_category'] = None
df_moto_1.loc[df_moto_1.clusters == 0, 'new_category'] = 'sport'
df_moto_1.loc[df_moto_1.clusters == 1, 'new_category'] = 'adventure'
df_moto_1.loc[df_moto_1.clusters == 2, 'new_category'] = 'naked'
df_moto_1.loc[df_moto_1.clusters == 3, 'new_category'] = 'touring'

df_moto_1['comparison_column'] = np.where(df_moto_1["category"] == df_moto_1["new_category"], True, False)
df_moto_2 = df_moto_1.loc[(df_moto_1['category'] == 'adventure') & (df_moto_1['comparison_column'] == False)]

tukey = pairwise_tukeyhsd(endog=df_moto_1['power_hp'],
                          groups=df_moto_1['new_category'],
                          alpha=0.05)
print(tukey) 

tukey2 = pairwise_tukeyhsd(endog=df_moto_1['net_weight'],
                          groups=df_moto_1['new_category'],
                          alpha=0.05)
print(tukey2) 

tukey3 = pairwise_tukeyhsd(endog=df_moto_1['load_capacity'],
                          groups=df_moto_1['new_category'],
                          alpha=0.05)
print(tukey3) 

tukey4 = pairwise_tukeyhsd(endog=df_moto_1['seat_height'],
                          groups=df_moto_1['new_category'],
                          alpha=0.05)
print(tukey4) 

tukey5 = pairwise_tukeyhsd(endog=df_moto_1['fuel_tank'],
                          groups=df_moto_1['new_category'],
                          alpha=0.05)
print(tukey5) 

tukey6 = pairwise_tukeyhsd(endog=df_moto_1['tank_range'],
                          groups=df_moto_1['new_category'],
                          alpha=0.05)
print(tukey6) 

tukey7 = pairwise_tukeyhsd(endog=df_moto_1['wheel_base'],
                          groups=df_moto_1['new_category'],
                          alpha=0.05)
print(tukey7) 
tukey8 = pairwise_tukeyhsd(endog=df_moto_1['ground_clear'],
                          groups=df_moto_1['new_category'],
                          alpha=0.05)
print(tukey8) 

tukey9 = pairwise_tukeyhsd(endog=df_moto_1['front_travel'],
                          groups=df_moto_1['new_category'],
                          alpha=0.05)
print(tukey9) 

tukey10 = pairwise_tukeyhsd(endog=df_moto_1['rear_wheel'],
                          groups=df_moto_1['new_category'],
                          alpha=0.05)
print(tukey10) 

tukey11 = pairwise_tukeyhsd(endog=df_moto_1['rear_travel'],
                          groups=df_moto_1['new_category'],
                          alpha=0.05)
print(tukey11) 

tukey12 = pairwise_tukeyhsd(endog=df_moto_1['hip_angle'],
                          groups=df_moto_1['new_category'],
                          alpha=0.05)
print(tukey12) 

tukey13 = pairwise_tukeyhsd(endog=df_moto_1['chain'],
                          groups=df_moto_1['new_category'],
                          alpha=0.05)
print(tukey13) 
