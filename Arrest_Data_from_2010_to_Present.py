#!/usr/bin/env python
# coding: utf-8

# # Arrest_Data_from_2010_to_Present

# ## Section 1

# In[1]:


import numpy as np

import pandas as pd

df = pd.read_csv('Arrest_Data_from_2010_to_Present.csv')

df.head()


# In[88]:


df.info()


# In[2]:


cols = ['Sex Code', 'Descent Code']


# In[3]:


df[cols] = df[cols].astype('category')


# In[91]:


df.info()


# In[4]:


df['Arrest Date'] = df['Arrest Date'].astype('datetime64[ns]')


# In[5]:


df.info()


# In[6]:


df.head(2)


# In[7]:


# filling missing values

list_of_cols = ['Time','Charge Group Code','Charge Group Description','Charge Description', 'Cross Street']

df[list_of_cols] = df[list_of_cols].fillna('NaN')

df.count()


# In[13]:


(df['Cross Street'] == 'NaN').value_counts()


# In[14]:


df.info()


# In[15]:


start_date = '2018/01/01'
end_date = '2018/12/31'


# In[16]:


filter_ = (df['Arrest Date'] >= start_date) & (df['Arrest Date'] <= end_date)


# In[17]:


# 1. Answer

df.loc[filter_].count()


# In[18]:


# 2. Answer

df.loc[filter_]['Area Name'].value_counts()


# In[19]:


list_of_charge_groups = ['Vehicle Theft','Robbery','Burglary','Receive Stolen Property']

charge_groups = df['Charge Group Description'].str.contains('|'.join(list_of_charge_groups), na=False)

df.loc[filter_]['Age'][charge_groups].describe()


# In[20]:


# verification of quantiles

df.loc[filter_]['Age'][charge_groups].quantile([.25,.5,.75])


# In[21]:


# 3. Answer

df.loc[filter_]['Age'][charge_groups].quantile(.95)


# In[22]:


from scipy import stats

# excluding charges related to minor and unknown

revised_charge_groups = df['Charge Group Description'].str.replace(
    r'Pre-Delinquency, Non-Criminal Detention','').str.contains('',na=False)

data_to_use = pd.to_numeric(df.loc[filter_]['Age'][revised_charge_groups])


# In[23]:


stats.zscore(data_to_use)


# In[24]:


# 4. Answer

abs(stats.zscore(data_to_use)).max()


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[26]:


df.head(2)


# In[9]:


df.index


# In[10]:


new_df = df.set_index('Arrest Date')


# In[11]:


new_df.index


# In[12]:


new_df.head(1)


# In[13]:


new_df.info()


# In[258]:


df_all_arrests_code = new_df[['Arrest Type Code']]
df_all_arrests_code


# In[260]:


felony_arrests = df_all_arrests_code[df_all_arrests_code['Arrest Type Code'] == 'F']['2010':'2018'].resample('A').count()

felony_arrests


# In[246]:


felony_arrests.plot(title='Felony Arrests (2010-2018)',marker='D')
plt.xlabel('Year')
plt.ylabel('Number of Arrests')
plt.legend()
plt.show()


# In[250]:


# 5. Answer

predicted_felony_arrest_2019 = int(felony_arrests.mean())

predicted_felony_arrest_2019


# In[255]:


new_df.head(2)


# In[286]:


dir(math)


# In[14]:


Location_coordinates = new_df[['Location']]
Location_coordinates


# In[16]:


(Location_coordinates['Location'] == '(0,0)').value_counts()


# In[ ]:


old_coordinates = ('34.050536, -118.247861')
new_longitude = ('34.41140, -118.148717') # calculated online using "http://edwilliams.org/gccalc.htm"


# In[46]:


# 6. Answer

arrests_within_2km_2018 = Location_coordinates[(Location_coordinates['Location'] >= '(34.050536, -118.247861)') &
                                               (Location_coordinates['Location'] <= '(34.41140, -118.148717)')]['2018'].count()

arrests_within_2km_2018


# In[464]:


new_df.head()


# In[71]:


pico_arrest_2018 = new_df[new_df['Address'] == 'PICO']['2018']

pico_arrest_2018.head(2)


# In[81]:


# length of 'Pico Boulevard' taken online from "https://en.wikipedia.org/wiki/Pico_Boulevard"

pico_blvd_length_km = 3

len(pico_arrest_2018)


# In[82]:


# 7. Answer

per_km_arrests_pico_2018 = int(len(pico_arrest_2018)/pico_blvd_length)

per_km_arrests_pico_2018


# In[328]:


new_df = df.set_index('Arrest Date')

new_df.head()


# In[329]:


new_df.info()


# In[331]:


df_charge_codes = new_df[['Area ID','Charge Group Code','Charge Group Description']]

df_charge_codes.head()


# In[358]:


df_charge_codes['Charge Group Code'] = df_charge_codes['Charge Group Code'].astype(float)


# In[359]:


df_charge_codes.info()


# In[360]:


df_2018 = df_charge_codes['2018']


# In[361]:


df_2018.head()


# In[421]:


grouped_df = df_2018.groupby('Charge Group Description')[['Charge Group Code','Area ID']].mean()
grouped_df


# In[431]:


grouped_df['area_to_charge_ratio'] = grouped_df['Area ID'] / grouped_df['Charge Group Code']


# In[432]:


grouped_df.head()


# In[437]:


total_of_charge_group_code = len(grouped_df['Charge Group Code'])

total_of_charge_group_code


# In[438]:


grouped_df['charge_group_ratio'] = grouped_df['Charge Group Code']/len(grouped_df['Charge Group Code'])


# In[439]:


grouped_df.head()


# In[441]:


grouped_df['ratio_bw_two_probab'] = grouped_df['area_to_charge_ratio']/grouped_df['charge_group_ratio']


# In[451]:


new_grouped_df = grouped_df.sort_values('ratio_bw_two_probab',ascending=False)


# In[453]:


# 8. Answer

new_grouped_df['ratio_bw_two_probab'].mean()

