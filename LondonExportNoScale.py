#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pandas import datetime
from matplotlib import pyplot as plt
import math
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import model_from_json



energy = pd.read_csv("energy.csv")

housecount = energy.groupby('day')[['LCLid']].nunique()



energy = energy.groupby('day')[['energy_sum']].sum()
energy = energy.merge(housecount, on = ['day'])
energy = energy.reset_index()


energy.day = pd.to_datetime(energy.day,format='%Y-%m-%d').dt.date



energy['avg_energy'] =  energy['energy_sum']/energy['LCLid']

#Weather data set
weather = pd.read_csv('weather_daily_darksky.csv')



weather['day']=  pd.to_datetime(weather['time']) # day is given as timestamp
weather.loc[0:5,'day']



weather['day']=  pd.to_datetime(weather['day'],format='%Y%m%d').dt.date
weather.loc[0:5,'day']



# selecting numeric variables
weather = weather[['temperatureMax', 'windBearing', 'dewPoint', 'cloudCover', 'windSpeed',
       'pressure', 'apparentTemperatureHigh', 'visibility', 'humidity',
       'apparentTemperatureLow', 'apparentTemperatureMax', 'uvIndex',
       'temperatureLow', 'temperatureMin', 'temperatureHigh',
       'apparentTemperatureMin', 'moonPhase','day']]
weather = weather.dropna()



#Merge weather and energy dataset
weather_energy =  energy.merge(weather,on='day')


#Get data on exchange rate. Exchange rate is used as an indicator of economic performance, on a daily basis.
exchange = pd.read_csv('exchange.csv')
exchange.tail(5)



exchange['Date']=  pd.to_datetime(exchange['Date'],format='%d/%m/%Y').dt.date
exchange['Date']=  pd.to_datetime(exchange['Date'],format='%Y/%m/%d').dt.date




exchange = exchange.rename(columns={'Date': 'day'})


# In[24]:


weather_energy_gbp =  weather_energy.merge(exchange,on='day')


# In[25]:


#Correcting an anomoly
weather_energy_gbp['avg_energy'][298] =weather_energy_gbp['avg_energy'].mean() 


# In[30]:


weather_energy_gbp['Close'].fillna((weather_energy_gbp['Close'].mean()),inplace=True)


weather_energy_gbp.isnull().values.any()


# In[41]:


columns=list(weather_energy_gbp.columns)
dropcolumn = ['day','energy_sum','LCLid','avg_energy']
for i in dropcolumn:
    if i in columns:
        columns.remove(i)
x = weather_energy_gbp[columns]
y = weather_energy_gbp['avg_energy']


# # In[42]:


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)


# # In[43]:


# #Get p-values (use of statsmodel insteal of lm package for detailed statistical summary)
# x_train=sm.add_constant(x_train)
# lm = sm.OLS(y_train,x_train )
# lm2 = lm.fit()
# print(lm2.summary())



# # In[ ]:





# # In[48]:


# #Refine model using factors with P < 0.5, with only 1 temperature factor to reduce multicollinearity
# x = weather_energy_gbp[['visibility','apparentTemperatureLow','uvIndex','Close']]
# y = weather_energy_gbp['avg_energy']
# x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.3, random_state=101)


# # In[49]:


# x_train2


# # In[50]:


# x_train2=sm.add_constant(x_train2)
# lm_2 = sm.OLS(y_train2,x_train2 )
# lm_3 = lm_2.fit()


# # In[51]:


# print (lm_3.summary())


# # Average Energy use = 20.7945 -0.0598(visibility ) -0.1606(apparentTemperatureLow) - -0.4113(uvIndex) -5.0074 (Close)

# # #PCA Principal Component Analysis
# # from sklearn.decomposition import PCA
# # weather_energy_gbp.shape
# # 

# # In[ ]:





# # In[ ]:





# In[ ]:





# #Creating weather clusters using K means clustering
# scaler = MinMaxScaler()
# weather_scaled = scaler.fit_transform(weather_energy_gbp[['visibility','apparentTemperatureLow','uvIndex','Close']])

# kmeans = KMeans(n_clusters=4, max_iter=600, algorithm = 'auto')
# kmeans.fit(weather_scaled)
# weather_energy_gbp['weather_cluster'] = kmeans.labels_

# # optimum K
# Nc = range(1, 15)
# kmeans = [KMeans(n_clusters=i) for i in Nc]
# kmeans
# 
# score = [kmeans[i].fit(weather_scaled).score(weather_scaled) for i in range(len(kmeans))]
# score
# plt.plot(Nc,score)
# plt.xlabel('Number of Clusters')
# plt.ylabel('Score')
# plt.title('Elbow Curve')
# plt.show()
# #Optimal k=3

# plt.figure(figsize=(20,5))
# plt.subplot(1, 3, 1)
# plt.scatter(weather_energy_gbp.weather_cluster,weather_energy_gbp.visibility)
# plt.title('Weather Cluster vs. visibility')
# plt.subplot(1, 3, 2)
# plt.scatter(weather_energy_gbp.weather_cluster,weather_energy_gbp.apparentTemperatureLow)
# plt.title('Weather Cluster vs. apparentTemperatureLow')
# plt.subplot(1, 3, 3)
# plt.scatter(weather_energy_gbp.weather_cluster,weather_energy_gbp.uvIndex)
# plt.title('Weather Cluster vs. uvIndex')
# #visibility','apparentTemperatureLow','uvIndex','Close'
# plt.show()

# fig, ax1 = plt.subplots(figsize = (10,6))
# ax1.scatter(weather_energy_gbp.temperatureMax, 
#             weather_energy_gbp.humidity, 
#             s = weather_energy_gbp.windSpeed*20,
#             c = weather_energy_gbp.weather_cluster)
# ax1.set_xlabel('Temperature')
# ax1.set_ylabel('Humidity')
# plt.show()



#Holiday
holiday = pd.read_csv('uk_bank_holidays.csv')
holiday['Bank holidays'] = pd.to_datetime(holiday['Bank holidays'],format='%Y-%m-%d').dt.date
holiday.head(4)


# In[54]:


weather_energy_gbp = weather_energy_gbp.merge(holiday, left_on = 'day',right_on = 'Bank holidays',how = 'left')
weather_energy_gbp['holiday_ind'] = np.where(weather_energy_gbp['Bank holidays'].isna(),0,1)


# #Arimax
# weather_energy['Year'] = pd.DatetimeIndex(weather_energy['day']).year  
# weather_energy['Month'] = pd.DatetimeIndex(weather_energy['day']).month
# weather_energy.set_index(['day'],inplace=True)

# model_data = weather_energy_gbp[['avg_energy','weather_cluster','holiday_ind']]
# # train = model_data.iloc[0:round(len(model_data)*0.90)]
# # test = model_data.iloc[len(train)-1:]
# train = model_data.iloc[0:(len(model_data)-30)]
# test = model_data.iloc[len(train):(len(model_data)-1)]

# train['avg_energy'].plot(figsize=(25,4))
# test['avg_energy'].plot(figsize=(25,4))

# In[55]:



sns.set(style="white")


# Compute the correlation matrix
corr = weather_energy_gbp.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:





# In[ ]:





# In[56]:


#LSTM
np.random.seed(11)
dataframe = weather_energy_gbp.loc[:,'avg_energy']
dataset = dataframe.values
dataset = dataset.astype('float32')


# In[57]:


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[58]:


reframed = series_to_supervised(dataset, 7,1)
reframed.head(3)


# reframed['weather_cluster'] = weather_energy_gbp.weather_cluster.values[7:]
# reframed['holiday_ind']= weather_energy_gbp.holiday_ind.values[7:]

# In[59]:


reframed['Close'] = weather_energy_gbp.Close.values[7:]
reframed['apparentTemperatureLow'] = weather_energy_gbp.apparentTemperatureLow.values[7:]
reframed['uvIndex'] = weather_energy_gbp.uvIndex.values[7:]
reframed['visibility'] = weather_energy_gbp.visibility.values[7:]
reframed['holiday_ind']= weather_energy_gbp.holiday_ind.values[7:]
#visibility','apparentTemperatureLow','uvIndex','Close'


# In[60]:


reframed = reframed.reindex(['Close', 'apparentTemperatureLow','uvIndex','visibility','holiday_ind','var1(t-7)', 'var1(t-6)', 'var1(t-5)', 'var1(t-4)', 'var1(t-3)','var1(t-2)', 'var1(t-1)', 'var1(t)'], axis=1)
reframed = reframed.values


# In[61]:


# scaler = MinMaxScaler(feature_range=(0, 1))
# reframed = scaler.fit_transform(reframed)


# In[62]:


# split into train and test sets
train = reframed[:(len(reframed)-30), :]
test = reframed[(len(reframed)-30):len(reframed), :]


# In[63]:


train


# In[64]:


train[:, :-1] #Taking all but last value as train


# In[65]:


train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]


# In[66]:


test_X[5]


# In[67]:


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[68]:


test_X[5]


# In[69]:


test_X[5]


# In[70]:


# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
model.save_weights("modelNew.h5")
print("Saved model to disk")
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, verbose=2, shuffle=False)

# serialize model to JSON
model_json = model.to_json()
with open("modelNew.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelNew.h5")
print("Saved model to disk")

# load json and create model
json_file = open('modelNew.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("modelNew.h5")
print("Loaded model from disk")

model.compile(loss='mae', optimizer='adam')
history = model.fit(train_X, train_y, epochs=50, batch_size=72, verbose=2, shuffle=False)



# In[71]:


# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.legend()
pyplot.show()


# In[72]:


test_X[1]


# In[73]:


yhat = model.predict(test_X)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[2])


# In[74]:


yhat[1]


# In[75]:


test_X[1]


# In[76]:


test_y[1]


# In[77]:


# invert scaling for forecast
inv_yhat = np.concatenate((test_X,yhat ), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_X,test_y), axis=1)


# In[78]:


inv_y[1] #First value is added in


# In[79]:


# inv_y = scaler.inverse_transform(inv_y)
inv_y[1]


# In[80]:


act = [i[12] for i in inv_y] # Last element is the predicted average energy
pred = [i[12] for i in inv_yhat] # Last element is the actual average energy
from sklearn.metrics import mean_squared_error
# calculate RMSE
import math
rmse = math.sqrt(mean_squared_error(act, pred))
percent = rmse*100/weather_energy_gbp['avg_energy'].mean() 
print('Test RMSE: %.3f' % rmse)
print('RMSE Percent error: %.3f' % percent)


# In[81]:


inv_yhat[1]


# In[82]:


inv_y[1]


# In[83]:


predicted_lstm = pd.DataFrame({'predicted':pred,'avg_energy':act})
predicted_lstm['avg_energy'].plot(figsize=(25,10),color = 'red')
predicted_lstm['predicted'].plot(color = 'blue')
pyplot.legend()
plt.show()


# In[84]:


predicted_lstm 


# In[85]:


latestpred= pred[-1] #Get latest predicted value
print('Latest predicted value: %.3f' % latestpred)



# In[ ]:


    


