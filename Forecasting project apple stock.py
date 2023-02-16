#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries and dataset

# In[256]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[257]:


data = pd.read_csv('AAPL.csv')
data.shape


# In[258]:


data.index


# In[259]:


data.info()
data.head()


# In[260]:


data.tail()


# In[261]:


data.describe()


# # Data Preprocessing

# In[262]:


## Converting Date to DateTime Object
data['Date']


# In[263]:


data['Date'] = pd.to_datetime(data['Date'],format='%Y-%m-%d')


# In[264]:


data.dtypes


# In[265]:


## Making Date as Index 
data.set_index('Date',inplace=True)


# In[266]:


data['Date'] = data.index


# In[267]:


data.head()


# In[268]:


## Checking of any null values
data.isnull().sum()


# In[269]:


## Visualizations


# In[270]:


col_names = data.columns

fig = plt.figure(figsize=(24, 24))
for i in range(6):
  ax = fig.add_subplot(6,1,i+1)
  ax.plot(data.iloc[:,i],label=col_names[i])
  data.iloc[:,i].rolling(100).mean().plot(label='Rolling Mean')
  ax.set_title(col_names[i],fontsize=18)
  ax.set_xlabel('Date')
  ax.set_ylabel('Price')
  ax.patch.set_edgecolor('black')  
  plt.style.context('fivethirtyeight')
  plt.legend(prop={'size': 12})
  plt.style.use('fivethirtyeight')
fig.tight_layout(pad=3.0)

plt.show()


# In[271]:


# Feature Selection
data_feature_selected = data.drop(axis=1,labels=['Open','High','Low','Adj Close','Volume'])


# We can ignore the features like Open, Low, High, AdjClose, Volume and consider Close as our target variable because it is the final value for that particular date.

# In[272]:


col_order = ['Date', 'Close']
data_feature_selected = data_feature_selected.reindex(columns=col_order)
data_feature_selected


# In[273]:


## Resampling
monthly_mean = data_feature_selected['Close'].resample('M').mean()


# In[274]:


monthly_data = monthly_mean.to_frame()
monthly_data


# In[275]:


close_data = pd.DataFrame(data['Close'])
close_data.head()


# In[276]:


fig = plt.figure(figsize=(20,5))
plt.plot(monthly_data['Close'],label='Monthly Averages Apple Stock')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.set_title('Monthly Resampled Data')
plt.style.use('fivethirtyeight')
plt.legend(prop={'size': 12})
plt.show()


# In[277]:


monthly_data['Year'] = monthly_data.index.year
monthly_data['Month'] = monthly_data.index.strftime('%B')
monthly_data['dayofweek'] = monthly_data.index.strftime('%A')
monthly_data['quarter'] = monthly_data.index.quarter
monthly_data


# In[278]:


# Box Plot
fig = plt.figure(figsize=(8,6))
sns.boxplot(monthly_data['Close']).set_title('Box Plot Apple Stock Price')
plt.style.context('fivethirtyeight')


# The Distribution in boxplot shows it is Right Skewed

# In[279]:


#QQ Plot
from statsmodels.graphics.gofplots import qqplot as qq
qq_plot = qq(monthly_data['Close'],line='s')
plt.title('QQ Plot Apple Stock Price')


# The above QQ plot shows extent of both right and left skews

# In[280]:


# Skewness & Kurtosis
print('Skewness of Distribution is ',monthly_data['Close'].skew())
print('Kurtosis of Distribution is ',monthly_data['Close'].kurtosis())


# In[281]:


#Boxplots for everyyear in data
plt.figure(figsize=(20,10))
ax = sns.boxplot(x=monthly_data['Year'],y=monthly_data['Close'],palette='RdBu')
ax.set_title('Box Plots Year Wise-Apple Stock Price')
plt.style.context('fivethirtyeight')


# The above boxplots shows there are Outliers present in Year 2012 and 2019
# The year 2019 is most volatile among all years
# We can see the Upward Rising Trend

# In[282]:


fig, ax = plt.subplots(figsize=(20,10))
palette = sns.color_palette("mako_r", 4)
a = sns.barplot(x="Year", y="Close",hue = 'Month',data=monthly_data)
a.set_title("Stock Prices Year & Month Wise",fontsize=15)
plt.legend(loc='upper left')
plt.show()


# In[283]:


fig = plt.figure(figsize=(15,20))
fig.set_size_inches(15,20)
group_cols = monthly_data.columns

for enum,i in enumerate(group_cols[1:]):
  ax = fig.add_subplot(4,1,enum+1)
  Aggregated = pd.DataFrame(monthly_data.groupby(str(i))["Close"].mean()).reset_index().sort_values('Close')
  sns.barplot(data=Aggregated,x=str(i),y="Close",ax=ax)
  ax.set(xlabel=str(i), ylabel='Mean Close')
  ax.set_title("Average Stock Price By {}".format(str(i)),fontsize=15)
  plt.xticks(rotation=45)
  
plt.tight_layout(pad=1)


# In[284]:


fig = plt.figure(figsize=(15,18))
fig.set_size_inches(15,18)
group_cols = monthly_data.columns

for enum,i in enumerate(group_cols[1:]):
  ax = fig.add_subplot(4,1,enum+1)
  Aggregated = pd.DataFrame(monthly_data.groupby(str(i))["Close"].mean()).reset_index().sort_values('Close')
  sns.lineplot(data=Aggregated,x=str(i),y="Close",ax=ax)
  ax.set(xlabel=str(i), ylabel='Mean Close')
  ax.set_title("Average Stock Price By {}".format(str(i)),fontsize=15)
  plt.xticks(rotation=45)
  
plt.tight_layout(pad=1)


# From the above plots, 2013 and 2016 are the only years where Mean price is lower than previous Year.
# The Average Stock Price is lower at start of the week in comparision to the end of the week.
# The Average Price is Highest in the Month of November.
# Quarter4 is the best for Apple according to average stock price.

# In[285]:


##Decomposition
from pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose as sd
rcParams['figure.figsize'] = 18, 8
plt.figure(figsize=(20,16))
decomposed_series = sd(monthly_data['Close'],model='additive')
decomposed_series.plot()
plt.show()


# In[286]:


decomposed_series.seasonal['2012':'2013'].plot()
fig = plt.figure(figsize=(8,8))


# From decompostion,we can see the
# Trend : Overall there is an Upward Trend
# Seasonality :we can see that upward and downward cycles in the plots,so there is seasonality.
# Stationarity:The given Time Series is Non-Stationary because it didn't have constant mean,constant variance.

# In[287]:


# ACF Plots
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig,(ax1,ax2) = plt.subplots(2,figsize=(12,12))
acf = plot_acf(monthly_data['Close'],lags=90,ax=ax1)
ax1.set_title('AutoCorrelation Long Term')
acf = plot_acf(monthly_data['Close'],lags=30,ax=ax2)
ax2.set_title('AutoCorrelation Short Term')
ax1.set_ylabel('Correlation')
ax1.set_xlabel('Lags')
ax2.set_ylabel('Correlation')
ax2.set_xlabel('Lags')


# From the above ACF Plot,we can see the Slow Decay of correlation values which indicates the series is Non-stationary.

# In[288]:


# PACF Plots
fig,(ax1,ax2) = plt.subplots(2,figsize=(10,10))
pacf = plot_pacf(monthly_data['Close'],ax=ax1)
ax1.set_title('Partial AutoCorrelation Long Term')
pacf = plot_pacf(monthly_data['Close'],lags=30,ax=ax2)
ax2.set_title('Partial AutoCorrelation Short Term')
ax1.set_ylabel('Correlation')
ax1.set_xlabel('Lags')
ax2.set_ylabel('Correlation')
ax2.set_xlabel('Lags')
plt.tight_layout(pad=1)


# From the above PACF plot,there is a Sudden Decay at lag -1

# In[289]:


##Stationarity Test of Time Series Using Augmented Dickey-Fuller(ADF) Test
from statsmodels.tsa.stattools import adfuller


# In[290]:


def ad_fuller_func(X):
  result_ad_fuller = adfuller(X)
  print('ADF Statistic: %f' % result_ad_fuller[0])
  print('p-value: %f' %result_ad_fuller[1])
  print('Critical Values:')
  for key, value in result_ad_fuller[4].items():
	  print('\t%s: %.3f' % (key, value))
 
  if result_ad_fuller[0] < result_ad_fuller[4]['5%']:
    print('Reject Null Hypothesis(Ho)-Time Series is Stationary')
  else:
    print('Failed to Reject Ho-Time Series is Non-Stationary')


# In[291]:


ad_fuller_func(monthly_data['Close'])


# Time Series is Not Stationary as observed earlier also by Decomposition(Trend and Seasonality Present)
# Statistically verified by ADF Test

# In[292]:


##Transformations To Make Series Stationary


# In[293]:


##Differencing By 1
monthly_diff = monthly_data['Close'] - monthly_data['Close'].shift(1)


# In[294]:


##ACF and PACF plots after transforming series into Stationary
fig,(ax1,ax2) = plt.subplots(2,figsize=(10,10))
acf = plot_acf(monthly_diff[1:],lags=30,ax=ax1)
pacf = plot_pacf(monthly_diff[1:],lags=30,ax=ax2)
ax1.set_title('Autocorrelation For Differenced(1)')
ax1.set_ylabel('Correlation')
ax1.set_xlabel('Lags')
ax2.set_title('Partial Autocorrelation For Differenced(1)')
ax2.set_ylabel('Correlation')
ax2.set_xlabel('Lags')
plt.tight_layout(pad=1)


# From the above ACF and PACF plots, we can confirm that Differencing once has transformed time series into Stationary.

# # MODELLING & FORECASTING

# In[295]:


#splitting the data
train_data = close_data.loc[:'2017-12-29']
test_data = close_data.loc['2018-01-02':]
train_data


# im splitting the first 6 years into training data and the final year as the testing data.

# In[296]:


print(len(train_data))
print(len(test_data))


# # ARIMA Model

# In[297]:


import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA


# From the Grid Search to Select Parameters especially for Seasonal Component of the Time Series.
# The p value (AR)derived by the PACF plot
# The q value (MA) derived by the ACF Plot
# The d value (Differencing) derived by differencing and observing stationarity.

# In[298]:


model_arima = ARIMA(close_data,order=(2,1,2))
arima_fit  = model_arima.fit()
arima_fit.summary()


# In[299]:


prediction = pd.DataFrame(arima_fit.predict(typ = 'levels'))
prediction.columns=['Close']
prediction.tail()


# In[300]:


plt.figure(figsize=(20,6), dpi=90)
plt.plot(close_data['Close'], label='Actual')
plt.plot(prediction, label='Prediction')
plt.title('Actual vs Prediction')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[301]:


# Residual plot
residuals = arima_fit.resid
fig, axs = plt.subplots(1,2,figsize=(25,5))
residuals.plot(title="Residuals", ax=axs[0])
residuals.plot(kind='kde', title='Density', ax=axs[1])
plt.show()


# In[302]:


model_arima = ARIMA(train_data,order=(2,1,2))
result = model_arima.fit()
result.summary()


# In[303]:


test_pred = pd.DataFrame(result.predict(len(train_data),len(train_data)+501,typ='levels'))
test_pred.index = test_data.index
test_pred.columns=test_data.columns
test_pred


# In[304]:


plt.figure(figsize=(12,5), dpi=80)
plt.plot(train_data, label = 'Train')
plt.plot(test_data, label='Test')
plt.plot(test_pred, label='Prediction')
plt.title('Actuals vs Prediction')
plt.legend(loc='upper left', fontsize=10)
plt.show()


# In[305]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

mse_arima = mean_squared_error(test_pred.Close,test_data.Close)
rmse_arima = np.sqrt(mse_arima)
mape_arima = np.round(mean_absolute_percentage_error(test_pred.Close,test_data.Close),3)

print('\n MSE = ',mse_arima)
print('\n RMSE = ',rmse_arima)
print('\n MAPE = ', mape_arima)


# # Seasonal ARIMA Model

# In[306]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm


# In[307]:


model_sarima = sm.tsa.SARIMAX(train_data,order=(2,1,2),seasonal_order=(1,1,0,57),enforce_invertibility=False,enforce_stationarity=False)
sarima_fit = model_sarima.fit()
sarima_fit.summary()


# In[308]:


#Residuals_plot
plt.figure(figsize=(20,4))
plt.plot(residuals)
plt.axhline(0, linestyle='--', color='k')
plt.title('Residuals from SARIMA Model', fontsize=20)
plt.ylabel('Error', fontsize=20)


# In[309]:


print('Root Mean Squared Error:', np.sqrt(np.mean(residuals**2)))


# In[310]:


test_prediction = pd.DataFrame(sarima_fit.predict(len(train_data),len(train_data)+501,typ='levels'))
test_prediction.index = test_data.index
test_prediction


# In[311]:


plt.figure(figsize=(12,8), dpi=80)
plt.plot(train_data, label = 'Train')
plt.plot(test_data, label='Test')
plt.plot(test_prediction, label='Prediction')
plt.title('Actuals vs Prediction')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True)
plt.show()


# In[312]:


mse_sarimax = mean_squared_error(test_pred,test_data)
rmse_sarimax= np.round(np.sqrt(mean_squared_error(test_data,test_prediction)),2)
mape_sarimax = np.round(mean_absolute_percentage_error(test_prediction,test_data),3)


# In[313]:


print('\n MSE = ',mse_sarimax)
print('\n RMSE = ',rmse_sarimax)
print('\n MAPE = ', mape_sarimax)


# In[ ]:





# In[321]:


## Final_Metrics table
cols = ['Model_Name', 'RMSE', 'MAPE']
Final_Metrics = pd.DataFrame(columns = cols)
def appending(x):
    Final_Metrics.append(x,ignore_index=True)


# In[322]:


arima = pd.Series({'Model_Name': "ARIMA Model",
                     'RMSE': rmse_arima,
                     'MAPE':mape_arima
                   })
Final_Metrics = Final_Metrics.append(arima,ignore_index=True)


# In[323]:


sarimax = pd.Series({'Model_Name': "Seasonal ARIMA Model",
                     'RMSE': rmse_sarimax,
                     'MAPE':mape_sarimax
                   })
Final_Metrics = Final_Metrics.append(sarimax,ignore_index=True)


# In[324]:


Final_Metrics

Among the above values, I select sarima model for deployment because of the low rmse value.
# In[319]:


# Forecasting for next 30 days using sarima model

forecast = sarima_fit.predict(len(close_data),len(close_data)+29)
forecast_df = pd.DataFrame(forecast)
forecast_df.columns = ['Close']
forecast_df


# Set appropriate date as index for plotting forecast data
datetime = pd.date_range('2020-01-01', periods=30,freq='B')
date_df = pd.DataFrame(datetime,columns=['Date'])

data_forecast = forecast_df.set_index(date_df.Date)
data_forecast

plt.figure(figsize=(15,6), dpi=100)
plt.plot(close_data, label = 'Actual')
plt.plot(data_forecast, label='Forecast',color='red')
plt.title('Actual vs Forecast')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True)
plt.show()

