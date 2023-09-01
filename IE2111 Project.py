import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import aic
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

electricity_df = pd.read_csv("IE2111 Project - Electricity Tariffs, Monthly.csv") #Read CSV into DataFrame Object
electricity_df['Month'] = pd.to_datetime(electricity_df['Month']) #Convert 'Month' into a Datetime Object
electricity_df = electricity_df.iloc[::3]
electricity_df.index = pd.DatetimeIndex(electricity_df['Month']) #Set 'Month' as a Pandas DatetimeIndex
print(electricity_df)

seasonal_decompose(electricity_df['Electricity_Tariff'], model='additive').plot() #Decompose the Time Series into Trend, Seasonality & Random Noise
plt.show() #Plot the Decomposed Time Series in the Console

energy_df = pd.read_csv("IE2111 Project - Average USEP, Monthly.csv") #Read CSV into DataFrame Object
energy_df['Month'] = pd.to_datetime(energy_df['Month'], dayfirst=True) #Convert 'Month' into a Datetime Object
energy_df.index = pd.DatetimeIndex(energy_df['Month']) #Set 'Month' as a Pandas DatetimeIndex
print(energy_df)

data_df = pd.merge(electricity_df['Electricity_Tariff'], energy_df['Energy_Price'], left_index=True, right_index=True) #Merge the 2 Time Series based on Month Index
data_df.reset_index(level=0, inplace=True)
print(data_df)

plt.plot(data_df['Month'], data_df['Electricity_Tariff'], label="Electricity_Tariff") #Plot Electricity_Tariff & Nat_Gas_Price Together
plt.plot(data_df['Month'], data_df['Energy_Price'], label="Energy_Price")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Energy Price VS Electricity Tariff Over Time')
plt.show()

print(grangercausalitytests(data_df[['Electricity_Tariff', 'Energy_Price']], maxlag=12, addconst=True, verbose=True))

adfuller_test = {'Electricity_Tariff': adfuller(data_df['Electricity_Tariff'])[1], 'Energy_Price': adfuller(data_df['Energy_Price'])[1]}

differenced_df, differenced = data_df, 0
while not max(adfuller_test.values()) < 0.05: #Difference DataFrame until both Time Series are Stationary as determined by the Adfuller Test
    differenced_df = differenced_df.diff().dropna()
    for series in adfuller_test:
        adfuller_test[series] = adfuller(differenced_df[series])[1]
    differenced += 1 #Count the Number of Differences
print(differenced_df)

data_array = np.array(data_df[['Electricity_Tariff', 'Energy_Price']].dropna())
differenced_array = np.array(differenced_df[['Electricity_Tariff', 'Energy_Price']].dropna())

n_obs = 8
df_train, df_test = differenced_array[:-n_obs], data_array[-n_obs:]

model = VAR(endog=df_train)
print(model.select_order(maxlags=12).summary())
model_fit = model.fit(ic='aic')
print(model_fit.summary())

lag_order = model_fit.k_ar
model_predict = model_fit.forecast(df_train[-lag_order:], n_obs)
predict_differenced = pd.DataFrame(model_predict, index=data_df.index[-n_obs:], columns=data_df[['Electricity_Tariff', 'Energy_Price']].columns + '_Predicted')
predict_df = predict_differenced.cumsum() + df_test

master_df = pd.merge(data_df, predict_df, left_index=True, right_index=True) #Merge the 2 Time Series based on Month Index
master_df.reset_index(level=0, inplace=True)
print(master_df)

plt.plot(master_df['Month'], master_df['Electricity_Tariff'], label="Actual") #Plot Electricity_Tariff & Electricity_Tariff_Predicted Together
plt.plot(master_df['Month'], master_df['Electricity_Tariff_Predicted'], label="Predicted")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Predicted VS Actual Electricity Tariffs')
plt.show()

df_actual = [df_test[i][0] for i in range(len(df_test))]
forecast_errors = [df_actual[i] - predict_df.Electricity_Tariff_Predicted[i+53] for i in range(len(df_actual))]
bias = sum(forecast_errors) * 1.0/len(df_actual)
mae = mean_absolute_error(df_actual, predict_df.Electricity_Tariff_Predicted)
mse = mean_squared_error(df_actual, predict_df.Electricity_Tariff_Predicted)
rmse = sqrt(mse)

df_forecast = differenced_array[-n_obs:]
model_forecast = model_fit.forecast(df_forecast[-lag_order:], n_obs)
forecast_differenced = pd.DataFrame(model_forecast, columns=data_df[['Electricity_Tariff', 'Energy_Price']].columns + '_Forecast')
forecast_df = forecast_differenced.cumsum() + [23.0, 88.64]
print(forecast_df)
