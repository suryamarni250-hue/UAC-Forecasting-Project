import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")

df = df.replace(',', '', regex=True)

for col in df.columns:
    if col != 'Date':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
df = df[~df.index.duplicated(keep='first')]
df = df.asfreq('D')
df.interpolate(method='linear', inplace=True)

target = 'Children in HHS Care'

df['lag_1'] = df[target].shift(1)
df['lag_7'] = df[target].shift(7)
df['rolling_mean_7'] = df[target].rolling(7).mean()

df['net_pressure'] = (
    df['Children transferred out of CBP custody'] -
    df['Children discharged from HHS Care']
)

df.dropna(inplace=True)

train = df[:-30]
test = df[-30:]

X_train = train.drop(target, axis=1)
y_train = train[target]

X_test = test.drop(target, axis=1)
y_test = test[target]

rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

arima_model = ARIMA(train[target], order=(2,1,2))
arima_fit = arima_model.fit()
arima_pred = arima_fit.forecast(steps=len(test))

st.title("Predictive Forecasting of Care Load")

st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Choose Model", ["Random Forest", "ARIMA"]
)

if model_choice == "Random Forest":
    predictions = rf_pred
else:
    predictions = arima_pred

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(test.index, y_test, label="Actual")
ax.plot(test.index, predictions, label="Forecast")
ax.legend()

st.pyplot(fig)

mae = np.mean(np.abs(y_test - predictions))
rmse = np.sqrt(np.mean((y_test - predictions)**2))

st.subheader("Model Performance")
st.write("MAE:", round(mae,2))
st.write("RMSE:", round(rmse,2))