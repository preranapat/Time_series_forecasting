import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from io import StringIO

# --- Page Configuration ---
st.set_page_config(
    page_title="Air Passengers Forecasting",
    page_icon="✈️",
    layout="wide"
)

# --- Helper Functions ---

@st.cache_data
def load_data(uploaded_file):
    """Loads data from an uploaded file."""
    if uploaded_file is not None:
        try:
            # To read file as string and then parse
            string_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
            data = pd.read_csv(string_data, parse_dates=['Month'], index_col='Month')
            data.rename(columns={'#Passengers': 'Passengers'}, inplace=True)
            return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return None

def test_stationarity(timeseries):
    """
    Performs the Augmented Dickey-Fuller test and returns the results as a dataframe.
    """
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    
    is_stationary = dftest[1] <= 0.05
    return is_stationary, dfoutput
# --- Main Application ---
st.title("✈️ PragyanAI - Air Passengers Time Series Forecasting(TS)")
st.write("""
This interactive web app performs a time series analysis on the classic Air Passengers dataset.
You can upload your own data (in the same format) or use the default dataset to see the analysis and forecast.
""")

# --- 1. Load Data ---
st.sidebar.header("1. Load Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

data = None
if uploaded_file is not None:
    data = load_data(uploaded_file)
else:
    st.sidebar.info("Using the default Air Passengers dataset. Upload a file to analyze your own data.")
    try:
        data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month')
        data.rename(columns={'#Passengers': 'Passengers'}, inplace=True)
    except FileNotFoundError:
        st.error("Default 'AirPassengers.csv' not found. Please upload the file.")
        st.stop()
Pragyan AI and DS School
3:43 PM
if data is not None:
    # --- 2. Exploratory Data Analysis (EDA) ---
    st.header("1. Exploratory Data Analysis")
    
    st.subheader("Raw Data")
    st.dataframe(data.head())

    st.subheader("Passenger Numbers Over Time")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data['Passengers'], label='Original Data')
    ax.set_title('Air Passengers over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Passengers')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Time Series Decomposition")
    decomposition = seasonal_decompose(data['Passengers'], model='multiplicative')
    st.pyplot(decomposition.plot())

    # --- 3. Check for Stationarity ---
    st.header("2. Stationarity Analysis")
    st.write("""
    A time series is stationary if its statistical properties (like mean and variance) are constant over time.
    We use the **Augmented Dickey-Fuller (ADF) test** to check for stationarity.
    - **p-value <= 0.05**: Reject the null hypothesis (H0), the data is stationary.
    - **p-value > 0.05**: Fail to reject the null hypothesis (H0), the data is non-stationary.
    """)
    
    # Test on original data
    is_stationary_orig, adf_results_orig = test_stationarity(data['Passengers'])
    st.subheader("ADF Test on Original Data")
    st.dataframe(adf_results_orig)
    if is_stationary_orig:
        st.success("The original data is stationary.")
    else:
        st.warning("The original data is non-stationary. We need to apply transformations.")

    # Apply differencing
    data['Seasonal_Difference'] = data['Passengers'].diff(12)
    data_diff = data['Seasonal_Difference'].dropna()

    # Test on differenced data
    is_stationary_diff, adf_results_diff = test_stationarity(data_diff)
    st.subheader("ADF Test after Seasonal Differencing")
    st.write("To make the series stationary, we apply a seasonal difference of 12 months.")
    st.dataframe(adf_results_diff)
    if is_stationary_diff:
        st.success("The differenced data is now stationary!")
    else:
        st.warning("The differenced data is still not stationary. Further transformations may be needed.")
# --- 4. SARIMA Modeling and Forecasting ---
    st.header("3. SARIMA Forecasting Model")
    
    st.sidebar.header("2. Model Parameters")
    p = st.sidebar.slider('Non-seasonal (p)', 0, 5, 1)
    d = st.sidebar.slider('Non-seasonal (d)', 0, 5, 1)
    q = st.sidebar.slider('Non-seasonal (q)', 0, 5, 1)
    P = st.sidebar.slider('Seasonal (P)', 0, 5, 1)
    D = st.sidebar.slider('Seasonal (D)', 0, 5, 1)
    Q = st.sidebar.slider('Seasonal (Q)', 0, 5, 1)
    s = st.sidebar.number_input('Seasonality (s)', 1, 24, 12)
    
    st.sidebar.header("3. Forecast Horizon")
    future_steps = st.sidebar.slider('Months to Forecast into the Future', 1, 48, 36)

    # Split data
    train_data = data['Passengers'][:-future_steps]
    test_data = data['Passengers'][-future_steps:]

    if st.button("Train Model and Forecast"):
        with st.spinner("Training the SARIMA model... This may take a moment."):
            try:
                # Build and fit model
                model = SARIMAX(data['Passengers'], # Train on all data for future forecast
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, s),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                results = model.fit()

                st.subheader("Model Summary")
                st.text(results.summary())

                # --- 5. Visualize Forecast ---
                st.subheader(f"Forecast for the Next {future_steps} Months")
                
                # Get forecast
                forecast = results.get_forecast(steps=future_steps)
                pred_mean = forecast.predicted_mean
                pred_ci = forecast.conf_int()
                
                # Plotting
                fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(data.index, data['Passengers'], label='Historical Data')
                ax.plot(pred_mean.index, pred_mean, label='Future Forecast', color='red')
                ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.5, label='Confidence Interval')
                ax.set_title(f'Air Passengers - Future Forecast ({future_steps} Months)')
                ax.set_xlabel('Date')
                ax.set_ylabel('Number of Passengers')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
                st.subheader("Forecasted Values")
                forecast_df = pd.DataFrame({'Forecast': pred_mean, 'Lower CI': pred_ci.iloc[:, 0], 'Upper CI': pred_ci.iloc[:, 1]})
                st.dataframe(forecast_df)

            except Exception as e:
                st.error(f"An error occurred during model training: {e}")
