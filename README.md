# Python (EDA + Forecasting & Trend Analysis)
# Tasks:
 
 
# (i)Adjusted stock prices

```python
pip install pandas numpy matplotlib seaborn statsmodels prophet
```
![Dashboard Screenshot](https://github.com/Krishna5620/New-Stock-Exchange/blob/main/EDA%20SS%201.PNG)
![Dashboard Screenshot](https://github.com/Krishna5620/New-Stock-Exchange/blob/main/EDA%20SS%202.PNG)
![Dashboard Screenshot](https://github.com/Krishna5620/New-Stock-Exchange/blob/main/EDA%20SS%203.png)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
```

```python
# Load Data

fundamentals = pd.read_csv("D:/Data Analyst/Cap stone project/Cap Stone project data set/fundamentals.csv")
prices = pd.read_csv("D:/Data Analyst/Cap stone project/Cap Stone project data set/prices.csv")
prices_adj = pd.read_csv("D:/Data Analyst/Cap stone project/Cap Stone project data set/prices-split-adjusted.csv")
securities = pd.read_csv("D:/Data Analyst/Cap stone project/Cap Stone project data set/securities.csv")


print("Fundamentals:", fundamentals.head(), "\n")
print("Prices:", prices.head(), "\n")
print("Adjusted Prices:", prices_adj.head(), "\n")

# Ensure Date is parsed properly
prices_adj["date"] = pd.to_datetime(prices_adj["date"])
prices_adj = prices_adj.sort_values("date")

# Ensure Date is parsed properly
prices_adj["date"] = pd.to_datetime(prices_adj["date"])
prices_adj = prices_adj.sort_values("date")

# 2. Select a Stock

ticker = "AAPL"   # change this to the stock symbol you want

df = prices_adj[prices_adj["symbol"] == ticker].copy()
df = df[["date", "close"]].rename(columns={"date": "Date", "close": "Adj_Close"})
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

print(df.head())

# ---------- Plot Adjusted Close ----------
plt.figure(figsize=(12,5))
plt.plot(df.index, df["Adj_Close"], label=f"{ticker} Adjusted Close", color="blue")
plt.title(f"{ticker} Adjusted Close Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# ---------- Plot Returns Distribution ----------
df["Returns"] = df["Adj_Close"].pct_change()

plt.figure(figsize=(10,5))
sns.histplot(df["Returns"].dropna(), bins=50, kde=True, color="purple")
plt.title(f"{ticker} Daily Returns Distribution")
plt.xlabel("Daily Returns")
plt.ylabel("Frequency")
plt.show()

# ---------- Plot Rolling Volatility ----------
df["RollingVolatility"] = df["Returns"].rolling(window=30).std()

plt.figure(figsize=(12,5))
plt.plot(df.index, df["RollingVolatility"], label=f"{ticker} 30-day Rolling Volatility", color="orange")
plt.title(f"{ticker} Rolling Volatility (30-day window)")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.show()
```
![Dashboard Screenshot](https://github.com/Krishna5620/New-Stock-Exchange/blob/main/B.PNG)

# (ii)Financial metrics over years

```python
# Sample DataFrame (replace with your real data)
data = {
    'Year': [2018, 2019, 2020, 2021, 2018, 2019, 2020, 2021],
    'Sector': ['Tech', 'Tech', 'Tech', 'Tech', 'Finance', 'Finance', 'Finance', 'Finance'],
    'Revenue': [100, 120, 130, 150, 80, 90, 95, 110],
    'Net_Income': [20, 25, 30, 35, 15, 18, 20, 25]
}
df = pd.DataFrame(data)

# Ensure 'Year' is numeric
df['Year'] = pd.to_numeric(df['Year'])

# Plot Revenue trend
plt.figure(figsize=(12,6))
for sector in df['Sector'].unique():
    sector_data = df[df['Sector'] == sector]
    
    # Actual data points
    plt.scatter(sector_data['Year'], sector_data['Revenue'], label=f"{sector} Revenue")
    
    # Linear trendline
    z = np.polyfit(sector_data['Year'], sector_data['Revenue'], 1)
    p = np.poly1d(z)
    plt.plot(sector_data['Year'], p(sector_data['Year']), linestyle='--', label=f"{sector} Revenue Trend")

plt.xlabel('Year')
plt.ylabel('Revenue')
plt.title('Sector-wise Revenue Trend Over Years')
plt.legend()
plt.show()

# Optional: Plot Net Income trend

plt.figure(figsize=(12,6))
for sector in df['Sector'].unique():
    sector_data = df[df['Sector'] == sector]
    
    plt.scatter(sector_data['Year'], sector_data['Net_Income'], label=f"{sector} Net Income")
    
    z = np.polyfit(sector_data['Year'], sector_data['Net_Income'], 1)
    p = np.poly1d(z)
    plt.plot(sector_data['Year'], p(sector_data['Year']), linestyle='--', label=f"{sector} Net Income Trend")

plt.xlabel('Year')
plt.ylabel('Net Income')
plt.title('Sector-wise Net Income Trend Over Years')
plt.legend()
plt.show()
```
![Dashboard Screenshot](https://github.com/RushiSonar123/New-York-Stock-Exchange/blob/main/Sector%20wise%20Revenue%20trend%20over%20year.png)
![Dashboard Screenshot](https://github.com/RushiSonar123/New-York-Stock-Exchange/blob/main/Sector%20wise%20net%20income%20trend%20over%20years.png)
# (iii)Sector-wise patterns
```python


# Sample Data
data = {
    'Year': [2018, 2019, 2020, 2021, 2018, 2019, 2020, 2021],
    'Sector': ['Tech', 'Tech', 'Tech', 'Tech', 'Finance', 'Finance', 'Finance', 'Finance'],
    'Revenue': [100, 120, 130, 150, 80, 90, 95, 110]
}
df = pd.DataFrame(data)

# Ensure Year is numeric
df['Year'] = pd.to_numeric(df['Year'])

# Plot actual revenue trends per sector
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x='Year', y='Revenue', hue='Sector', marker='o')

# Add trendlines manually for each sector
for sector in df['Sector'].unique():
    sector_data = df[df['Sector'] == sector]
    z = np.polyfit(sector_data['Year'], sector_data['Revenue'], 1)
    p = np.poly1d(z)
    plt.plot(sector_data['Year'], p(sector_data['Year']), linestyle='--', label=f"{sector} Trend")

plt.title('Sector-wise Revenue Patterns Over Years')
plt.xlabel('Year')
plt.ylabel('Revenue')
plt.legend()
plt.show()
```
![Dashboard Screenshot](https://github.com/RushiSonar123/New-York-Stock-Exchange/blob/main/Sector%20wise%20Revenue%20patterns%20over%20years.png)
# 2 Time-Series Dataset Creation
# (i) Monthly average adjusted closing prices 

```python
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load Data and Preparation ---

# Load the prices-split-adjusted.csv file
try:
    df = pd.read_csv('prices-split-adjusted.csv')
except FileNotFoundError:
    print("Error: 'prices-split-adjusted.csv' not found. Please ensure the file is in the same directory.")
    exit()

# Convert 'date' to datetime object and set as index
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# --- 2. Calculate Monthly Average Close Price ---

# Calculate the daily average closing price across all stocks
daily_avg_close = df.groupby(df.index)['close'].mean()

# Resample the daily average to a monthly average ('M')
monthly_avg_close = daily_avg_close.resample('M').mean()

# --- 3. Display Data in Table Format ---

# Create a display-friendly version of the table
monthly_avg_df = monthly_avg_close.reset_index()
monthly_avg_df['date'] = monthly_avg_df['date'].dt.strftime('%Y-%m')
monthly_avg_df = monthly_avg_df.rename(columns={'close': 'Monthly Average Close Price'})

print("--- Monthly Average Adjusted Closing Prices ---")
print(monthly_avg_df)

# --- 4. Plot the Trend ---

plt.figure(figsize=(12, 6))

# Plot using the original monthly_avg_close Series with datetime index
plt.plot(monthly_avg_close.index, monthly_avg_close.values, linestyle='-', color='indigo', linewidth=2)

plt.title('Monthly Average Adjusted Closing Price Trend (2010-2016)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Closing Price ($)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show() # Use plt.show() for notebook
```
![Dashboard Screenshot](https://github.com/RushiSonar123/New-York-Stock-Exchange/blob/main/Monthly%20average%20Adjusted%20Closing%20Price%20Trends.png)
# (ii) Monthly revenue (using fundamentals)
```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_average_annual_revenue(file_path='fundamentals.csv'):
    """
    Loads the fundamentals data, calculates the average annual total revenue 
    across all companies, and plots the trend from 2012 onwards.
    
    Args:
        file_path (str): The path to the fundamentals.csv file.
    """
    try:
        # Load the fundamentals.csv file
        df_fund = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # Convert 'Period Ending' to datetime
    df_fund['Period Ending'] = pd.to_datetime(df_fund['Period Ending'])

    # Extract the year from 'Period Ending'
    df_fund['year'] = df_fund['Period Ending'].dt.year

    # Group by year and calculate the average Total Revenue across all companies
    # The result is automatically in the same unit as the source data (USD)
    avg_annual_revenue = df_fund.groupby('year')['Total Revenue'].mean().reset_index()
    
    # Convert Total Revenue to Billions of USD for better plot readability
    avg_annual_revenue['Total Revenue (Billions $)'] = avg_annual_revenue['Total Revenue'] / 1e9 

    # Filter out years with very low values (like 2003, 2004, 2006, 2007, 2017) 
    # that likely represent incomplete or sparse data, focusing on the main 2012-2016 range.
    avg_annual_revenue_filtered = avg_annual_revenue[avg_annual_revenue['year'] >= 2012]

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    # Plot the trend
    plt.plot(
        avg_annual_revenue_filtered['year'], 
        avg_annual_revenue_filtered['Total Revenue (Billions $)'], 
        marker='o', 
        linestyle='-', 
        color='firebrick',
        linewidth=2
    )

    plt.title('Average Annual Total Revenue Trend (Across All Companies)', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average Total Revenue (in Billions \$)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure only full years are shown on the x-axis
    plt.xticks(avg_annual_revenue_filtered['year'].unique())
    
    plt.tight_layout()
    plt.show()

# Call the function to run the analysis and plot the chart
plot_average_annual_revenue()
```
![Dashboard Screenshot](https://github.com/RushiSonar123/New-York-Stock-Exchange/blob/main/Average%20annual%20total%20revenue%20trend.png)

# 3. Trend & Seasonality

# (i) Use statsmodels or Prophet to detect seasonality in adjusted close prices

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# --- 1. Data Preparation: Calculate Monthly Average Close Price ---

try:
    # Load the prices-split-adjusted.csv file
    df = pd.read_csv('prices-split-adjusted.csv')
except FileNotFoundError:
    print("Error: 'prices-split-adjusted.csv' not found. Please ensure the file is in the same directory.")
    # Use plt.show() to prevent the code block from failing if run in a notebook without the file
    plt.show() 
    exit()

# Convert 'date' to datetime object and set as index
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# 1. Calculate the daily average closing price across all stocks
daily_avg_close = df.groupby(df.index)['close'].mean()

# 2. Resample the daily average to a monthly average ('M')
# We use the mean() for resampling to get the average price for the month
monthly_avg_close = daily_avg_close.resample('M').mean()

# --- 2. Time Series Decomposition ---

# We will decompose the time series using the statsmodels seasonal_decompose function.
# - model='multiplicative': Assumes that the seasonal component is proportional to the trend (common for economic data).
# - period=12: Looks for a seasonal pattern that repeats every 12 months (annual seasonality).

decomposition = seasonal_decompose(
    monthly_avg_close, 
    model='multiplicative', 
    period=12
)

# --- 3. Plotting the Decomposition Components ---

fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
plt.style.use('ggplot')

# Plot 1: Original Series
decomposition.observed.plot(ax=axes[0], color='blue', linewidth=2)
axes[0].set_title('Original Monthly Average Adjusted Close Price', fontsize=14)
axes[0].set_ylabel('Price ($)')

# Plot 2: Trend Component
decomposition.trend.plot(ax=axes[1], color='orange', linewidth=2)
axes[1].set_title('Trend Component', fontsize=14)
axes[1].set_ylabel('Price ($)')

# Plot 3: Seasonal Component
# We plot the seasonal factor (not the seasonal values)
decomposition.seasonal.plot(ax=axes[2], color='green', linewidth=2)
axes[2].set_title('Seasonal Component (Annual Cycle)', fontsize=14)
axes[2].set_ylabel('Seasonal Factor')

# Plot 4: Residual Component (Noise)
decomposition.resid.plot(ax=axes[3], color='purple', linewidth=2)
axes[3].set_title('Residual Component (Irregular/Noise)', fontsize=14)
axes[3].set_ylabel('Residual Factor')
axes[3].tick_params(axis='x', rotation=45)

plt.suptitle('Monthly Average Adjusted Close Price Decomposition (2010-2016)', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
```
![Dashboard Screenshot](https://github.com/RushiSonar123/New-York-Stock-Exchange/blob/main/Monthly%20avearage%20adjusted%20close%20price%20decomposition.png)

# (ii) Forecast next 12 months of adjusted price for top 5 companies

```python
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings

# Suppress Prophet specific warnings that can clutter the output
warnings.filterwarnings('ignore', category=UserWarning, module='prophet')

# --- 1. Data Loading and Top 5 Identification ---

try:
    # Load the prices-split-adjusted.csv file (contains daily data)
    df_all = pd.read_csv('prices-split-adjusted.csv')
except FileNotFoundError:
    print("Error: 'prices-split-adjusted.csv' not found. Please ensure the file is in the same directory.")
    # Show an empty plot just to fulfill the plotting requirement in case of error
    plt.figure(figsize=(10, 6))
    plt.title("Data File Not Found Error", color='red')
    plt.show() 
    exit()

# Convert 'date' to datetime object
df_all['date'] = pd.to_datetime(df_all['date'])

# Identify the top 5 companies by their average adjusted closing price
avg_price_by_symbol = df_all.groupby('symbol')['close'].mean()
top_5_symbols = avg_price_by_symbol.nlargest(5).index.tolist()

print(f"--- Forecasting for Top 5 Companies by Average Price: {top_5_symbols} ---")

# Dictionary to store the forecast results for plotting and the original data
forecast_results = {}
original_data = {} # New dictionary to store original data frames

# --- 2. Prophet Modeling and 12-Month Forecasting Loop ---

for symbol in top_5_symbols:
    # Filter for the current stock's daily data
    df_sym = df_all[df_all['symbol'] == symbol].copy()
    
    # Prepare data for Prophet: must have columns 'ds' (datetime) and 'y' (value)
    # Using 'date' as 'ds' and 'close' as 'y'
    prophet_df = df_sym[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
    
    # Store the original data for plotting actuals later
    original_data[symbol] = prophet_df 
    
    # Initialize Prophet model. We can set seasonality to multiplicative as stock price fluctuations are often proportional to the price level.
    m = Prophet(
        seasonality_mode='multiplicative', 
        yearly_seasonality=True,
        daily_seasonality=False, # Daily data should be sufficient for yearly patterns
        changepoint_prior_scale=0.05 # Default value for change points flexibility
    )
    
    # Fit the model
    m.fit(prophet_df)
    
    # Create a future DataFrame for the next 12 months (using Month End frequency 'M')
    # We add 12 periods to the end of the historical data
    future = m.make_future_dataframe(periods=12, freq='M') 
    
    # Make the forecast
    forecast = m.predict(future)
    
    # Store the result
    forecast_results[symbol] = forecast

# --- 3. Plotting the Forecast for All 5 Companies ---

plt.figure(figsize=(18, 9))
plt.style.use('seaborn-v0_8-whitegrid')

# Determine the start date for plotting historical data (e.g., last 3 years for clarity)
start_date = df_all['date'].max() - pd.DateOffset(years=3)

for symbol, forecast in forecast_results.items():
    # Plot the full fitted/forecasted line (yhat)
    # Filter the forecast for the last few historical points and the future points
    plot_data = forecast[forecast['ds'] >= start_date]

    # Plot the forecasted trend (yhat)
    plt.plot(plot_data['ds'], plot_data['yhat'], 
             label=f'{symbol} Trend & Forecast', 
             linestyle='-', 
             linewidth=2)
             
    # Plot the confidence interval as shaded region
    plt.fill_between(plot_data['ds'], plot_data['yhat_lower'], plot_data['yhat_upper'], 
                     alpha=0.1, label=f'{symbol} Uncertainty')

    # --- FIX: Use the original_data dictionary for plotting actuals ---
    # Retrieve the original, unfiltered Prophet data for the current symbol
    df_actuals = original_data[symbol]
    
    # Filter the original data only for the desired plotting range
    historical_data = df_actuals[df_actuals['ds'] >= start_date]

    plt.plot(historical_data['ds'], historical_data['y'], 
             marker='.', 
             linestyle='', 
             alpha=0.4, 
             # Only label the actuals once to avoid cluttering the legend
             label=f'{symbol} Actuals' if symbol == top_5_symbols[0] else '_nolegend_') 

plt.title('12-Month Adjusted Close Price Forecast for Top 5 Companies', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Adjusted Close Price ($)', fontsize=14)
# Draw a vertical line to mark the start of the forecast period
plt.axvline(x=df_all['date'].max(), color='black', linestyle='--', linewidth=1, label='Forecast Start')
plt.legend(loc='upper left', ncol=2, fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 4. Displaying Specific Forecast Data for the Top Company ---

top_symbol = top_5_symbols[0]
top_forecast = forecast_results[top_symbol]

# Filter to show only the 12 forecast months
future_forecast = top_forecast.iloc[-12:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
future_forecast = future_forecast.rename(columns={'yhat': 'Predicted Price', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'})

print(f"\n--- 12-Month Forecast Data for Top Company: {top_symbol} ---")
# Changed to_markdown() to to_string() to avoid the 'tabulate' dependency warning
print(future_forecast.to_string(index=False))
```
![Dashboard Screenshot](https://github.com/RushiSonar123/New-York-Stock-Exchange/blob/main/12-Months%20Adjusted%20close%20price%20forcast%20for%20top%205%20companies.png)
# Category-Level Forecasting
 

# *Revenue forecast by GICS Sector

```python
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings

# Suppress Prophet specific warnings that can clutter the output
warnings.filterwarnings('ignore', category=UserWarning, module='prophet')

# --- 1. Data Loading and Merging ---

try:
    df_fund = pd.read_csv('fundamentals.csv')
    df_sec = pd.read_csv('securities.csv')
except FileNotFoundError:
    print("Error: Required CSV file(s) not found.")
    plt.figure(figsize=(10, 6))
    plt.title("Data File Not Found Error", color='red')
    plt.show() 
    exit()

# Prepare Fundamentals Data
# Convert 'Period Ending' to datetime
df_fund['Period Ending'] = pd.to_datetime(df_fund['Period Ending'])
# Select relevant columns for revenue analysis
df_fund = df_fund[['Ticker Symbol', 'Period Ending', 'Total Revenue']].copy()

# Prepare Securities Data
# Rename column for merge compatibility and select relevant columns
df_sec = df_sec.rename(columns={'Ticker symbol': 'Ticker Symbol'})
df_sec = df_sec[['Ticker Symbol', 'GICS Sector']]

# Merge dataframes on Ticker Symbol
df_merged = pd.merge(df_fund, df_sec, on='Ticker Symbol', how='inner')

# --- 2. Sector-level Annual Revenue Aggregation ---

# Group by Sector and Annual Reporting Date, summing the Total Revenue
df_sector_revenue = df_merged.groupby(['GICS Sector', 'Period Ending'])['Total Revenue'].sum().reset_index()

# Convert Total Revenue to Billions for plotting and display clarity
df_sector_revenue['Total Revenue (Billion $)'] = df_sector_revenue['Total Revenue'] / 1e9

# --- 3. Identify Top 5 Sectors for Forecasting ---

# Calculate the average annual revenue for each sector to determine the top 5
avg_revenue_by_sector = df_sector_revenue.groupby('GICS Sector')['Total Revenue'].mean()
top_5_sectors = avg_revenue_by_sector.nlargest(5).index.tolist()

print(f"--- Forecasting Annual Revenue for Top 5 GICS Sectors: {top_5_sectors} ---")

forecast_results = {}
original_data = {}

# --- 4. Prophet Modeling and 5-Year Forecasting Loop ---

# Since revenue data is annual, we set the forecast period to 5 years (5 periods at 'Y' frequency)

for sector in top_5_sectors:
    # Filter data for the current sector
    df_sector = df_sector_revenue[df_sector_revenue['GICS Sector'] == sector].copy()
    
    # Prepare data for Prophet: ds (datetime) and y (value in Billions $)
    prophet_df = df_sector[['Period Ending', 'Total Revenue (Billion $)']].rename(columns={'Period Ending': 'ds', 'Total Revenue (Billion $)': 'y'})
    
    # Store original data for plotting actuals later
    original_data[sector] = prophet_df 
    
    # Initialize Prophet model. We disable seasonality as the input data is already annual.
    m = Prophet(
        seasonality_mode='multiplicative', 
        yearly_seasonality=False, # Data is annual, so no yearly cycle to model
        daily_seasonality=False, 
        changepoint_prior_scale=0.05 
    )
    
    m.fit(prophet_df)
    
    # Create a future DataFrame for the next 5 years (Year End frequency 'Y')
    future = m.make_future_dataframe(periods=5, freq='Y') 
    
    # Make the forecast
    forecast = m.predict(future)
    
    # Store the result
    forecast_results[sector] = forecast

# --- 5. Plotting the 5-Year Forecast for All 5 Sectors ---

plt.figure(figsize=(16, 8))
plt.style.use('seaborn-v0_8-whitegrid')

# Use the earliest date in the aggregated data for the plot start
start_date = df_sector_revenue['Period Ending'].min()

for sector, forecast in forecast_results.items():
    
    plot_data = forecast[forecast['ds'] >= start_date]

    # Plot the forecasted trend (yhat)
    plt.plot(plot_data['ds'], plot_data['yhat'], 
             label=f'{sector} Forecast', 
             linestyle='-', 
             linewidth=2)
             
    # Plot the confidence interval as shaded region
    plt.fill_between(plot_data['ds'], plot_data['yhat_lower'], plot_data['yhat_upper'], 
                     alpha=0.1)

    # Plot the actual historical revenue (dots)
    df_actuals = original_data[sector]
    historical_data = df_actuals[df_actuals['ds'] >= start_date]

    plt.plot(historical_data['ds'], historical_data['y'], 
             marker='o', 
             linestyle='', 
             alpha=0.6) 

plt.title('5-Year Annual Total Revenue Forecast by GICS Sector', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Total Annual Revenue (in Billions $)', fontsize=14)
# Draw a vertical line to mark the start of the forecast period
plt.axvline(x=df_sector_revenue['Period Ending'].max(), color='black', linestyle='--', linewidth=1, label='Forecast Start')
plt.legend(loc='upper left', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6. Displaying Specific Forecast Data for the Top Sector ---

top_sector = top_5_sectors[0]
top_forecast = forecast_results[top_sector]

# Filter to show only the 5 forecast years
future_forecast = top_forecast.iloc[-5:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
future_forecast = future_forecast.rename(columns={'yhat': 'Predicted Revenue (B$)', 'yhat_lower': 'Lower Bound (B$)', 'yhat_upper': 'Upper Bound (B$)'})
future_forecast['ds'] = future_forecast['ds'].dt.strftime('%Y')

print(f"\n--- 5-Year Annual Revenue Forecast Data for Top Sector: {top_sector} ---")
print("Note: Revenue is in Billions of USD.")
print(future_forecast.to_string(index=False))
```
![Dashboard Screenshot](https://github.com/RushiSonar123/New-York-Stock-Exchange/blob/main/5-Year%20annual%20total%20revenue%20forcast%20by%20GICS%20sector.png)

# 5 Visualizations

# (i) Line plots, trend lines, rolling averages

```python
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings

# Suppress Prophet specific warnings that can clutter the output
warnings.filterwarnings('ignore', category=UserWarning, module='prophet')

# --- 1. Data Loading and Merging ---

try:
    df_fund = pd.read_csv('fundamentals.csv')
    df_sec = pd.read_csv('securities.csv')
    # Load the price data to create a contextual plot
    df_prices = pd.read_csv('prices-split-adjusted.csv') 
except FileNotFoundError:
    print("Error: Required CSV file(s) not found.")
    plt.figure(figsize=(10, 6))
    plt.title("Data File Not Found Error", color='red')
    plt.show() 
    exit()

# Prepare Fundamentals Data
# Convert 'Period Ending' to datetime
df_fund['Period Ending'] = pd.to_datetime(df_fund['Period Ending'])
# Select relevant columns for revenue analysis
df_fund = df_fund[['Ticker Symbol', 'Period Ending', 'Total Revenue']].copy()

# Prepare Securities Data
# Rename column for merge compatibility and select relevant columns
df_sec = df_sec.rename(columns={'Ticker symbol': 'Ticker Symbol'})
df_sec = df_sec[['Ticker Symbol', 'GICS Sector']]

# Merge dataframes on Ticker Symbol for Revenue Analysis
df_merged = pd.merge(df_fund, df_sec, on='Ticker Symbol', how='inner')

# --- 2. Sector-level Annual Revenue Aggregation ---

# Group by Sector and Annual Reporting Date, summing the Total Revenue
df_sector_revenue = df_merged.groupby(['GICS Sector', 'Period Ending'])['Total Revenue'].sum().reset_index()

# Convert Total Revenue to Billions for plotting and display clarity
df_sector_revenue['Total Revenue (Billion $)'] = df_sector_revenue['Total Revenue'] / 1e9

# --- 3. Identify Top 5 Sectors for Forecasting (Based on Revenue) ---

# Calculate the average annual revenue for each sector to determine the top 5
avg_revenue_by_sector = df_sector_revenue.groupby('GICS Sector')['Total Revenue'].mean()
top_5_sectors = avg_revenue_by_sector.nlargest(5).index.tolist()

print(f"--- Forecasting Annual Revenue for Top 5 GICS Sectors: {top_5_sectors} ---")

forecast_results = {}
original_data = {}

# --- 4. Prophet Modeling and 5-Year Forecasting Loop ---

# Since revenue data is annual, we set the forecast period to 5 years (5 periods at 'Y' frequency)

for sector in top_5_sectors:
    # Filter data for the current sector
    df_sector = df_sector_revenue[df_sector_revenue['GICS Sector'] == sector].copy()
    
    # Prepare data for Prophet: ds (datetime) and y (value in Billions $)
    prophet_df = df_sector[['Period Ending', 'Total Revenue (Billion $)']].rename(columns={'Period Ending': 'ds', 'Total Revenue (Billion $)': 'y'})
    
    # Store original data for plotting actuals later
    original_data[sector] = prophet_df 
    
    # Initialize Prophet model. We disable seasonality as the input data is already annual.
    m = Prophet(
        seasonality_mode='multiplicative', 
        yearly_seasonality=False, # Data is annual, so no yearly cycle to model
        daily_seasonality=False, 
        changepoint_prior_scale=0.05 
    )
    
    m.fit(prophet_df)
    
    # Create a future DataFrame for the next 5 years (Year End frequency 'Y')
    future = m.make_future_dataframe(periods=5, freq='Y') 
    
    # Make the forecast
    forecast = m.predict(future)
    
    # Store the result
    forecast_results[sector] = forecast

# --- 5. Plotting the 5-Year Revenue Forecast for All 5 Sectors (Plot 1) ---

plt.figure(figsize=(16, 8))
plt.style.use('seaborn-v0_8-whitegrid')

# Use the earliest date in the aggregated data for the plot start
start_date = df_sector_revenue['Period Ending'].min()

for sector, forecast in forecast_results.items():
    
    plot_data = forecast[forecast['ds'] >= start_date]

    # Plot the forecasted trend (yhat)
    plt.plot(plot_data['ds'], plot_data['yhat'], 
             label=f'{sector} Forecast', 
             linestyle='-', 
             linewidth=2)
             
    # Plot the confidence interval as shaded region
    plt.fill_between(plot_data['ds'], plot_data['yhat_lower'], plot_data['yhat_upper'], 
                     alpha=0.1)

    # Plot the actual historical revenue (dots)
    df_actuals = original_data[sector]
    historical_data = df_actuals[df_actuals['ds'] >= start_date]

    plt.plot(historical_data['ds'], historical_data['y'], 
             marker='o', 
             linestyle='', 
             alpha=0.6) 

plt.title('Plot 1: 5-Year Annual Total Revenue Forecast by GICS Sector', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Total Annual Revenue (in Billions $)', fontsize=14)
# Draw a vertical line to mark the start of the forecast period
plt.axvline(x=df_sector_revenue['Period Ending'].max(), color='black', linestyle='--', linewidth=1, label='Forecast Start')
plt.legend(loc='upper left', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6. Plotting Average Price Trend for Top 5 Revenue Sectors (Plot 2 - New) ---

# Prepare price data by merging with sector information
df_prices_sec = df_prices.rename(columns={'symbol': 'Ticker Symbol'})
df_prices_merged = pd.merge(df_prices_sec, df_sec, on='Ticker Symbol', how='inner')

# Filter for only the top 5 revenue sectors
df_prices_top_sectors = df_prices_merged[df_prices_merged['GICS Sector'].isin(top_5_sectors)].copy()
df_prices_top_sectors['date'] = pd.to_datetime(df_prices_top_sectors['date'])

# Calculate the daily average adjusted close price for each of the top sectors
df_sector_price_avg = df_prices_top_sectors.groupby(['date', 'GICS Sector'])['close'].mean().reset_index()

plt.figure(figsize=(16, 8))
plt.style.use('seaborn-v0_8-whitegrid')

# Plot the average price trend for each of the top 5 sectors
for sector in top_5_sectors:
    df_plot = df_sector_price_avg[df_sector_price_avg['GICS Sector'] == sector]
    plt.plot(df_plot['date'], df_plot['close'], label=sector, linewidth=1.5)

plt.title('Plot 2: Average Daily Adjusted Close Price Trend for Top 5 Revenue Sectors', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Average Adjusted Close Price ($)', fontsize=14)
plt.legend(loc='upper left', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()


# --- 7. Displaying Specific Forecast Data for the Top Sector ---

top_sector = top_5_sectors[0]
top_forecast = forecast_results[top_sector]

# Filter to show only the 5 forecast years
future_forecast = top_forecast.iloc[-5:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
future_forecast = future_forecast.rename(columns={'yhat': 'Predicted Revenue (B$)', 'yhat_lower': 'Lower Bound (B$)', 'yhat_upper': 'Upper Bound (B$)'})
future_forecast['ds'] = future_forecast['ds'].dt.strftime('%Y')

print(f"\n--- 5-Year Annual Revenue Forecast Data for Top Sector: {top_sector} ---")
print("Note: Revenue is in Billions of USD.")
print(future_forecast.to_string(index=False))
```
![Dashboard Screenshot](https://github.com/RushiSonar123/New-York-Stock-Exchange/blob/main/Average%20Daily%20adjusted%20close%20price%20trend%20for%20top%205%20revenue%20sector.png)

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from prophet import Prophet
import warnings

# Suppress Prophet specific warnings that can clutter the output
warnings.filterwarnings('ignore', category=UserWarning, module='prophet')

# --- 1. Data Loading and Merging ---

try:
    df_fund = pd.read_csv('fundamentals.csv')
    df_sec = pd.read_csv('securities.csv')
    # Load the price data to create a contextual plot
    df_prices = pd.read_csv('prices-split-adjusted.csv') 
except FileNotFoundError:
    print("Error: Required CSV file(s) not found.")
    plt.figure(figsize=(10, 6))
    plt.title("Data File Not Found Error", color='red')
    plt.show() 
    exit()

# Prepare Fundamentals Data
# Convert 'Period Ending' to datetime
df_fund['Period Ending'] = pd.to_datetime(df_fund['Period Ending'])
# Select relevant columns for revenue analysis
df_fund = df_fund[['Ticker Symbol', 'Period Ending', 'Total Revenue']].copy()

# Prepare Securities Data
# Rename column for merge compatibility and select relevant columns
df_sec = df_sec.rename(columns={'Ticker symbol': 'Ticker Symbol'})
df_sec = df_sec[['Ticker Symbol', 'GICS Sector']]

# Merge dataframes on Ticker Symbol for Revenue Analysis
df_merged = pd.merge(df_fund, df_sec, on='Ticker Symbol', how='inner')

# --- 2. Sector-level Annual Revenue Aggregation ---

# Group by Sector and Annual Reporting Date, summing the Total Revenue
df_sector_revenue = df_merged.groupby(['GICS Sector', 'Period Ending'])['Total Revenue'].sum().reset_index()

# Convert Total Revenue to Billions for plotting and display clarity
df_sector_revenue['Total Revenue (Billion $)'] = df_sector_revenue['Total Revenue'] / 1e9

# --- 3. Identify Top 5 Sectors for Forecasting (Based on Revenue) ---

# Calculate the average annual revenue for each sector to determine the top 5
avg_revenue_by_sector = df_sector_revenue.groupby('GICS Sector')['Total Revenue'].mean()
top_5_sectors = avg_revenue_by_sector.nlargest(5).index.tolist()

print(f"--- Calculating Predicted Growth for Top 5 GICS Sectors: {top_5_sectors} ---")

forecast_results = {}
original_data = {}

# --- 4. Prophet Modeling and 5-Year Forecasting Loop ---

# Since revenue data is annual, we set the forecast period to 5 years (5 periods at 'Y' frequency)
last_historical_date = df_sector_revenue['Period Ending'].max()

for sector in top_5_sectors:
    # Filter data for the current sector
    df_sector = df_sector_revenue[df_sector_revenue['GICS Sector'] == sector].copy()
    
    # Prepare data for Prophet: ds (datetime) and y (value in Billions $)
    prophet_df = df_sector[['Period Ending', 'Total Revenue (Billion $)']].rename(columns={'Period Ending': 'ds', 'Total Revenue (Billion $)': 'y'})
    
    # Store original data for plotting actuals later
    original_data[sector] = prophet_df 
    
    # Initialize Prophet model. We disable seasonality as the input data is already annual.
    m = Prophet(
        seasonality_mode='multiplicative', 
        yearly_seasonality=False, # Data is annual, so no yearly cycle to model
        daily_seasonality=False, 
        changepoint_prior_scale=0.05 
    )
    
    m.fit(prophet_df)
    
    # Create a future DataFrame for the next 5 years (Year End frequency 'Y')
    future = m.make_future_dataframe(periods=5, freq='Y') 
    
    # Make the forecast
    forecast = m.predict(future)
    
    # Store the result
    forecast_results[sector] = forecast


# --- 7. Calculate and Plot Predicted Growth Heatmap/Bar Chart (Plot 3 - NEW) ---

sector_growth = {}

for sector, forecast_df in forecast_results.items():
    
    # 1. Get the predicted revenue for the last historical year (base of forecast)
    start_yhat_row = forecast_df[forecast_df['ds'] == last_historical_date]
    
    if not start_yhat_row.empty:
        start_revenue_yhat = start_yhat_row['yhat'].iloc[0]
    else:
        # Fallback to the last actual historical value if the prediction point is missed
        start_revenue_yhat = original_data[sector]['y'].iloc[-1]
        
    # 2. Get the predicted revenue for the end of the 5-year forecast
    end_revenue_yhat = forecast_df['yhat'].iloc[-1]
    
    # 3. Calculate 5-Year Simple Percentage Growth
    if start_revenue_yhat > 0:
        growth_rate = ((end_revenue_yhat - start_revenue_yhat) / start_revenue_yhat) * 100
    else:
        growth_rate = 0 # Cannot calculate growth from zero or negative base
        
    sector_growth[sector] = growth_rate

# Convert to DataFrame for plotting and display
df_growth = pd.DataFrame(sector_growth.items(), columns=['GICS Sector', 'Predicted 5-Year Growth (%)'])
df_growth = df_growth.sort_values(by='Predicted 5-Year Growth (%)', ascending=False)

# --- Plotting the Heatmap/Bar Chart (Plot 3) ---

plt.figure(figsize=(10, 6))
# Create colors based on the growth rate (using a gradient for 'heatmap' effect)
# Normalize growth rates to the colormap range (0 to 1)
min_growth = df_growth['Predicted 5-Year Growth (%)'].min()
max_growth = df_growth['Predicted 5-Year Growth (%)'].max()

if max_growth > min_growth:
    normalized_growth = (df_growth['Predicted 5-Year Growth (%)'] - min_growth) / (max_growth - min_growth)
else:
    normalized_growth = [0.5] * len(df_growth) # Default color if no difference

# Use Red-Yellow-Green colormap for growth (Green=High, Red=Low)
colors = cm.RdYlGn(normalized_growth)

plt.bar(df_growth['GICS Sector'], df_growth['Predicted 5-Year Growth (%)'], color=colors)

plt.title('Plot 3: Predicted 5-Year Revenue Growth by GICS Sector', fontsize=16)
plt.ylabel('Predicted Growth Rate (%)', fontsize=12)
plt.xlabel('GICS Sector', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels on top of the bars
for i, row in df_growth.reset_index(drop=True).iterrows():
    # Adjust position slightly based on positive/negative
    y_pos = row['Predicted 5-Year Growth (%)']
    if y_pos >= 0:
        y_label_pos = y_pos + 0.5
    else:
        y_label_pos = y_pos - 1.5

    plt.text(i, y_label_pos, 
             f"{row['Predicted 5-Year Growth (%)']:.1f}%", 
             ha='center', 
             fontsize=10,
             weight='bold')

plt.tight_layout()
plt.show()

```
![Dashboard Screenshot](https://github.com/RushiSonar123/New-York-Stock-Exchange/blob/main/Predicted%205%20year%20growth%20by%20GICS%20sector.png)
