from flask import Flask, render_template, Response
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)
# first we want to load and preprocess the data for C02_emissions.csv and GlobalLandTemperaturesByCountry.csv files

def load_and_process_data():
    # we do this first for the co2 emissions data
    temperature_read_csv = pd.read_csv('GlobalLandTemperaturesByCity.csv')
     # we convert the 'dt' time to datetime format so that we can use it for time series analysis
    temperature_read_csv['dt'] = pd.to_datetime(temperature_read_csv['dt'])
    # we extract the year out from the data column
    temperature_read_csv['Year'] = temperature_read_csv['dt'].dt.year
    # we group each of the years together and calculate the respective means
    global_temp = temperature_read_csv.groupby('Year')['AverageTemperature'].mean()
    # we reset the index so that we can use the year as a column
    global_temp = global_temp.reset_index()
    # we remove the rows with missing values
    global_temp = global_temp.dropna()

    # we do this for the co2 emissions data
    c02_data = pd.read_csv('CO2_emission.csv')
    co2_data_melted = pd.melt(c02_data, 
                          id_vars=['Country Name', 'country_code', 'Region', 'Indicator Name'],
                          var_name='Year', 
                          value_name='CO2_Emissions')
    # As we have multiple types of indicators, we only want to choose 1 so that we don't use mix together different types of data
    co2_data_melted = co2_data_melted[co2_data_melted['Indicator Name'] == 'CO2 emissions (metric tons per capita)']

    # we convert the 'Year' column to integer, as the current type of the data is string
    co2_data_melted['Year'] = pd.to_numeric(co2_data_melted['Year'])

    # first drop any missing values in C02 emissions
    co2_data_melted = co2_data_melted.dropna(subset = ['CO2_Emissions'])

    # we group the data by year and calculate the average CO2 emissions for each year
    co2_data_grouped = co2_data_melted.groupby('Year')['CO2_Emissions'].mean().reset_index()

    return global_temp, co2_data_grouped


# Helper function to create base64-encoded images for plots
def plot_to_base64(plt):
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()


# Helper function to perform linear regression and predict future values

def perform_regression_and_predict(df, target_col, year_col='Year', start_year = 2021, end_year = 2050):
    # Split the data into training and testing sets
    X= df[year_col].values.reshape(-1,1)
    y = df[target_col].values

    # fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # predict future values
    future_years = np.arange(start_year, end_year).reshape(-1,1)
    future_predictions = model.predict(future_years)
    
    return future_years, future_predictions

# flask route for the home page
@app.route('/')
def home():
    co2_data, temp_data = load_and_process_data()

    # we use linear regression for both of them to predict data

    # linear regression for c02 emissions
    co2_future_years, co2_future_predictions = perform_regression_and_predict(co2_data, 'CO2_Emissions')

    # linear regression for temperature data
    temp_future_years, temp_future_predictions = perform_regression_and_predict(temp_data, 'AverageTemperature')

    # Create CO2 emissions plot with predictions
    plt.figure(figsize=(10,6))
    plt.plot(co2_data['Year'], co2_data['CO2_Emissions'], label='CO2 Emissions')
    plt.plot(co2_future_years, co2_future_predictions, label='Future Predictions')
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions (metric tons per capita)')
    plt.title('CO2 Emissions Over Time')
    plt.legend()
    co2_plot = plot_to_base64(plt)


    # Create temperature plot with predictions
    plt.figure(figsize=(10,6))
    plt.plot(temp_data['Year'], temp_data['AverageTemperature'], label='Average Temperature')
    plt.plot(temp_future_years, temp_future_predictions, label='Future Predictions')
    plt.xlabel('Year')
    plt.ylabel('Average Temperature')
    plt.title('Global Average Temperature Over Time')
    plt.legend()
    temp_plot = plot_to_base64(plt)

    return render_template('index.html', co2_plot=co2_plot, temp_plot=temp_plot)


if __name__ == '__main__':
    app.run(debug=True)