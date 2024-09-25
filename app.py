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
# Load the data
temperature_read_csv = pd.read_csv('GlobalLandTemperaturesByCity.csv')

# we convert the 'dt' time to datetime format so that we can use it for time series analysis
temperature_read_csv['dt'] = pd.to_datetime(temperature_read_csv['dt'])

# we extract the year out from the data column
temperature_read_csv['year'] = temperature_read_csv['dt'].dt.year

# we group each of the years together and calculate the respective means
global_temp = temperature_read_csv.groupby('year')['AverageTemperature'].mean()

# we reset the index so that we can use the year as a column
global_temp = global_temp.reset_index()

# we remove the rows with missing values
global_temp = global_temp.dropna()

sns.set(style="whitegrid")

# Set the figure size
plt.figure(figsize=(14, 8))
plt.title('Global Average Temperature Over Time')
plt.xlabel('Year')
plt.ylabel('Average Temperature')
plt.plot(global_temp['year'], global_temp['AverageTemperature'], color='blue', marker='o')
plt.show()

# we prepare the data for linear regression
X = global_temp['year'].values.reshape(-1, 1)
y = global_temp['AverageTemperature'].values

# we split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# create a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# predict temperature anomalies for future years
# we conver the range of years to a nump array before we can use it for prediction- that will get rid of possible errors/warnings in console.
future_years = pd.DataFrame(np.arange(2012, 2041), columns=['year'])  # 'year' should be 'Year' for consistency
future_predictions = model.predict(future_years)

# helper function to convert plots to base64 PNGs
def plot_to_img_tag(plt):
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# define the home route for the app
@app.route('/')
def home():
    # plot global average temperature over time
    plt.figure(figsize=(10, 5))
    sns.lineplot(global_temp['year'], global_temp['AverageTemperature'], label='Historical', color='blue')
    sns.lineplot(x=future_years['year'], y=future_predictions, label= 'Predicted',linestyle='--') 

    # plot the future predictions
    plt.title('Global Average Temperature Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average Temperature (°C)')
    plt.show()

    # save the plot to a base64-encoded image to embed in the HTML
    img_tag = plot_to_img_tag(plt)
    # Render the HTML template with the plot embedded
    return render_template('index.html', plot_img=img_tag)

if __name__ == '__main__':
    app.run(debug=True)
    