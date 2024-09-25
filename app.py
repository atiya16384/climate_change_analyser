from flask import Flask, render_template, Response
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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