from flask import Flask, request, render_template, redirect
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from datetime import datetime
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
import scipy.stats as stat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from main import *
warnings.filterwarnings('ignore')

app = Flask(__name__)


@app.route('/', methods=['GET'])
def Home():
    Pred = trainmodel()
    weekday = ['Monday', 'Tuesday', 'Wednesday',
               'Thursday', 'Friday', 'Saturday']
    day = []
    month = []
    year = []
    for item in unique_categories:
        for key, value in item.items():
            if key == 'day':
                day.extend(value)
            elif key == 'month':
                month.extend(value)
            elif key == 'year':
                year.extend(value)
    columns = ['Liter_size', 'Bottle_qty']
    context = {
        'Weekday': weekday,
        'Day': day,
        'Month': month,
        'Year': year
    }
    return render_template('index.html', context=context, columns=columns)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        weekday = request.form['Weekday']
        liter_size = int(request.form['Liter_size'])
        bottle_qty = int(request.form['Bottle_qty'])
        day = int(request.form['Day'])
        month = int(request.form['Month'])
        year = int(request.form['Year'])
        data = pd.DataFrame([[weekday, liter_size, bottle_qty, day, month, year]], columns=[
                            'weekday', 'liter_size', 'bottle_qty', 'day', 'month', 'year'])
        prediction = testmodel(data)[0]
        prediction = round(prediction, 1)
        return render_template('predict.html', prediction_text=f'The total sales generated from the lot is $ {prediction}')


if __name__ == '__main__':
    app.run(debug=True)
