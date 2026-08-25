import pickle
import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## Interact with Ridge.pkl (Prediction) and Standard.pkl (Transformation)
ridge_model = pickle.load(open('ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('scaler.pkl', 'rb'))

## Create Home Page

@app.route('/')
def index():
    return render_template('index.html') # render_template go find the index.html file in templates folder

@app.route('/predictdata', methods=['GET','POST'])
def predict_fwi():
    if request.method == 'POST':
        Temperature = float(request.form['Temperature'])    
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])

        new_scaled_data = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_scaled_data)
        return render_template('home.html', results=result[0])
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2000)
