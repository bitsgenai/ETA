from flask import Flask, render_template, request, send_file, jsonify
from electricity_predictor import ElectricityPredictor, read_csv
import pandas as pd
import numpy as np

app = Flask(__name__)
#test

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/single-customer')
def single_customer():
    return render_template('index.html')

@app.route('/predict_one_customer',methods=['POST'])
def predict_one_customer():
    '''
    For rendering results on HTML GUI
    '''

    int_features = [ x for x in request.form.values()]

    contract_no = int(int_features[0])
    new_consumption = float(int_features[1])
  

    rawData = pd.read_csv('balanced_data.csv')


    rawData=rawData.loc[(rawData['contract']==contract_no)]

    rawData['fraud_consumption'] = new_consumption

    print("rawData : ",rawData)
    required_columns_model = ['contract','invoice_type','billing_type', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'fraud_consumption',
                   'SERVICE_STATUS', 'POWER_SUSCRIBED', 'TARIFF', 'ACTIVITY_CMS', 'READWITH', 'SEGMENT',
                   'agency', 'zone','block']
    rawData.replace('', np.nan, inplace=True)
    df1 = rawData[required_columns_model]
    df1.set_index('contract', inplace = True)

    prediction = ElectricityPredictor()
    predictions = prediction.predict(df1)
    for i in range(len(predictions)):


        if predictions[i] == 'normal':
            predictions[i] = 1

        if predictions[i] == 'abnormal':
            predictions[i] = 0

    pred = [round(value) for value in predictions]
    df1['prediction'] = pred

    output = round(pred[0], 0)
    
    if(output==0):
        res="Normal Consumption"
    else:
        res="Fraud Consumption"

    return render_template('index.html', prediction_text='PREDICTION IS {}'.format(res))


@app.route('/multiple-customers')
def multiple_customers():
    return render_template('csv_input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('csv_input.html', error="No file found")
    
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return render_template('csv_input.html', error="Only CSV files are supported")
    df = pd.read_csv(file)
    df.sort_values("contract", inplace=True)

    # dropping ALL duplicate values
    df.drop_duplicates(subset="contract",
                     keep=False, inplace=True)


    rawData = pd.read_csv('balanced_data.csv')
    rawData.sort_values("contract", inplace=True)

    # dropping ALL duplicate values
    rawData.drop_duplicates(subset="contract",
                     keep=False, inplace=True)


    df.replace('', np.nan, inplace=True)
    #df.set_index('contract', inplace = True)

    df_merged = df.merge(rawData, on='contract', how='right')
    df_merged = df_merged.rename(columns = {'fraud_consumption_x':'fraud_consumption'})


    
    required_columns_model = ['contract','invoice_type','billing_type', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'fraud_consumption',
                   'SERVICE_STATUS', 'POWER_SUSCRIBED', 'TARIFF', 'ACTIVITY_CMS', 'READWITH', 'SEGMENT',
                   'agency', 'zone','block']
    df_merged.replace('', np.nan, inplace=True)
    df_merged = df_merged[required_columns_model]
    df_merged.set_index('contract', inplace = True)


    
    prediction = ElectricityPredictor()
    predictions = prediction.predict(df_merged)
    for i in range(len(predictions)):


        if predictions[i] == 'normal':
            predictions[i] = 1

        if predictions[i] == 'abnormal':
            predictions[i] = 0

    pred = [round(value) for value in predictions]
    df_merged['prediction'] = pred

    output_filename = 'predictions.csv'
    df_merged.to_csv(output_filename, index=True)
    return send_file(output_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
