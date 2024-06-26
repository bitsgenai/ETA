from electricity_predictor import ElectricityPredictor, read_csv
import pandas as pd
import numpy as np

# Read the CSV file
df = read_csv('new_consumption.csv')
df.sort_values("contract", inplace=True)
df.drop_duplicates(subset="contract",
                     keep=False, inplace=True)
rawData = pd.read_csv('balanced_data.csv')
rawData.sort_values("contract", inplace=True)

# dropping ALL duplicate values
rawData.drop_duplicates(subset="contract",
                     keep=False, inplace=True)

df.replace('', np.nan, inplace=True)
df_merged = df.merge(rawData, on='contract', how='right')
df_merged = df_merged.rename(columns = {'fraud_consumption_x':'fraud_consumption'})
required_columns_model = ['contract','invoice_type','billing_type', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'fraud_consumption',
                   'SERVICE_STATUS', 'POWER_SUSCRIBED', 'TARIFF', 'ACTIVITY_CMS', 'READWITH', 'SEGMENT',
                   'agency', 'zone','block']
df_merged.replace('', np.nan, inplace=True)
df_merged = df_merged[required_columns_model]
df_merged.set_index('contract', inplace = True)


# Initialize the predictor
prediction = ElectricityPredictor()

# Fit the model (you might want to fit on historical normal data)
#predictor.fit(data)

# Predict new data
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
