#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import sys
import os
import sklearn
import json

sklearn.__version__

year = int(sys.argv[1]) # 2023
month = int(sys.argv[2]) # 3
output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'

os.makedirs('output', exist_ok=True)

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

df = read_data(input_file)

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

file_size_M = os.path.getsize(output_file) / 1024 / 1024

with open('Pipfile.lock', 'r') as pipfile:
    scikit_hash = json.load(pipfile)["default"]["scikit-learn"]["hashes"][0]

print(f'Q1 Standard deviation for {year:04d}-{month:02d}: {round(y_pred.std(), 2)}')
print(f'Q2 Size of the output parquet file: {round(file_size_M, 2)}')
print(f'Q3 Command to turn the notebook into a script: jupyter nbconvert --to script starter.ipynb')
print(f'Q4 First hash for the Scikit-Learn dependency: {scikit_hash}')
print(f'Q5 Mean predicted duration for {year:04d}-{month:02d}: {round(y_pred.mean(), 2)}')

