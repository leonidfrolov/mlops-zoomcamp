import pandas as pd
from datetime import datetime
import batch
import os

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def prepare_data():
    test_data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']

    df = pd.DataFrame(test_data, columns=columns)
    return df

def main(year, month):
    os.environ['INPUT_FILE_PATTERN'] = f"s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
    storage_options = batch.get_storage_options()
    input_file = batch.get_input_path(year, month)
    df = prepare_data()
    batch.save_data(input_file, df, storage_options)
    print(f"Q3 Dataframe: \n{df}")

if __name__ == "__main__":
    main(2023, 1)
