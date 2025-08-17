import pandas as pd
from pandas.testing import assert_frame_equal
from datetime import datetime
import batch

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():

    test_data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    expected_data = [
        (None, None, dt(1, 1), dt(1, 10), 9.000000),
        (1, 1, dt(1, 2), dt(1, 10), 8.000000),
    ]

    categorical = ['PULocationID', 'DOLocationID']
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    expected_columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration']

    df = pd.DataFrame(test_data, columns=columns)
    actual_df = batch.prepare_data(df, categorical)

    expected_df = pd.DataFrame(expected_data, columns=expected_columns)
    expected_df[categorical] = expected_df[categorical].fillna(-1).astype('int').astype('str')

    assert_frame_equal(actual_df, expected_df)
