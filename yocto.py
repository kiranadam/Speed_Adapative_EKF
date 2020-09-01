import pandas as pd
import numpy as np
import os
from signals import SECONDS_2_TIMESTAMP
    

YOCTO_FILE_PREFIX = 'yocto.'
YOCTO_FILE_EXTENSION = 'csv'
SENSOR_NAMES = [
    'accelerometer.x',
    'accelerometer.y',
    'accelerometer.z',
    'gyro.x',
    'gyro.y',
    'gyro.z',
    'compass',
]
YOCTO_TIMESTAMP_OFFSET = 18 * SECONDS_2_TIMESTAMP


def read_sensors(data_dir):

    sensors = {}
    for name in SENSOR_NAMES:

        # Generating sensor file path    
        file_name = YOCTO_FILE_PREFIX + name + '.' + YOCTO_FILE_EXTENSION
        file_path = os.path.join(data_dir, file_name)


        # Reading sensor data
        sensor_df = pd.read_csv(file_path, names=['timestamp', 'value'])

        # Filter out missing & NaN values
        sensor_df = sensor_df[(~sensor_df.timestamp.isnull()) & (~sensor_df.value.isnull())].reset_index(drop=True)

        sensor_df.loc[:, 'timestamp'] = (sensor_df.loc[:, 'timestamp'] * SECONDS_2_TIMESTAMP).astype(np.int64)

        # There is a timing sync issue with the Yocto data, fixing the difference in timing here.
        sensor_df.loc[:, 'timestamp'] += YOCTO_TIMESTAMP_OFFSET
        
        sensors[name] = sensor_df

    return sensors

