import pandas as pd
import arrow
import numpy as np
from signals import SECONDS_2_TIMESTAMP


RTK_COLUMNS = [
    'date',
    'time',
    'lat',
    'lng', 
    'height',
    'Q', 
    'ns',
    'sdn',
    'sde',
    'sdu',
    'sdne',
    'sdeu',
    'sdun',
    'age',
    'ratio',
    'vn',
    've',
    'vu',
    'sdvn',
    'sdve',
    'sdvu',
    'sdvne',
    'sdveu',
    'sdvun'
]

def isodate_2_epoch(iso_date):
    """
    Converts ISO 8601 formatted date & time to nanoseconds since epoch.

    Parameters
    ----------
    iso_date : str
        ISO 8601 formatted date and time

    Returns
    -------
    np.int64
        nanosecond epoch timestamp corresponding to the input ISO date & time

    """
    return np.int64(arrow.get(iso_date, 'YYYY/MM/DDTHH:mm:ss.S').float_timestamp * SECONDS_2_TIMESTAMP)


def read_rtk_solution(file_path):
    """
    Reads a RTKLIB solution file to a Pandas DataFrame. Each row is one RTK solution, which is corresponds to one GNSS fix.

    The DataFrame has the following columns:

    - date: 
        Date of the GNSS fix in YYYY/MM/DD format

    - time: 
        Time of the GNSS fix in HH:mm:ss.SSS format 

    - timestamp:
        Date & time converted to nanoseconds since epoch

    - lat: 
        Latitutde (WGS84) 

    - lng: 
        Longitude (WGS84)

    - height: 
        Ellipsoidal height

    - Q: 
        Quality of the RTK solution (
            1 = Fixed, solution by carrier‐based relative positioning and the integer ambiguity is properly resolved., 
            2 = Float, solution by carrier‐based relative positioning but the integer ambiguity is not resolved. (i.e. centimeter accuracy not guaranteed)
        )

    - ns: 
        The number of valid satellites for solution estimation

    - Standard deviations (sdn, sde, sdu, sdne, sdeu, sdun):
        The estimated standard deviations of the solution assuming a priori error model and error parameters by the positioning options.
        The sdn, sde or sdu means N (north), E (east) or U (up) component of the standard deviations in m. The absolute value of
        sdne, sdeu or sdun means square root of the absolute value of NE, EU or UN component of the estimated covariance matrix. The sign
        represents the sign of the covariance. With all of the values, user can reconstruct the full covariance matrix.

    - age: 
        The time difference between the observation data epochs of the rover receiver and the base station in second.

    - ratio: 
        The ratio factor of "ratio‐test" for standard integer ambiguity validation strategy. The value means the ratio of the squared sum of
        the residuals with the second best integer vector to with the best integer vector.

    - Velocity (vn, ve, vu):
        The estimated receiver velocity in directions North, Easth & Up

    - Velocity standard deviations (sdvn, sdve, sdvu, sdvne, sdveu, sdvun):
        The estimated standard devations of the receiver velocity
    

    Parameters
    ----------
    file_path : str
        The path to the solutions file

    Returns
    -------
    DataFrame
        Pandas DataFrame with the following columns
    """

    
    df = pd.read_csv(
        file_path, 
        delim_whitespace=True, # Remove duplicate spaces between columns
        comment='%', # Remove header comments
        header=None, 
        names=RTK_COLUMNS, 
        index_col=False
    )
    
    # Convert date & time column to nanosecond epoch timestamp
    df['timestamp'] = df.apply(lambda x: isodate_2_epoch(x.date + 'T' + x.time), axis=1)

    return df


