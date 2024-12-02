import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport

from data_quality import DataQuality


def main():
    csv_file_path = 'archive.zip'

    df = read_csv_file(csv_file_path)
    dq_engine = DataQuality(df)

    #data validation
    validate_missing_values(dq_engine, df)
    validate_duplicates(dq_engine, df)
    validate_data_types(dq_engine)
 
    #data transformation
    add_sample_datetime_column(df)
    add_transaction_type_column(df)
    
    #aggregations
    aggregation_funcs_1 = {
        'amount': ['mean', 'sum'],
        'type': 'nunique'
    }
    aggregate_data(df, 'H', aggregation_funcs_1)

    aggregation_funcs_2 = {
        'transaction_type': 'nunique'
    }
    aggregate_data(df, 'D', aggregation_funcs_2)

    aggregation_funcs_3 = {
        'isFraud': 'sum',
        'isFlaggedFraud': 'sum'
    }
    aggregate_data(df, 'W', aggregation_funcs_3)

    #profiling report
    profiling(df, 0.2, stratified_sample='isFraud')


def read_csv_file(file_path:str) -> pd.DataFrame:
    '''
    Reads a compressed CSV file from the specified file path and returns it as a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file, which should be compressed in a ZIP format.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the data from the CSV file.
    '''
    df = pd.read_csv(file_path, compression='zip', header=0, sep=',')
    print(f'Reading data from {file_path} \nSample data: \n {df.head()}')

    return df


def validate_missing_values(dq_engine: object, df: pd.DataFrame, delete_missing_value: str = True) -> None:
    '''
    This function checks each column in the given DataFrame for missing values using the `dq_engine`.
    It prints the number of missing values in each column, and if there are missing values and the 
    `delete_missing_value` flag is set to `True`, it removes the rows with missing values from the 
    specified columns.

    Parameters:
    dq_engine (object): A data quality engine object that provides a method `check_missing_values()`.
        This method is expected to return a dictionary with columns as keys and missing values count as values.
    df (pd.DataFrame): The DataFrame to validate and clean.
    delete_missing_value (bool, optional): A flag to determine whether to remove rows with missing 
        values (default is `True`). If `False`, missing values are reported but not removed.

    Returns:
    None: This function modifies the DataFrame in place and does not return a value.
    '''
    for column in df.columns:
        missing_values = dq_engine.check_missing_values()[column]
        print(f'Column: {column}, number of missing values: {missing_values}')

        if missing_values > 0 and delete_missing_value:
            df.dropna(subset=[column], inplace=True)
            print('Missing values removed from dataset')


def validate_duplicates(dq_engine: object, df: pd.DataFrame, delete_duplicate: str = True) -> None:
    '''
    This function checks for duplicate rows in the DataFrame using the `dq_engine`. It prints the 
    number of duplicate rows, and if duplicates are found and the `delete_duplicate` flag is set 
    to `True`, it removes the duplicate rows from the DataFrame.

    Parameters:
    dq_engine (object): A data quality engine object that provides a method `check_duplicates()`. 
        This method is expected to return the count of duplicate rows in the DataFrame.
    df (pd.DataFrame): The DataFrame to validate and clean.
    delete_duplicate (bool, optional): A flag to determine whether to remove duplicate rows (default is `True`). 
        If `False`, duplicates are reported but not removed.

    Returns:
    None: This function modifies the DataFrame in place and does not return a value.
    '''
    duplicate_rows = dq_engine.check_duplicates()
    print(f'Amount of duplicates rows: {duplicate_rows}')

    if duplicate_rows > 0 and delete_duplicate:
        df.drop_duplicates(inplace=True)
        print('Duplicate values removed from dataset')


def validate_data_types(dq_engine: object) -> None:
    '''
    This function uses the `dq_engine` to check the data types of columns in a dataset and 
    prints the data types for review.

    Parameters:
    dq_engine (object): A data quality engine object that provides a method `check_data_types()`. 
        This method is expected to return a dictionary or a DataFrame with column names as keys and 
        their respective data types as values.

    Returns:
    None: This function does not return a value; it only prints the data types.
    '''
    print(f'Data types: \n{dq_engine.check_data_types()}')


def add_sample_datetime_column(df: pd.DataFrame) -> None:
    '''
    This function assumes that the 'step' column in the DataFrame contains numeric values representing 
    a number of hours. It calculates the corresponding datetime for each row by adding the value in 
    the 'step' column (in hours) to the current time. A new column 'sample_datetime' is created to 
    store these calculated datetime values.

    Parameters:
    df (pd.DataFrame): The DataFrame to which the 'sample_datetime' column will be added. It is assumed 
        that the DataFrame contains a 'step' column with numeric values representing steps in hours.

    Returns:
    None: The function modifies the DataFrame in place by adding a new column 'sample_datetime'. 
        No value is returned.
    '''
    def steps_to_datetime(steps):
        current_time = datetime.datetime.now()
        final_time = current_time + datetime.timedelta(hours=steps)
        return final_time

    df['sample_datetime'] = df['step'].apply(steps_to_datetime)


def add_transaction_type_column(df: pd.DataFrame) -> None:
    '''
    This function classifies each transaction into one of four types based on the values in the 'nameOrig' 
    and 'nameDest' columns. The classification is as follows:
    - 'CC' if both 'nameOrig' and 'nameDest' contain 'C' (Customer to Customer).
    - 'CM' if 'nameOrig' contains 'C' and 'nameDest' contains 'M' (Customer to Merchant).
    - 'MC' if 'nameOrig' contains 'M' and 'nameDest' contains 'C' (Merchant to Customer).
    - 'MM' if both 'nameOrig' and 'nameDest' contain 'M' (Merchant to Merchant).
    A new column 'transaction_type' is added to the DataFrame to store these classifications.

    Parameters:
    df (pd.DataFrame): The DataFrame to which the 'transaction_type' column will be added. It must contain 
        the columns 'nameOrig' and 'nameDest', which are assumed to contain string values representing either a
        customer (denoted by 'C') or a merchant (denoted by 'M').

    Returns:
    None: The function modifies the DataFrame in place by adding the 'transaction_type' column. 
        No value is returned.
    '''
    conditions = [
        (df['nameOrig'].str.contains('C') & df['nameDest'].str.contains('C')),
        (df['nameOrig'].str.contains('C') & df['nameDest'].str.contains('M')),
        (df['nameOrig'].str.contains('M') & df['nameDest'].str.contains('C')),
        (df['nameOrig'].str.contains('M') & df['nameDest'].str.contains('M')) 
    ]
    values = ['CC', 'CM', 'MC', 'MM']
    
    df['transaction_type'] = np.select(conditions, values)   


def aggregate_data(df: pd.DataFrame, time_interval: str, aggregation_funcs: dict) -> pd.DataFrame:
    '''
    This function resamples the DataFrame using the 'sample_datetime' column, applying the specified
    aggregation functions to other columns. The time interval for resampling is provided as a string,
    and the aggregation functions are specified in a dictionary where the keys are column names and 
    the values are the aggregation functions to be applied (e.g., 'sum', 'mean', etc.).

    Parameters:
    df (pd.DataFrame): The DataFrame to aggregate. It must contain a 'sample_datetime' column, which
        will be used for resampling the data.
    time_interval (str): A string representing the time interval for resampling (e.g., 'D' for day, 
        'H' for hour, 'W' for week, etc.). The time interval must be compatible with pandas' resampling 
        functionality.
    aggregation_funcs (dict): A dictionary where keys are column names, and values are the aggregation
        functions to apply to those columns (e.g., {'column1': 'sum', 'column2': 'mean'}).

    Returns:
    pd.DataFrame: A new DataFrame containing the aggregated data.

    Raises:
    ValueError: If the 'sample_datetime' column is not present in the DataFrame or if some values in 
        'sample_datetime' cannot be converted to datetime format.
    '''
    if 'sample_datetime' not in df.columns:
        raise ValueError('The DataFrame must contain a "sample_datetime" column.')
    
    if not pd.api.types.is_datetime64_any_dtype(df['sample_datetime']):
        df['sample_datetime'] = pd.to_datetime(df['sample_datetime'], errors='coerce')
        if df['sample_datetime'].isnull().any():
            raise ValueError('Some values in the "sample_datetime" column could not be converted to datetime.')

    df.set_index('sample_datetime', inplace=True)
    resampled_df = df.resample(time_interval)
    aggregated_df = resampled_df.agg(aggregation_funcs)
    print(aggregated_df.head())

    df.reset_index(inplace=True)
    return aggregated_df


def profiling(df: pd.DataFrame, sample_size: float, stratified_sample: str = None) -> None:
    '''
    This function generates a profiling report of the specified DataFrame, either using the entire data 
    (full report) or by creating a sample of the data. The sample can be a simple random sample or a 
    stratified sample if a column name is provided. The report is saved as an HTML file using the 
    `ProfileReport` from the `ydata_profiling` library.
    The following columns are excluded from the profiling report:
    - **'step'**: Excluded because the `sample_datetime` column will be analyzed instead.
    - **'nameOrig'** and **'nameDest'**: These are excluded because they are identifiers. Instead, the 
    `transaction_type` column will be analyzed.

    Parameters:
    df (pd.DataFrame): The DataFrame to generate the profiling report for. It should contain columns 
        such as 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 
        'isFraud', 'isFlaggedFraud', 'sample_datetime', and 'transaction_type'.
    sample_size (float): A number between 0 and 1 representing the fraction of the data to sample. 
        If the value is less than 1, a sample is used; otherwise, the full DataFrame is used.
    stratified_sample (str, optional): The column name to use for stratified sampling. If provided, 
        the sample is generated with respect to the values in this column. If not provided, a simple
        random sample is taken.

    Returns:
    None: The function generates and saves an HTML file containing the profiling report. The report is 
          saved with different names based on the sampling method used:
          - 'profile_report_stratified_sample.html' for stratified sampling.
          - 'profile_report_sample.html' for simple random sampling.
          - 'profile_report.html' for the full report.
    '''
    columns_to_profile = [
        'type',
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest',
        'isFraud',
        'isFlaggedFraud',
        'sample_datetime',
        'transaction_type'
    ]

    df = df[columns_to_profile]

    if sample_size < 1:
        if stratified_sample:
            print(f'Generating Profiling Report for Stratified Sample for {stratified_sample}')
            sample_df, _ = train_test_split(df, test_size=(1-sample_size), stratify=df[stratified_sample])
            profile = ProfileReport(sample_df, title=f'Profiling Report - Stratified Sample for {stratified_sample}', explorative=True)
            profile.to_file('profile_report_stratified_sample.html')
        else:
            print('Generating Sample Profiling Report')
            sample_df = df.sample(frac=sample_size)
            profile = ProfileReport(sample_df, title='Sample Profiling Report', explorative=True)
            profile.to_file('profile_report_sample.html')
    else:
        print('Generating Full Profiling Report')
        profile = ProfileReport(df, title='Profiling Report', explorative=True)
        profile.to_file('profile_report.html')


main()