import pandas as pd


class DataQuality:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def check_missing_values(self):
        '''
        Check for missing values in the dataset.
        
        Returns:
            pd.Series: A series with each column and the number of missing values.
        '''
        return self.data.isnull().sum()

    def check_duplicates(self):
        '''
        Check for duplicate rows in the dataset.
        
        Returns:
            int: The number of duplicate rows.
        '''
        return self.data.duplicated().sum()

    def check_data_types(self):
        '''
        Check the data types of each column to ensure consistency.
        
        Returns:
            pd.Series: A series of data types for each column.
        '''
        return self.data.dtypes

    def summary(self):
        '''
        Provides a summary of all the data quality checks, without column names in outliers.
        
        Returns:
            dict: A dictionary with the results of various checks.
        '''
        summary = {
            'missing_values': self.check_missing_values(),
            'duplicate_rows': self.check_duplicates(),
            'data_types': self.check_data_types(),
        }
        return summary