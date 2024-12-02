import json

from pyspark.sql import functions as F
from pyspark.sql import SparkSession, DataFrame


def main():
    spark = SparkSession.builder.getOrCreate() 

    file_path = 'creditcard.csv'
    df = read_csv_file(spark, file_path)
    
    #process and transform data
    df = validate_missing_values(df)
    df = validate_duplicates(df)

    df = add_sample_datetime_column(df)

    #implement partitioning strategies
    df = df.orderBy('sample_datetime').repartitionByRange(2, 'sample_datetime')
    df.cache()
    
    #aggregations
    agg_dict1 = {
        'amount': [F.min, F.max, F.avg],
        'class': [F.sum, F.count]
    }
    aggregate_data(df, ['sample_datetime'], agg_dict1)

    agg_dict2 = {
        'v10': [F.stddev, F.min, F.max, F.avg],
        'v11': [F.stddev, F.min, F.max, F.avg],
        'amount': [F.min, F.max, F.avg],
        'class': [F.sum, F.count]
    }
    aggregate_data(df, 
        [F.to_date(F.col('sample_datetime')).alias('date'), 
            F.hour(F.col('sample_datetime')).alias('hour')], agg_dict2)
    
    #statistics
    generate_data_quality_and_statistics_report(df)


def read_csv_file(spark, file_path:str) -> DataFrame:
    '''
    This function reads a CSV file from the provided file path, infers the schema, 
    and casts the 'time' column to an integer type. It then selects a predefined set 
    of columns including 'v1' to 'v28', 'amount', and 'class'. The schema of the 
    resulting DataFrame is printed.

    Parameters:
    spark (SparkSession): The Spark session used to read the CSV file.
    file_path (str): The path to the CSV file to be loaded into a Spark DataFrame.

    Returns:
    DataFrame: A Spark DataFrame with the selected columns and schema transformations.
    '''
    df = spark.read.option('maxPartitionBytes', '128MB').csv(file_path, header=True, inferSchema=True)
    df.printSchema()

    df = df.select(
        F.col('time').cast('integer')
        ,F.col('v1')
        ,F.col('v2')
        ,F.col('v3')
        ,F.col('v4')
        ,F.col('v5')
        ,F.col('v6')
        ,F.col('v7')
        ,F.col('v8')
        ,F.col('v9')
        ,F.col('v10')
        ,F.col('v11')
        ,F.col('v12')
        ,F.col('v13')
        ,F.col('v14')
        ,F.col('v15')
        ,F.col('v16')
        ,F.col('v17')
        ,F.col('v18')
        ,F.col('v19')
        ,F.col('v20')
        ,F.col('v21')
        ,F.col('v22')
        ,F.col('v23')
        ,F.col('v24')
        ,F.col('v25')
        ,F.col('v26')
        ,F.col('v27')
        ,F.col('v28')
        ,F.col('amount')
        ,F.col('class')
    )
    return df


def validate_missing_values(df: DataFrame, delete_missing_value: str = True) -> DataFrame:
    '''
    Validates and optionally removes null values from the DataFrame. 

    Parameters:
    df (pyspark.sql.DataFrame): The DataFrame to check for null values.
    delete_missing_value (bool, optional): If True, rows with null values will be dropped. 
        Defaults to True.

    Returns:
    pyspark.sql.DataFrame: A DataFrame with missing values dropped if `delete_missing_value` is True.
        Otherwise, the original DataFrame is returned.
    '''
    null_counts = df.select(
        [F.sum(F.col(c).isNull().cast('int')).alias(c) for c in df.columns]
    )
    
    null_counts_dict = null_counts.first().asDict()

    sum_null_counts = 0
    print('Null values per column:')
    for column, count in null_counts_dict.items():
        sum_null_counts += count
        print(f'{column}: {count} null values')

    if sum_null_counts > 0 and delete_missing_value:
        df = df.dropna()
        print('Missing values removed from dataset')
    return df


def validate_duplicates(df: DataFrame, delete_duplicate: str = True) -> DataFrame:
    '''
    Validates and optionally removes duplicate rows from the DataFrame.
    It calculates and prints the number of total rows and duplicate rows.

    Parameters:
    df (pyspark.sql.DataFrame): The DataFrame to check for duplicate rows.
    delete_duplicate (bool, optional): If True, duplicate rows will be removed. Defaults to True.

    Returns:
    pyspark.sql.DataFrame: A DataFrame with duplicates removed if `delete_duplicate` is True.
        Otherwise, the original DataFrame is returned.
    '''
    total_rows = df.count()
    unique_rows = df.distinct().count()
    duplicate_rows = total_rows - unique_rows
    
    print(f'Total rows: {total_rows}')
    print(f'Duplicate rows: {duplicate_rows}')

    if duplicate_rows > 0 and delete_duplicate:
        df = df.dropDuplicates()
        print('Duplicate values removed from dataset')
    return df


def add_sample_datetime_column(df: DataFrame) -> DataFrame:
    '''
    Adds a 'sample_datetime' column to the DataFrame by converting the 'time' column (representing the 
    number of seconds elapsed between each transaction and the first transaction in the dataset) into a 
    datetime. The 'sample_datetime' is calculated by adding the 'time' value (seconds) to the current Unix timestamp.

    Parameters:
    df (pyspark.sql.DataFrame): The DataFrame containing the 'time' column.

    Returns:
    pyspark.sql.DataFrame: A DataFrame with an additional 'sample_datetime' column, which represents the 
        datetime calculated by adding the 'time' value (in seconds) to the current time.
    '''
    df = df.withColumn('sample_datetime', 
        (F.col('time') + F.unix_timestamp()).cast('timestamp'))
    
    return df


def aggregate_data(df: DataFrame, group_by_columns: list, agg_dict: dict) -> DataFrame:
    '''
    Perform aggregation on a PySpark DataFrame.

    Parameters:
    df (pyspark.sql.DataFrame): The DataFrame to perform aggregation on.
    group_by_columns (list): A list of column names to group by.
    agg_dict (dict): A dictionary where keys are column names to aggregate and values are the aggregation functions.

    Returns:
    pyspark.sql.DataFrame: A DataFrame containing the aggregated results.
    '''
    grouped_df = df.groupBy(*group_by_columns)

    agg_exprs = []
    for column, agg_funcs in agg_dict.items():
        if isinstance(agg_funcs, list):
            for agg_func in agg_funcs:
                agg_exprs.append(agg_func(F.col(column)).alias(f'{column}_{agg_func.__name__}'))
        else:
            agg_exprs.append(agg_funcs(F.col(column)).alias(f'{column}_{agg_funcs.__name__}'))

    result_df = grouped_df.agg(*agg_exprs)
    result_df.show(5)
    return result_df


def generate_data_quality_and_statistics_report(df: DataFrame) -> dict:
    '''
    Generates a comprehensive data quality and statistical report for a PySpark DataFrame. The report includes:
    - Missing values count for each column
    - Duplicate rows count
    - Data types of all columns
    - Unique values count for each column
    - Summary statistics for numeric columns
    - Most frequent values for categorical columns
    - Outliers detection for numeric columns
    Additionally, the report is beautified and printed in a human-readable format using JSON formatting.

    Parameters:
    df (pyspark.sql.DataFrame): The DataFrame for which the report will be generated. It should contain both numeric 
        and categorical columns to generate the relevant statistics.

    Returns:
    dict: A dictionary containing the data quality and statistical summary.
    '''
    report = {}

    missing_values = df.select(
        [F.sum(F.col(c).isNull().cast('integer')).alias(c) for c in df.columns]
    ).first().asDict()
    report['Missing Values'] = missing_values

    duplicates = df.count() - df.distinct().count()
    report['Duplicates'] = duplicates

    data_types = df.dtypes
    report['Data Types'] = {col: dtype for col, dtype in data_types}

    unique_values = df.select(
        [F.countDistinct(F.col(c)).alias(c) for c in df.columns]
    ).first().asDict()
    report['Unique Values'] = unique_values

    numeric_columns = [col for col, dtype in df.dtypes if dtype in ['int', 'double', 'float']]
    numeric_summary = df.select(numeric_columns).describe().collect()
    numeric_summary_dict = {}
    for row in numeric_summary:
        numeric_summary_dict[row['summary']] = row.asDict()
    report['Numeric Summary'] = numeric_summary_dict

    categorical_columns = [col for col, dtype in df.dtypes if dtype in ['string', 'boolean']]
    mode_values = {}
    for col in categorical_columns:
        mode_value = df.groupBy(col).count().orderBy(F.desc('count')).first()
        mode_values[col] = mode_value[col] if mode_value else None
    report['Most Frequent Values'] = mode_values

    def detect_outliers(col_name):
        quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
        Q1, Q3 = quantiles
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df.filter((F.col(col_name) < lower_bound) | (F.col(col_name) > upper_bound)).count()
        return outliers

    outliers = {}
    for col in numeric_columns:
        outliers[col] = detect_outliers(col)
    
    report['Outliers'] = outliers

    def beautify_json(json_data):
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        
        return json.dumps(json_data, indent=4, separators=(',', ': '))

    print(beautify_json(report))
    return report


main()