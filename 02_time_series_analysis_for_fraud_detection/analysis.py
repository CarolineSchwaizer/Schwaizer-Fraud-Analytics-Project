import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import auto_arima 
from scipy.stats import zscore
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import zipfile

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def main():
    file_path = 'ieee-fraud-detection.zip'

    df_transaction, df_identity = read_csv_file(file_path)

    #pre processing
    df = merge_datasets(df_transaction, df_identity)
    add_sample_datetime_column(df)

    #eda
    exploratory_analysis(df)
    df_window_functions = window_functions(df)

    #time series analysis
    statistics = {
        'TransactionAmt': {
            'statistic': 'sum', 
            'period': ['day', 'hour', 'week'],
            'rolling_statistics': 7 
        },
        'TransactionID': {
            'statistic': 'size',
            'period': ['day', 'week', 'day_of_week'],
            'rolling_statistics': 0
        }
    }
    
    time_series_analysis(df, statistics)

    #feature selection
    df_feat = feature_importance(df)
    X = scaled_data(df_feat)

    #anomaly detection
    anomalies_detection_zscore(df_feat)
    anomalies_detection_isolation_forest(df_feat, X)
    anomalies_detection_k_means(df_feat, X)

    #final report
    generate_final_report(df)
        
        
def read_csv_file(file_path: str) -> tuple:
    '''
    This function opens a ZIP archive located at the specified file path, extracts the 5th 
    and 4th files (indexed as 4 and 3, respectively) inside the ZIP file, and reads them into 
    two pandas DataFrames: one for transaction data and another for identity data.

    Parameters:
        file_path (str): The path to the ZIP file containing the CSV files.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - df_transaction (pd.DataFrame): The DataFrame containing transaction data.
            - df_identity (pd.DataFrame): The DataFrame containing identity data.
    '''
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_files = zip_ref.namelist()

        with zip_ref.open(zip_files[4]) as file:
            df_transaction = pd.read_csv(file)
        with zip_ref.open(zip_files[3]) as file:
            df_identity = pd.read_csv(file)
            
    return df_transaction, df_identity


def merge_datasets(df_transaction: pd.DataFrame, df_identity: pd.DataFrame) -> pd.DataFrame:
    '''
    This function performs a left join between the `df_transaction` and `df_identity` DataFrames 
    based on the 'TransactionID' column. After merging, it reduces memory usage by downcasting 
    the data types of numerical columns (float64 and int64) to more memory-efficient types. 
    Additionally, the function makes a copy of the merged DataFrame to avoid potential memory fragmentation.

    Parameters:
        df_transaction (pd.DataFrame): A DataFrame containing transaction data, including the 'TransactionID' column.
        df_identity (pd.DataFrame): A DataFrame containing identity data, including the 'TransactionID' column.

    Returns:
        pd.DataFrame: A merged DataFrame with reduced memory usage containing the combined data 
        from both `df_transaction` and `df_identity` on the 'TransactionID' column.
    '''
    df = pd.merge(df_transaction, df_identity, how='left', on='TransactionID')
    def reduce_memory_usage(df):
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            if df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
        return df

    df = reduce_memory_usage(df)
    df = df.copy() # force Pandas to reallocate the memory for the DataFrame due to PerformanceWarning: DataFrame is highly fragmented
    return df


def add_sample_datetime_column(df: pd.DataFrame) -> None:
    '''
    This function takes the 'TransactionDT' column, which represents time in seconds 
    from the current time, and applies a conversion to each value in that column. 
    It adds a new column 'sample_datetime' to the DataFrame, where each value is 
    a datetime calculated by adding the 'TransactionDT' value (in seconds) to 
    the current timestamp.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the 'TransactionDT' column, 
        which represents the time in seconds since the current time.

    Returns:
        None: This function modifies the input DataFrame in place by adding the 
        'sample_datetime' column.
    '''
    def seconds_to_datetime(TransactionDT):
        current_time = datetime.datetime.now()
        final_time = current_time + datetime.timedelta(seconds=TransactionDT)
        return final_time

    df['sample_datetime'] = df['TransactionDT'].apply(seconds_to_datetime)


def exploratory_analysis(df: pd.DataFrame) -> None:
    '''
    This function generates various summary statistics, visualizations, and insights
    from the dataset. It helps to understand the distribution of numerical and categorical
    features, the presence of missing or duplicated data, and the distribution of fraud 
    across different features.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the transaction data for analysis. It 
        is assumed to have columns such as 'isFraud' (indicating whether a transaction is 
        fraudulent) and other features related to the transaction.

    Outputs:
        Prints summary statistics and visualizations to the console and saves plots to files:
            - Summary statistics of numerical features (using `df.describe()`).
            - Count of missing values for each column.
            - Count of duplicated rows.
            - Distribution of fraud (the 'isFraud' column).
            - Information about categorical features, including their unique values and value counts.
            - Histograms and kernel density estimates (KDE) comparing the distribution of numerical 
            features for fraudulent and non-fraudulent transactions.

        Saves
        - For each part of the numerical features, a set of histograms comparing the distribution
        of fraud vs non-fraud for each feature is saved as PNG images.
    '''
    print(f'Summary statistics: {df.describe()}')
    print(f'Missing values: {df.isnull().sum().to_dict()}')
    print(f'Duplicated rows: {df.duplicated().sum()}')
    print(f'Fraud distribution: {df["isFraud"].value_counts(normalize=True).to_dict()}')

    categorical_columns = df.select_dtypes(include=['object']).columns
    print(f'Categorical features: {categorical_columns}')
    for col in categorical_columns:
        print(f'{col} unique values: {df[col].nunique()}')
        print(df[col].value_counts().head())

    numerical_columns = df.select_dtypes(include=['float', 'integer']).columns
    print(f'{len(numerical_columns)} Numerical features: {numerical_columns}')

    n_parts = 12
    n_features = len(numerical_columns)
    part_size = n_features // n_parts
    extra = n_features % n_parts

    for part in range(n_parts):
        start_idx = part * part_size + min(part, extra) 
        end_idx = (part + 1) * part_size + min(part + 1, extra)
        
        part_features = numerical_columns[start_idx:end_idx]
        
        n_cols = 6 
        n_rows = (len(part_features) // n_cols) + (len(part_features) % n_cols != 0)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
        axes = axes.flatten()
        
        for i, feature in enumerate(part_features):
            ax = axes[i]
            ax2 = ax.twinx()

            sns.histplot(df[df['isFraud'] == 0][feature], kde=True, bins=20, color='blue', ax=ax, label='Non-Fraud')
            sns.histplot(df[df['isFraud'] == 1][feature], kde=True, bins=20, color='red', ax=ax2, label='Fraud')
            ax.set_title(feature)
            ax.set_xlabel('Value')
            ax.set_ylabel('Non-Fraud Count', color='blue') 
            ax2.set_ylabel('Fraud Count', color='red')

        # remove unused axis in the figures
        for i in range(len(part_features), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(f'eda_numeric_features_distribution_part_{part+1}.png')
        plt.close()


def window_functions(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function generates new time-based features in the DataFrame by applying 
    several window functions on the 'TransactionAmt' and 'sample_datetime' columns.
    It calculates features such as cumulative sums, ranks, rolling averages, and 
    time-based shifts, which can be useful for time-series analysis or fraud detection.

    Parameters:
        df (pd.DataFrame): The DataFrame containing transaction data, with columns 
        such as 'TransactionAmt' (transaction amount) and 'sample_datetime' (timestamp 
        of the transaction).

    Returns:
        pd.DataFrame: The input DataFrame with new columns added
    '''
    df['month'] = df['sample_datetime'].dt.month

    # row_number()
    df['row_number'] = df.groupby('month').cumcount() + 1

    # rank()
    df['rank'] = df.groupby('month')['TransactionAmt'].rank(method='min', ascending=False)

    # lead()
    df['next_transaction_amt'] = df['TransactionAmt'].shift(-1)

    # lag()
    df['prev_transaction_amt'] = df['TransactionAmt'].shift(1)

    # cumulative sum()
    df['cumulative_sum'] = df.groupby('month')['TransactionAmt'].cumsum()

    # moving average (rolling avg)
    df['rolling_avg_3'] = df['TransactionAmt'].rolling(window=3).mean()

    # rolling max and min
    df['rolling_max'] = df['TransactionAmt'].rolling(window=3).max()
    df['rolling_min'] = df['TransactionAmt'].rolling(window=3).min()
    return df


def time_series_analysis(df: pd.DataFrame, statistics: dict, analyze_full_transactions: bool = True, analyze_decomposition: bool = True, analyze_arima: bool = True) -> None: 
    '''
    This function performs time series analysis on fraud data, providing insights into the 
    distribution of fraud and non-fraud transactions over different time periods. It computes 
    various statistics such as count, sum, and mean for specified features and periods, and 
    also analyzes trends with rolling statistics, seasonal decomposition, and ARIMA models for forecasting.

    Parameters:
        df (pd.DataFrame): The DataFrame containing transaction data with columns such as 
        'sample_datetime' (timestamp), 'isFraud' (fraud flag), and the features to be analyzed.
        
        statistics (dict): A dictionary specifying which features to analyze and how to 
        compute statistics. Each entry should contain:
            - 'statistic': The type of statistic ('size', 'sum', or 'mean').
            - 'period': A list of periods to group by (e.g., ['day', 'week', 'hour']).
            - 'rolling_statistics': The window size for calculating rolling statistics (integer).
        
        analyze_full_transactions (bool): A flag indicating whether to analyze the full 
        transactions (fraud and non-fraud) over time (default is True).
        
        analyze_decomposition (bool): A flag indicating whether to perform seasonal 
        decomposition of the daily fraud rate (default is True).

        analyze_arima (bool): A flag indicating whether to apply ARIMA for forecasting (default is True).

    Returns:
        None: This function generates and saves time series plots for each feature, 
        fraud vs non-fraud analysis, seasonal decomposition, and ARIMA forecasts. The plots 
        are saved as PNG files.

    '''
    df['day'] = df['sample_datetime'].dt.date
    df['hour'] = df['sample_datetime'].dt.hour
    df['week'] = df['sample_datetime'].dt.isocalendar().week
    df['day_of_week'] = df['sample_datetime'].dt.dayofweek

    fraud_df = df[df['isFraud'] == 1]

    for feature, params in statistics.items():
        statistic = params['statistic']
        periods = params['period']
        rolling = params['rolling_statistics']
        
        for period in periods:
            if statistic == 'size':
                stat_result = fraud_df.groupby(period).size()
            elif statistic == 'sum':
                stat_result = fraud_df.groupby(period)[feature].sum()
            elif statistic == 'mean':
                stat_result = fraud_df.groupby(period)[feature].mean()

            stat_result.plot(color='blue', label=f'{statistic.capitalize()} of {feature} per {period.capitalize()}')
            plt.xlabel(f"{period.capitalize()}")
            plt.ylabel(f"{statistic.capitalize()} of {feature}")
            plt.title(f"{statistic.capitalize()} of {feature} per {period.capitalize()}")
            plt.legend(loc='best')
            plt.savefig(f'time_series_{feature}_{statistic}_{period}.png')
            plt.close()

            if rolling > 0:
                if statistic == 'size':
                    fraud_rate = df.groupby(period)['isFraud'].mean()
                else:
                    fraud_rate = fraud_df.groupby(period)[feature].mean()
                rolmean = fraud_rate.rolling(rolling).mean()
                rolstd = fraud_rate.rolling(rolling).std()

                plt.plot(fraud_rate, color='blue', label=f'{feature.capitalize()} Mean: {period.capitalize()}')
                plt.plot(rolmean, color='red', label='Rolling Mean')
                plt.plot(rolstd, color='black', label='Rolling Std')
                plt.title(f'{feature.capitalize()} {period.capitalize()}: Rolling Mean & Std')
                plt.legend(loc='best')
                plt.savefig(f'time_series_{feature}_{period}_rolling_mean_stdv.png')
                plt.close()
    
    if analyze_full_transactions:
        for period in periods:
            if statistic == 'size':
                fraud_vs_nonfraud = df.groupby([period, 'isFraud']).size().unstack().fillna(0)
            elif statistic == 'sum':
                fraud_vs_nonfraud = df.groupby([period, 'isFraud'])[feature].sum().unstack().fillna(0)
            elif statistic == 'mean':
                fraud_vs_nonfraud = df.groupby([period, 'isFraud'])[feature].mean().unstack().fillna(0)

            plt.plot(fraud_vs_nonfraud[0], color='blue', label='Non-Fraud')
            plt.plot(fraud_vs_nonfraud[1], color='red', label='Fraud')
            plt.xlabel(f"{period.capitalize()}")
            plt.ylabel(f"{statistic.capitalize()} of {feature}")
            plt.title(f"{statistic.capitalize()} of {feature} for Fraud vs Non-Fraud by {period.capitalize()}")
            plt.legend(loc='best')
            plt.savefig(f'time_series_fraud_vs_non_fraud_{period}_{statistic}.png')
            plt.close()

    if analyze_decomposition:
        daily_fraud_rate = df.resample('D', on='sample_datetime')['isFraud'].mean()
        decomposition = seasonal_decompose(daily_fraud_rate, model='additive')
        decomposition.plot()
        plt.savefig('time_series_decomposition.png')
        plt.close()

    if analyze_arima:
        daily_fraud_rate = df.resample('D', on='sample_datetime')['isFraud'].mean()

        plot_acf(daily_fraud_rate.dropna(), lags=50)
        plt.savefig('time_series_acf_plot.png')
        plt.close()

        plot_pacf(daily_fraud_rate.dropna(), lags=50)
        plt.savefig('time_series_pacf_plot.png')
        plt.close()

        model = auto_arima(daily_fraud_rate, seasonal=True, m=7, stepwise=True, trace=True, error_action='ignore', suppress_warnings=True)

        print(f"Best ARIMA model: {model.summary()}")

        forecast_steps = 30  # Number of days to forecast
        forecast, conf_int = model.predict(n_periods=forecast_steps, return_conf_int=True)
        forecast_index = pd.date_range(daily_fraud_rate.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

        plt.plot(daily_fraud_rate, label='Observed Fraud Rate')
        plt.plot(forecast_index, forecast, label='ARIMA Forecast', color='red')
        plt.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3, label='95% Confidence Interval')
        plt.title('ARIMA Forecast of Daily Fraud Rate')
        plt.legend(loc='best')
        plt.savefig('time_series_arima_forecast.png')
        plt.close()

        residuals = daily_fraud_rate - model.predict_in_sample()
        plt.plot(residuals)
        plt.title('Residuals of ARIMA Model')
        plt.savefig('time_series_arima_residuals.png')
        plt.close()

        sm.qqplot(residuals, line='45')
        plt.savefig('time_series_arima_residuals_qqplot.png')
        plt.close()


def feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function calculates the feature importance using a Random Forest classifier to 
    identify the most important features for predicting fraud in the dataset. It trains 
    a Random Forest model on the numerical features of the input DataFrame and computes 
    the importance of each feature based on the model's trained coefficients. The function 
    prints the top 20 most important features.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing transaction data, with features 
        of numerical types (e.g., 'float', 'int'), as well as the 'isFraud' column for 
        fraud labels.

    Returns:
        pd.DataFrame: A DataFrame containing the top 20 most important features selected 
        by the Random Forest classifier, along with the 'sample_datetime' and 'isFraud' columns.
    '''
    X = df.select_dtypes(include=['float', 'int']).copy()
    y = df['isFraud']
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    print(f'Top 20 most important features: \n{feature_importances.head(20)}')

    top_20_features = feature_importances.head(20).index
    return df[top_20_features.tolist() + ['sample_datetime', 'isFraud']]


def scaled_data(df: pd.DataFrame) -> np.ndarray:
    '''
    This function drops the 'isFraud' and 'sample_datetime' columns, fills any missing 
    values in the remaining features with 0, and applies the scaling transformation 
    to the numerical features.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the features to be scaled.
        The DataFrame must include a column for 'isFraud' (target) and 'sample_datetime' 
        (datetime column).

    Returns:
        np.ndarray: A NumPy array of the scaled feature values, excluding the 
        'isFraud' and 'sample_datetime' columns.
    '''
    X = df.drop(['isFraud', 'sample_datetime'], axis=1)
    X.fillna(0, inplace = True)

    scaler = StandardScaler()
    return scaler.fit_transform(X)


def anomalies_detection_zscore(df: pd.DataFrame) -> None:
    '''
    This function performs anomaly detection using the Z-score method to identify 
    transactions with unusually high or low values based on their numerical features.
    The Z-score is calculated for each feature, and transactions with an absolute 
    Z-score greater than a threshold (3) are flagged as anomalous. 
    The results are visualized with a scatter plot of the transaction amounts, 
    showing normal and anomalous transactions on different days.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the transaction data. 
        The DataFrame must include numerical features (e.g., 'TransactionAmt') 
        and a 'sample_datetime' column for datetime information.

    Returns:
        None: The function modifies the input DataFrame in place by adding an 
        'anomaly_score' column. It also generates and saves a scatter plot image 
        of the anomalies.
    '''
    X = df.select_dtypes(include=['float', 'int']).columns

    df_zscore = df[X].apply(zscore)

    threshold = 3
    df['anomaly_score'] = (np.abs(df_zscore) > threshold).any(axis=1).astype(int)
    df['day'] = df['sample_datetime'].dt.date

    anomalous_transactions = df[df['anomaly_score'] == 1]
    normal_transactions = df[df['anomaly_score'] == 0]

    plt.figure(figsize=(10, 6))
    plt.scatter(normal_transactions['TransactionAmt'], normal_transactions['day'], color='blue', label='Normal Transactions', alpha=0.5)
    plt.scatter(anomalous_transactions['TransactionAmt'], anomalous_transactions['day'], color='red', label='Anomalous Transactions', alpha=0.7)
    plt.xlabel('Transaction Amount')
    plt.ylabel('Day of Transaction')
    plt.title('Z Score Anomalies Detection')
    plt.legend(loc='best')
    plt.savefig(f'anomalies_zscore.png')
    plt.close()

    print(anomalous_transactions[['TransactionAmt', 'day', 'anomaly_score']].head())


def anomalies_detection_isolation_forest(df: pd.DataFrame, X: np.ndarray) -> None:
    '''
    This function applies the Isolation Forest model to identify anomalous transactions 
    in a dataset based on the `TransactionAmt` and other numerical features in the input 
    DataFrame. It visualizes the anomalies by plotting transaction amounts against the 
    transaction dates, and classifies transactions as anomalous or normal based on a 
    specified contamination rate.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the transaction data, including the 
            'TransactionAmt', 'sample_datetime', and other relevant features. The 
            'sample_datetime' column should be in datetime format.
        X (np.ndarray): The feature matrix used for anomaly detection. This should include 
            the numerical features of the dataset (excluding non-numeric columns) already 
            scaled.

    Returns:
        None: The function modifies the input DataFrame by adding the 'anomaly_score' column, 
            which indicates whether a transaction is anomalous (1) or normal (0). It also 
            generates and saves a scatter plot visualizing the normal and anomalous 
            transactions based on their amount and day.
    '''
    iso_forest = IsolationForest(contamination=0.035)  # contamination = 0.035 due to % of fraud in the dataset
    df['anomaly_score'] = iso_forest.fit_predict(X)
    df['anomaly_score'] = df['anomaly_score'].map({1: 0, -1: 1}) 
    df['day'] = df['sample_datetime'].dt.date

    anomalous_transactions = df[df['anomaly_score'] == 1]
    normal_transactions = df[df['anomaly_score'] == 0]   

    plt.scatter(normal_transactions['TransactionAmt'], normal_transactions['day'], color='blue', label='Normal Transactions', alpha=0.5)
    plt.scatter(anomalous_transactions['TransactionAmt'], anomalous_transactions['day'], color='red', label='Anomalous Transactions', alpha=0.7)
    plt.xlabel('Transaction Amount')
    plt.ylabel('Day of Transaction')
    plt.title('Isolation Forest Anomalies Detection')
    plt.legend(loc='best')
    plt.savefig(f'anomalies_isolation_forest.png')
    plt.close()

    print(anomalous_transactions[['TransactionAmt', 'day']].head())


def anomalies_detection_k_means(df: pd.DataFrame, X: np.ndarray) -> None:
    '''
    This function applies the K-Means algorithm to cluster transactions into groups based 
    on their numerical features. It calculates the distance of each transaction from its 
    respective cluster centroid, and uses this distance to identify anomalous transactions 
    as those with a distance greater than the 95th percentile of the distance values.

    Parameters:
        df (pd.DataFrame): The DataFrame containing transaction data. This must include 
            'TransactionAmt', 'sample_datetime', and other numerical features relevant 
            for clustering. The 'sample_datetime' column should be in datetime format.
        X (np.ndarray): The feature matrix used for clustering. This should include the 
            numerical features of the dataset (excluding non-numeric columns).

    Returns:
        None: The function modifies the input DataFrame by adding the 'cluster' column (indicating 
            the cluster assignment), the 'distance_from_centroid' column (the distance to the 
            nearest cluster centroid), and the 'anomaly_score' column (1 for anomalous transactions, 
            0 for normal transactions). It also generates and saves a scatter plot visualizing the 
            clusters and highlighting the anomalous transactions.
    '''
    kmeans = KMeans(n_clusters=3)

    df['cluster'] = kmeans.fit_predict(X)
    df['day'] = df['sample_datetime'].dt.date
    df['distance_from_centroid'] = kmeans.transform(X).min(axis=1)

    threshold = df['distance_from_centroid'].quantile(0.95)

    df['anomaly_score'] = (df['distance_from_centroid'] > threshold).astype(int)

    anomalous_transactions = df[df['anomaly_score'] == 1]

    plt.scatter(df['TransactionAmt'], df['distance_from_centroid'], c=df['cluster'], cmap='viridis', label='Clustered Points')
    plt.scatter(anomalous_transactions['TransactionAmt'], 
                anomalous_transactions['distance_from_centroid'], 
                color='red', label='Anomalies', edgecolors='black', s=100)
    plt.xlabel('Transaction Amount')
    plt.ylabel('Distance from Centroid')
    plt.title('K Means Anomalies Detection')
    plt.legend(loc='best')
    plt.savefig(f'anomalies_kmeans.png')
    plt.close()


def generate_final_report(df: pd.DataFrame) -> None:
    '''
    Generates a PDF report for fraud detection analysis, including summary statistics, 
    time series analysis, and anomaly detection results.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing transaction data. It should include the 
            'TransactionAmt', 'isFraud', and 'sample_datetime' columns, along with other relevant
            features used for analysis.

    Returns:
        None: This function generates a PDF report saved as 'fraud_detection_report.pdf'.
    '''
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, "Fraud Detection Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(200, 10, "Summary Statistics for Transaction Amounts:", ln=True)
    pdf.set_font("Arial", size=12)
    
    transaction_summary = df.describe()['TransactionAmt']
    for stat, value in transaction_summary.items():
        pdf.cell(200, 10, f"{stat}: {value:.2f}", ln=True)
    
    fraud_distribution = df["isFraud"].value_counts(normalize=True).to_dict()
    for stat, value in fraud_distribution.items():
        pdf.cell(200, 10, f"{stat}: {value:.2f}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(200, 10, "Top 20 Features and their Importance:", ln=True)
    pdf.set_font("Arial", size=12)
   
    pdf.ln(10)
    pdf.cell(200, 10, "Variables Histogram:", ln=True)
    pdf.image('eda_numeric_features_distribution_part_1.png', x=10, w=180)
    pdf.image('eda_numeric_features_distribution_part_2.png', x=10, w=180)
    pdf.image('eda_numeric_features_distribution_part_3.png', x=10, w=180)
    pdf.image('eda_numeric_features_distribution_part_4.png', x=10, w=180)
    pdf.image('eda_numeric_features_distribution_part_5.png', x=10, w=180)
    pdf.image('eda_numeric_features_distribution_part_6.png', x=10, w=180)
    pdf.image('eda_numeric_features_distribution_part_7.png', x=10, w=180)
    pdf.image('eda_numeric_features_distribution_part_8.png', x=10, w=180)
    pdf.image('eda_numeric_features_distribution_part_9.png', x=10, w=180)
    pdf.image('eda_numeric_features_distribution_part_10.png', x=10, w=180)
    pdf.image('eda_numeric_features_distribution_part_11.png', x=10, w=180)
    pdf.image('eda_numeric_features_distribution_part_12.png', x=10, w=180)

    pdf.cell(200, 10, "Time Series Analysis:", ln=True)
    pdf.image('time_series_acf_plot.png', x=10, w=180)
    pdf.image('time_series_arima_forecast.png', x=10, w=180)
    pdf.image('time_series_arima_residuals.png', x=10, w=180)
    pdf.image('time_series_arima_residuals_qqplot.png', x=10, w=180)
    pdf.image('time_series_TransactionID_size_day.png', x=10, w=180)
    pdf.image('time_series_fraud_vs_non_fraud_day_size.png', x=10, w=180)
    pdf.image('time_series_TransactionAmt_sum_day.png', x=10, w=180)
    pdf.image('time_series_TransactionAmt_day_rolling_mean_stdv.png', x=10, w=180)
    pdf.image('time_series_decomposition.png', x=10, w=180)

    pdf.ln(10)
    pdf.cell(200, 10, "Anomaly Detection (Z-Score) - Transactions:", ln=True)
    pdf.image('anomalies_zscore.png', x=10, w=180)

    pdf.ln(10)
    pdf.cell(200, 10, "Anomaly Detection (Isolation Forest) - Transactions:", ln=True)
    pdf.image('anomalies_isolation_forest.png', x=10, w=180)

    pdf.ln(10)
    pdf.cell(200, 10, "Anomaly Detection (K-Means) - Transactions:", ln=True)
    pdf.image('anomalies_kmeans.png', x=10, w=180)

    pdf.output("fraud_detection_report.pdf")
    print("Report generated successfully.")


main()