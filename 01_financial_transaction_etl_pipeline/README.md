# Financial Transaction ETL Pipeline

## Project Requirements

This project implements an ETL (Extract, Transform, Load) pipeline to process and analyze financial transaction data. It includes the following requirements:

- Create a data pipeline that: 
    - Cleanses and validates transaction data
    - Performs necessary transformations (currency conversion, timestamp standardization)
    - Aggregates data at different time intervals
-	Implement data quality checks and validation rules
-	Create summary statistics and data profiling reports


## PaySim Dataset Summary

### Context
The PaySim dataset simulates mobile money transactions and is designed for fraud detection research. It uses synthetic data generated from real-world mobile transaction logs in an African country. This dataset helps researchers develop and test fraud detection algorithms. 
For more details, please refer to the [Kaggle Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1/data).

### Content
- **Simulated Transactions**: The dataset includes mobile money transactions over 30 days, with both legitimate and fraudulent activities.
- **Fraudulent Transactions**: Fraudulent transactions are marked with `isFraud`, and attempts to transfer more than 200,000 are flagged (`isFlaggedFraud`).
- **Time and Frequency**: The dataset contains 744 steps, where each step represents one hour of simulated time (total of 31 days).

### Columns

| Column Name        | Description                                                                                                                                           |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **step**           | Represents 1 hour of real-world time (744 steps for 31 days).                                                                                         |
| **type**           | Transaction type: `CASH-IN`, `CASH-OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`.                                                                             |
| **amount**         | The transaction amount in local currency.                                                                                                             |
| **nameOrig**       | Customer initiating the transaction.                                                                                                                 |
| **oldbalanceOrg**  | Initial balance of the sender.                                                                                                                       |
| **newbalanceOrig** | Sender's balance after the transaction.                                                                                                              |
| **nameDest**       | Recipient of the transaction.                                                                                                                        |
| **oldbalanceDest** | Recipient's initial balance (not available for merchants).                                                                                           |
| **newbalanceDest** | Recipient's balance after the transaction (not available for merchants).                                                                             |
| **isFraud**        | Indicates whether the transaction is fraudulent (1) or legitimate (0).                                                                               |
| **isFlaggedFraud** | Flags transactions involving amounts greater than 200,000 as suspicious.                                                                             |

### Data Usage Considerations
- **Fraud Detection**: Ideal for testing fraud detection models.
- **Columns to Avoid**: For fraud detection, avoid using `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, and `newbalanceDest`, as these relate to cancelled fraudulent transactions.


## Requirements

- Python 3.x

Before running the project, make sure to install the necessary Python packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `ydata-profiling`


## File Structure

- etl_pipeline.py: Main ETL pipeline script to process the data
- data_quality.py: Data quality validation functions
- archive.zip: Compressed CSV data file

NOTE: data_quality.py is separated from the main ETL pipeline to provide reusable functions for data quality checks. This allows it to be integrated into other projects that require similar validation logic for data processing.


## Project Overview

The ETL pipeline processes financial transaction data and performs the following steps:

1. **Data Validation**
   - Validates the dataset by checking for missing values, duplicates, and data type inconsistencies.
   - Cleans the data by removing rows with missing values and duplicates.

2. **Data Transformation**
   - Adds new columns to the dataset:
     - `sample_datetime`: A datetime column derived from the `step` column.
     - `transaction_type`: Classifies transactions into categories (e.g., 'CC', 'CM', 'MC', 'MM') based on the `nameOrig` and `nameDest` columns.

3. **Data Aggregation**
   - Aggregates the dataset by different time intervals (hourly, daily, weekly) and applies custom aggregation functions (mean, sum, count, etc.).

4. **Profiling Report**
   - Generates a detailed profiling report of the dataset, either using the full data or a random/stratified sample. The report includes various statistics and visualizations about the dataset.