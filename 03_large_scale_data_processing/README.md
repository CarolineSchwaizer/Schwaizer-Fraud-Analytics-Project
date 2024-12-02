# Large-Scale Data Processing

## Project Requirements
This project is designed to load, process, and analyze a credit card transaction dataset using PySpark. It includes the following requirements:
-	Process and transform data using PySpark DataFrame operations
-	Implement efficient data partitioning strategies
-	Create aggregations and statistical summaries
-	Optimize performance using appropriate Spark configurations
-	Generate comprehensive data quality reports


## Dataset Summary
### Context
The dataset used in this project is the **Credit Card Fraud Detection** dataset, available from Kaggle. It contains anonymized credit card transaction data, where each row represents a transaction made by a cardholder. 

### Columns
- **Size**: The dataset consists of 284,807 transactions, of which 492 are fraudulent (about 0.17% of the data).
- **Features**: The dataset includes 31 features:
  - **Time**: The number of seconds elapsed between each transaction and the first transaction in the dataset.
  - **V1, V2, ..., V28**: Anonymized features resulting from a PCA transformation (Principal Component Analysis). These features represent various transaction attributes but are not directly interpretable.
  - **Amount**: The transaction amount.
  - **Class**: The target variable, where 1 indicates a fraudulent transaction and 0 indicates a legitimate transaction.

NOTE: Since fraudulent transactions are much fewer in number compared to legitimate transactions, the dataset is **highly imbalanced**.

For more details, please refer to the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).


## Requirements
Ensure the following dependencies are installed to run the project:

- **Apache Spark**: A distributed computing framework to handle large datasets.
- **PySpark**: Python interface for Apache Spark.
- **Python 3.x**: The programming language used for the ETL pipeline.


## Files Structure
- etl_pipeline.py: Main ETL pipeline script to process the data
- creditcard.csv: Data source


## Project Overview
This project uses PySpark to process and analyze the credit card transaction data. The following steps are performed in the pipeline:
1. **Read and Preprocess Data**: Load the CSV file, handle missing values, remove duplicates, and add a datetime column.
2. **Data Transformation**: Partition data by time and perform aggregations on various columns.
3. **Data Quality & Statistical Reporting**: Generate a comprehensive report on data quality, including missing values, duplicates, unique values, summary statistics, and outlier detection.

The end result is a clean dataset ready for analysis and fraud detection model training.


## Optimization & Partitioning Strategies
To optimize the performance of the ETL pipeline, PySpark provides several configuration and partitioning strategies (a few were used in this project - those with code examples - and the others are worth mentioning since they could improve the project's performance):

### 1. **Data Partitioning**:
Partitioning data allows Spark to distribute data across multiple nodes in the cluster, optimizing parallel processing. In this project, the data is repartitioned based on the range of values in the `sample_datetime` column, which ensures that the data is split based on time and can be processed efficiently. In this case, it redistributes the DataFrame based on the values of the `sample_datetime` column and sorts the data within each partition. This type of partitioning is most commonly used when it is needed to perform range-based operations (such as sorting or aggregations) or when dealing with time-series data.
```python
df = df.orderBy('sample_datetime').repartitionByRange(2, 'sample_datetime')
```

### 2. **Spark Configuration for Optimization**:
Several Spark configurations can help optimize performance by controlling memory usage, the number of partitions during data loading, and more:

- **Max Partition Size**: Setting the `maxPartitionBytes` option helps control the maximum size of each partition when reading data. This optimization helps manage memory usage effectively across the cluster.
```python
df = spark.read.option('maxPartitionBytes', '128MB').csv(file_path, header=True, inferSchema=True)
```
  
- **Shuffle Partitions**: The number of partitions used during shuffling operations (e.g., during joins or aggregations) can be controlled by adjusting `spark.sql.shuffle.partitions`. The default value is 200, but for larger datasets, increasing this number can improve performance.

- **Executor & Driver Memory**: Adjusting the memory settings for the Spark executors and the driver can prevent memory bottlenecks during computation. For larger datasets, consider increasing the memory allocation to ensure smooth processing.

### 3. **Broadcast Joins**:
Broadcast joins allow for more efficient joins when working with a large dataset and a smaller reference dataset. By broadcasting the smaller dataset to all nodes in the cluster, Spark avoids the need to shuffle large amounts of data, significantly speeding up the join operation.

### 4. **Persisting Data**:
When performing multiple transformations on the same dataset, it's a good practice to persist intermediate results to avoid re-computing them every time. By using `.cache()` or `.persist()`, Spark stores the DataFrame in memory, improving the speed of iterative operations.
```python
df.cache()
```

### 5. **Parallelism**:
PySpark's inherent parallel processing capability can be leveraged by controlling the number of cores used by each executor. Optimizing the parallelism of your jobs can significantly speed up ETL pipelines, especially for large datasets. You can set the level of parallelism based on the size of the data and the resources available in the cluster.