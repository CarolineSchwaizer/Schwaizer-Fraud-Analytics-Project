# Time Series Analysis for Fraud Detection

## Project Requirements

This project implements fraud analytics by using machine learning and statistical techniques. It includes the following requirements:
- Analyze temporal patterns in fraudulent transactions
- Create moving averages and identify seasonal trends
- Develop statistical measures for anomaly detection
- Implement SQL window functions for time-based analysis
- Visualize patterns and create interpretable reports


## Dataset Summary

The dataset used for this project is the **IEEE-CIS Fraud Detection** dataset, which is available on [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data).

This dataset contains information about financial transactions, with features such as transaction amount, time of transaction, and customer identity data. The goal is to detect fraudulent transactions using this data.


## Requirements

- Python 3.x

Before running the project, make sure to install the necessary Python packages:

- `pandas` – Data manipulation and analysis.
- `numpy` – Numerical computing.
- `matplotlib` – Data visualization.
- `seaborn` – Statistical data visualization.
- `scipy` – Scientific computations.
- `statsmodels` – Time series analysis.
- `scikit-learn` – Machine learning algorithms.
- `fpdf` – PDF document creation.
- `zipfile` – Handling ZIP files.
- `datetime` – Date and time handling.
- `pmdarima` – Automate arima model selection and forecasting.


## Files Structure

The project follows this file structure:

- fraud_detection.py: Main Python script for fraud detection
- ieee-fraud-detection.zip: ZIP file containing the dataset

And it generates the following outputs:

- eda_numeric_features_distribution_part_X.png: EDA output images
- time_series_*.png: Time series analysis output images
- anomalies_*.png: Anomaly detection output images
- fraud_detection_report.pdf: Final generated PDF report


## Project Overview

This project performs various steps for fraud detection, including:

1. **Data Preprocessing**:
   - Merging two datasets: `train_transaction.csv` and `train_identity.csv`.
   - Reducing memory usage of the dataset by downcasting data types.
   - Adding a `sample_datetime` column based on the transaction timestamp (`TransactionDT`).

2. **Exploratory Data Analysis (EDA)**:
   - Statistical summary and missing values check.
   - Visualizing the distribution of transaction amounts for fraudulent and non-fraudulent transactions.
   - Plotting histograms for numerical features and visualizing relationships between fraud and transaction features.
   - It also provides a few of window functions to be applied to the dataset, if needed.

3. **Time Series Analysis**:
   - Analyzing the time-related features across different periods, in this step, it is expected to provide a dictionary with the specific analysis to be made, in the following format:
   ```python
   statistics = {
      'feature_to_be_analyzed': {
         'statistic': ['sum'], #sum, size or mean
         'period': ['day', 'hour', 'week'],
         'rolling_statistics': 7 
      }
   }   
   ```
   - Performing rolling statistics (mean, standard deviation) on key features to analyze time series data and capture trends, seasonality, and volatility over a moving window.
   - Analyzing trends, patterns, and seasonality in the fraud detection process, by using the ```sm.tsa.seasonal_decompose``` function from the statsmodels library, which performs seasonal decomposition of a time series. It breaks the time series into three main components:

      1. Trend: The long-term movement or direction of the data.
      2. Seasonality: The repeating short-term patterns or cycles in the data, typically on a fixed period (e.g., yearly, monthly, daily).
      3. Residual (or Noise): The random noise or residual component after removing the trend and seasonality.

   - ARIMA Forecasting: it is aldo applied the ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting on the fraud rate. It computes and plots the forecast for the next 30 days along with a 95% confidence interval. The function generates the following plots:
      1. ACF Plot: Identifies autocorrelation patterns and helps in determining the q (moving average) parameter.
      2. PACF Plot: Identifies partial autocorrelation patterns and helps in determining the p (autoregressive) parameter.
      3. ARIMA Forecast Plot: Visualizes the future predictions and the associated confidence intervals.
      4. Residuals Plot: Evaluates the goodness of fit of the ARIMA model by inspecting the residuals.
      5. QQ Plot: Assesses the normality of the residuals to check model assumptions.

4. **Feature Selection**:
   - Using a random forest classifier to identify the most important features for predicting fraud: it calculates feature importance based on how much each feature contributes to reducing the impurity in the decision trees. The importance score of each feature reflects its relevance in predicting the target variable (in this case, fraud or non-fraud).
   - Scaling the features using standardization: standardization refers to the process of rescaling features to have a mean of 0 and a standard deviation of 1, allowing each feature to have a standard normal distribution, and ensuring that all features contribute equally to the model. 

5. **Anomaly Detection**:
   - Detecting anomalies using three methods:
     - **Z-Score**: Identifying outliers based on statistical standard deviation.
     - **Isolation Forest**: Using an ensemble method to detect anomalies.
     - **K-Means Clustering**: Clustering data points and identifying anomalies based on distance from centroids.

6. **Final Report Generation**:
   - A PDF report summarizing key findings, including:
     - Statistical summary of transaction amounts.
     - Visualizations of the distribution of numerical features.
     - Time series analysis plots.
     - Anomaly detection results.
   - The PDF report is generated using the `fpdf` library.

To run the project, simply execute the script `fraud_detection.py`. It will read the dataset, process the data, perform analysis, and generate the final fraud detection report as a PDF file.

### Execution
To run the project, follow these steps:

1. Install the required packages:
```bash
pip install pmdarima pandas numpy matplotlib seaborn scipy statsmodels scikit-learn fpdf
```
   
2. Download the **IEEE-CIS Fraud Detection** dataset from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data).

3. Ensure the dataset files are in the same directory as the script.

4. Run the Python script:
```bash
python fraud_detection.py
```

This will generate the final PDF report (`fraud_detection_report.pdf`) with all analysis and visualizations.

### Final Conclusion

In this analysis, several techniques were employed to detect fraud in financial transactions from the IEEE-CIS Fraud Detection dataset. The goal was to identify unusual or anomalous behaviors in transaction data that could indicate fraudulent activity. Here's a summary of the key insights and conclusions drawn from the analysis:

---

### 1. **Exploratory Data Analysis (EDA)**
The exploratory analysis revealed important patterns and insights from the dataset:
- **Missing Values & Duplicates:** some missing values were observed in the dataset, but the imputation strategy (filling missing values with zeros or appropriate values) helped maintain data integrity. No duplicates were found, ensuring the dataset was clean for further analysis.
- **Fraud Distribution:** The distribution of fraud (`isFraud`) showed that fraudulent transactions represented a small fraction of the total transactions, which is typical in fraud detection scenarios ~3.5%.
- **Feature Exploration:** Several features, especially numeric ones like `TransactionAmt`, exhibited significant differences between fraudulent and non-fraudulent transactions. Visualizing these features revealed clear patterns that could aid in fraud detection.

---

### 2. **Time Series Analysis**
The time series analysis, focusing on transaction amounts and counts over different periods (day, week, hour), highlighted:
- **Temporal Patterns:** Fraudulent transactions showed distinct temporal patterns that could be used to better detect anomalies. 
- **Rolling Statistics:** Rolling mean and standard deviation were useful for smoothing out fluctuations and identifying trends in fraud rates over time. This helps in distinguishing between normal variations and outliers.
- **Decomposition Analysis:** The seasonal decomposition of transaction data revealed periodic trends and irregularities, indicating that fraud patterns might exhibit periodic behavior that can be captured for better prediction.
- **ARIMA Forecasting:** The ARIMA model provided insights into future fraud rates, with forecasts for the next 30 days. The inclusion of a 95% confidence interval offered a measure of uncertainty in the predictions, highlighting potential future risks.
   - Best Model:
   The best-fitting model for this time series analysis was identified as **ARIMA(0,1,1)(0,0,1)[7]**, which successfully predicted future fraud rates and captured temporal dependencies within the data.

   - **Total Fit Time:** 4.139 seconds
   - **Model Summary:**
   - **Model Type:** SARIMAX (Seasonal ARIMA with exogenous variables)
   - **Number of Observations:** 183
   - **Log Likelihood:** 611.169
   - **AIC (Akaike Information Criterion):** -1216.338
   - **BIC (Bayesian Information Criterion):** -1206.726
   - **HQIC (Hannan-Quinn Information Criterion):** -1212.441
   - **Sample Period:** 12-02-2024 to 06-02-2025

   ### ARIMA Model Coefficients:

   | **Variable**  | **Coefficient** | **Std Error** | **z-value** | **p-value**  | **95% Confidence Interval** |
   |---------------|-----------------|---------------|-------------|--------------|-----------------------------|
   | ma.L1         | -0.7550         | 0.048         | -15.674     | 0.000        | (-0.849, -0.661)            |
   | ma.S.L7       | -0.1192         | 0.071         | -1.677      | 0.094        | (-0.259, 0.020)             |
   | sigma2        | 7.052e-05       | 8.33e-06      | 8.468       | 0.000        | (5.42e-05, 8.68e-05)        |

   ### Model Diagnostics:

   - **Ljung-Box (L1) Q-Test:** 0.96 (p-value: 0.33) – This indicates that there is no significant autocorrelation at lag 1.
   - **Jarque-Bera (JB) Test:** 1.41 (p-value: 0.49) – The residuals are normally distributed with no significant skew or excess kurtosis.
   - **Heteroskedasticity (H):** 1.24 (p-value: 0.40) – No evidence of heteroskedasticity, indicating that the variance of the residuals is constant over time.
   - **Skewness:** 0.17 – The distribution of the residuals is almost symmetrical.
   - **Kurtosis:** 2.72 – The distribution of residuals is close to normal, as it is close to 3.

- **Residual Analysis:** The residuals from the ARIMA model were analyzed through plots and a QQ plot, ensuring that the model adequately captured the underlying data structure. Any patterns or deviations in residuals would prompt further refinement of the model.

---

### 3. **Feature Importance Analysis**
Using **Random Forest Classifier**, the most important features influencing fraud detection were identified. The **Top 20 most important features** (ranked by their importance score) are as follows:

- **card2**: 0.020576
- **row_number**: 0.019868
- **cumulative_sum**: 0.019687
- **rank**: 0.019358
- **rolling_avg_3**: 0.018149
- **TransactionAmt**: 0.017653
- **rolling_min**: 0.016822
- **next_transaction_amt**: 0.016788
- **prev_transaction_amt**: 0.016415
- **C1**: 0.016323
- **rolling_max**: 0.015950
- **C13**: 0.015818
- **addr1**: 0.015113
- **hour**: 0.015034
- **C14**: 0.012148
- **V257**: 0.011595
- **card5**: 0.011544
- **V258**: 0.011269
- **month**: 0.010945
- **day_of_week**: 0.010461

These features play a crucial role in distinguishing fraudulent transactions from legitimate ones, and focusing on them could improve model performance.

---

### 4. **Anomaly Detection**
Three different anomaly detection techniques were employed to identify unusual transaction patterns:
- **Z-Score:** This method was useful for identifying extreme outliers based on statistical deviation from the mean. It helped detect transactions with unusually high or low amounts compared to the average.
- **Isolation Forest:** This ensemble method was particularly effective for detecting anomalies in a high-dimensional space. It successfully identified transactions that differed significantly from the majority of the dataset, which could indicate fraudulent behavior.
- **K-Means Clustering:** By grouping transactions into clusters, K-Means helped identify data points far from the centroid of any cluster, marking them as potential anomalies. This approach provided valuable insights into atypical transaction patterns.

---

### 5. **Recommendations for Further Action**
Based on the findings of this analysis, the following recommendations can be made:
- **Model Refinement:** Future models could benefit from using a wider range of features, including additional time-related variables, and more advanced anomaly detection algorithms like **autoencoders** or **one-class SVM**.
- **Real-Time Detection:** The insights from the time series and anomaly detection analysis can be applied to build a real-time fraud detection system that flags suspicious transactions based on temporal trends and clustering behaviors.
- **Continual Monitoring:** Given the evolving nature of fraud, it is important to continually monitor the model's performance and retrain it periodically with fresh data to ensure its effectiveness.