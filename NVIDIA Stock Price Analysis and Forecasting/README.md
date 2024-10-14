# NVIDIA Stock Price Analysis and Forecasting

## Project Introduction
**Author:** Qiaoying (Annie) Zhang

This project focuses on building a comprehensive end-to-end solution to analyze and forecast the stock price of NVIDIA (NVDA), combining ETL pipelines, time series analysis, interactive visualizations, and advanced machine learning techniques. The goal is to accurately predict NVIDIA’s stock price while comparing traditional time series models with deep learning and ensemble-based approaches to ensure high accuracy and robustness.

*The stock price data is sourced from the public API - Yahoo Finance (5 years of NVDA data)*

*Data Preview:*
| Date       | Open       | High       | Low        | Close      | Adj Close  | Volume      |
|------------|------------|------------|------------|------------|------------|-------------|
| 2019-10-15 | 4.754000   | 4.982250   | 4.740000   | 4.909250   | 4.885488   | 664124000   |
| 2019-10-16 | 4.875000   | 4.980500   | 4.843750   | 4.855250   | 4.831748   | 428944000   |
| 2019-10-17 | 4.900000   | 4.945250   | 4.802500   | 4.857250   | 4.833740   | 263436000   |
| 2019-10-18 | 4.857750   | 4.890500   | 4.687500   | 4.762250   | 4.739199   | 307440000   |
| 2019-10-21 | 4.824000   | 4.913750   | 4.805000   | 4.900250   | 4.876531   | 261868000   |

![NVIDIA Stock Chart](nvda_chart.png)

### Time Series Analysis: ARIMA to SARIMAX
To forecast future stock prices, the analysis first employs time series models, specifically ARIMA (Auto-Regressive Integrated Moving Average). The order and seasonal order of the model are optimized using metrics like AIC and diagnostics such as residual analysis to ensure stationarity and minimize autocorrelation. However, the presence of seasonal trends and exogenous factors, such as earnings reports or macroeconomic events, makes ARIMA limited in scope. To address these, the analysis extends to SARIMAX (Seasonal ARIMA with Exogenous Variables), which can model both seasonality and external influences, thus improving the accuracy of predictions.

### Machine Learning and Deep Learning Approaches
While time series models provide solid baselines, the project goes beyond with machine learning and deep learning models to capture non-linear relationships and complex patterns within the data:

- **Keras-based Neural Networks**: These are trained with dropout and early stopping to capture non-linear trends and reduce overfitting, allowing the model to generalize well to unseen data. This approach is particularly useful in scenarios where the stock exhibits high volatility.
  
- **XGBoost Regressor**: Selected for its ability to handle feature interactions efficiently through gradient boosting, it excels in datasets with multiple input features, including engineered ones like volatility indices or technical indicators, giving a more refined prediction.

- **Stacked Ensemble Models**: Developed by combining the outputs of individual models (e.g., SARIMAX, XGBoost, and neural networks) using a meta-learner. This ensemble strategy leverages the strengths of each model, balancing the precision of statistical models with the adaptability of machine learning.

### Rationale for Chosen Approaches
- **Statistical Models (ARIMA, SARIMAX)**: These are utilized for their interpretability and ability to handle time series data effectively, particularly with seasonal patterns.
  
- **Deep Learning Models (Keras)**: Integrated to account for complex, non-linear dependencies that are difficult for traditional time series models to capture.

- **XGBoost**: Offers robust performance with structured data, leveraging boosting techniques to improve predictions iteratively.

- **Stacked Ensembles**: Provide a powerful way to combine predictions from multiple models, minimizing errors and producing more accurate results than individual models alone.

By employing a combination of time series models, machine learning models, and ensemble techniques, this project ensures that both short-term fluctuations and long-term trends are captured accurately, ultimately aiming to outperform baseline statistical forecasts.

![Model Results](nvda_results.png)

## Requirements
- Python 3.x
- Libraries: Pandas, NumPy, scikit-learn, statsmodels, Keras, XGBoost, Plotly, etc.

## Installation
```bash
pip install -r requirements.txt
