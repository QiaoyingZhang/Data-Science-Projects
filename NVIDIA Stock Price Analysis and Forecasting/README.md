# NVIDIA Stock Price Analysis and Forecasting

## Project Introduction
**Author:** Qiaoying (Annie) Zhang

This project focuses on building a comprehensive end-to-end solution to analyze and forecast the stock price of NVIDIA (NVDA), combining ETL pipelines, time series analysis, interactive visualizations, and advanced machine learning techniques. The goal is to accurately predict NVIDIA’s stock price while comparing traditional time series models with deep learning and ensemble-based approaches to ensure high accuracy and robustness.

*The stock price data is sourced from the public API - Yahoo Finance (5 years of NVDA data)*

![NVIDIA Stock Chart](nvidia_chart.png)

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

![Model Results](nvidia_results.png)

## Table of Contents
- [Project Introduction](#project-introduction)
- [Import Libraries](#import-libraries)
- [Extract Data](#extract-data)
- [ETL Pipeline (Extract, Transform, Load)](#etl-pipeline-extract-transform-load)
    - [Data Quality Check](#data-quality-check)
    - [Transform Data](#transform-data)
- [Data Visualization (Interactive Chart with Plotly)](#data-visualization-interactive-chart-with-plotly)
- [Metrics](#metrics)
- [Time Series Analysis (ARIMA + SARIMAX)](#time-series-analysis-arima--sarimax)
    - [Preparing Data](#preparing-data)
    - [ACF and PACF Plots](#acf-and-pacf-plots)
    - [Applying ARIMA Model](#applying-arima-model)
    - [Applying SARIMAX Model](#applying-sarimax-model)
    - [Evaluate ARIMA & SARIMAX Models](#evaluate-arima--sarimax-models)
    - [Time-Series Forecast Comparison](#time-series-forecast-comparison)
- [Deep Learning + Machine Learning Approach](#deep-learning--machine-learning-approach)
    - [Preparing Data](#preparing-data-1)
    - [DL Approach](#dl-approach)
    - [ML Approach](#ml-approach)
    - [Stack Predictions](#stack-predictions)
    - [Evaluate Ensemble Model](#evaluate-ensemble-model)
- [Overall Comparison (Stacked Model vs ARIMA vs SARIMAX)](#overall-comparison-stacked-model-vs-arima-vs-sarimax)

## Requirements
- Python 3.x
- Libraries: Pandas, NumPy, scikit-learn, statsmodels, Keras, XGBoost, Plotly, etc.

## Installation
```bash
pip install -r requirements.txt
