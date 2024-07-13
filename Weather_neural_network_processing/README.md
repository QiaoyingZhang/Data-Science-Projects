# Weather Prediction – ANN & RNN
This project is a part of the personal projects done by Qiaoying(Annie) Zhang.  Other projects can be found at the [main GitHub repo](https://github.com/QiaoyingZhang/Data-Science-Projects).

#### -- Project Status: [Completed]

## Project Intro/Objective
The purpose of this project is to delve into the intricacies of neural networks, specifically Artificial Neural Networks (ANNs) and Recurrent Neural Networks (RNNs), while drawing comparisons between them. By utilizing weather prediction data, the project explores relationships between weather parameters such as precipitation, temperature, and humidity. Incorporating data mining techniques with neural networks aims to enhance the accuracy and efficiency of daily weather forecasting, potentially offering more cost-effective and reliable results compared to other prediction models. This project can significantly impact civic planning and resource management by providing more accurate weather predictions.

### Methods Used
* Inferential Statistics
* Machine Learning
* Data Visualization
* Predictive Modeling
* etc.

### Technologies
* Python
* Pandas, matplotlib
* Jupyter
* Tensorflow
* StandardScaler 

## Project Description
1. Data Sources:
    1. The dataset includes weather-related parameters such as temperature, precipitation, and humidity collected from various public records and open data portals.
    2. Data spans a significant time frame, ensuring a comprehensive set for robust model training and evaluation.
    3. MUTHUKUMAR.J.(2017). "Weather Dataset," Kaggle, https://www.kaggle.com/datasets/muthuj7/weather-dataset.
2. Goals and Hypotheses:
    1. Goal: To improve daily weather forecasting accuracy using neural networks.
    2. Hypothesis: RNNs, given their ability to handle sequential data, will outperform ANNs in weather prediction tasks due to the temporal dependencies in weather patterns.
4. Data Analysis and Visualization:
    1. Exploratory Data Analysis (EDA): Extensive EDA to understand distributions, correlations, and trends in the weather data.
    2. Visualization: Use of charts and graphs (e.g., line plots, histograms, heatmaps) to visualize relationships between different weather parameters.
    3. Feature Engineering: Transformed 'Formatted Date' into 'Hour',	'Year',	'Month', 'Day'.
6. Modeling Work:
    1. ANN and RNN Models: Development and training of both ANN and RNN models, utilizing backpropagation for learning.
    2. Evaluation Metrics: Comparison of models using model accuracy and loss.
    3. Validation: Validate the models with test set and plot the model accuracy and loss by epoch.
8. Challenges and Blockers:
    1. Data Quality: Handling missing data and outliers, which could significantly impact model performance.
    2. Model Tuning: Fine-tuning hyperparameters to achieve optimal model performance while avoiding overfitting.

This project aims to leverage neural network techniques to create a reliable and efficient weather prediction model. The insights gained from this project could significantly enhance the accuracy of weather forecasts, contributing to better civic planning and resource management.

## Needs of this project

- data exploration/descriptive statistics
- data processing/cleaning
- statistical modeling
- machine learning modeling
- writeup/reporting

## Instructions for running & compilation of code
**Step 1** - Download the following necessary libraries:

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import datetime as dt
    from datetime import datetime
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

     
**Step 2** - Download dataset [weatherHistory.csv](https://github.com/QiaoyingZhang/Data-Science-Projects/blob/main/Weather_neural_network_processing/weatherHistory.csv) within the [Weather_neural_network_processing](https://github.com/QiaoyingZhang/Data-Science-Projects/tree/main/Weather_neural_network_processing) folder.

**Step 3** - To read data locally:

     data = pd.read_csv('weatherHistory.csv')
     
     data.head()
     
     # Checking data type and count of features
     data.info()
     # Checking Statistical Summary
     data.describe()
     # Checking Target Variable
     print(data["Summary"].value_counts())

**Step 4** - After completing the above steps, begin to run the code from this cell 6 onwards. For reference, cell 4 states:
   
     plt.figure(figsize=(12,7))
     plt.xticks(rotation=90)
     sns.barplot(data=data, x="Summary", y="Temperature (C)",hue="Precip Type")


## Contribution

**[Qiaoying(Annie) Zhang](https://github.com/QiaoyingZhang)(@QiaoyingZhang)**
