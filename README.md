# Combining Forecasts

This project focuses on combining forecasts using the GEFCOM dataset, which includes data on zonal prices, system loads, and other relevant variables. The objective is to evaluate different calibration windows and forecast combinations, while assessing their accuracy through various performance metrics.

## Project Overview

The project consists of the following key tasks:

1. **Regression Model Forecasting**:
   - Obtain forecasts using a regression model for calibration windows of 28 days, 364 days, and 728 days. 
   - Calculate the Mean Absolute Error (MAE) of the obtained predictions to evaluate performance.

2. **Forecast Combination Evaluation**:
   - Assess the performance of various forecast combinations, specifically:
     - AW(364, 728)
     - AW(28, 364)
     - AW(28, 728)
   - Compare the results of these combinations with those from the initial regression model forecasting in Task 1.

3. **Naive Model Prediction Intervals**:
   - Utilize forecasts from naive #1 and naive #2 models to calculate 50% and 90% prediction intervals for the period from days 366 to 1082. Employ a 365-day rolling calibration window and treat each hour separately, calculating the empirical coverage of these intervals.

4. **Probabilistic Prediction Averaging**:
   - Average the obtained probabilistic predictions from the naive models and calculate the empirical coverage. Compare this coverage with that of the base models from Task 3 to assess improvement.

## Data

The project utilizes the GEFCOM dataset, which includes the following columns: YYYYMMDD, HH, zonal price, system load, zonal load, and day-of-the-week. This dataset serves as the foundation for all forecasting tasks.

## Results

The outputs of the project will include forecasts for specified periods, performance metrics such as MAE, and empirical coverage calculations for the probabilistic forecasts. This will provide insights into the effectiveness of different forecasting strategies and their combinations.
