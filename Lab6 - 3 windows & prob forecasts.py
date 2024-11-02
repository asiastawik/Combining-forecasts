import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#TASK 1 - fixed calibration window
#Obtain forecasts for 28-day, 364-day and 728-day calibration
#windows using the following regression model:
#ˆ Pd,h = β0 + β1Pd−1,h + β2Pd−7,h + β3Ld,h,
#Calculate MAE of obtained predictions.

d = np.loadtxt('GEFCOM.txt')
data = d[:, 2]
data_L = d[:, 4]

windows = [28, 364, 728]
max_win = max(windows)

forecasts_window_lists=[]

for T in windows:
    # print(T)
    forecast_list = []
    real_list = []
    err_h = []
    for h in range(24):
        p_hour = data[h::24]
        # print(len(p_hour))
        p_hour_L = data_L[h::24]

        cal_data = p_hour[max_win-T:max_win] #728-28 : 728
        cal_data_L = p_hour_L[max_win-T:max_win]
        X2 = cal_data[0:T - 7] #0 : 28-7
        X1 = cal_data[6:T - 1]
        X0 = np.ones(np.shape(X1))
        Y = cal_data[7:T]
        X3 = cal_data_L[7:T]

        X = np.column_stack([X0, X1, X2, X3])
        betas = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

        # print(len(np.ones((len(p_hour) - max_win)))) #1082 - 728
        # print(len(p_hour[max_win - 1:len(p_hour) - 1])) # 728-1 : 1082 -1
        # print(len(p_hour[max_win - 7:len(p_hour) - 7])) # 728 - 7 : 1082 - 7
        # print(len(p_hour_L[max_win:])) # 728 : 1082
        X_fut = np.column_stack([np.ones((len(p_hour) - max_win)), p_hour[max_win - 1:len(p_hour) - 1],
                                     p_hour[max_win - 7:len(p_hour) - 7], p_hour_L[max_win:]])
        real = p_hour[max_win:]
        forecast = np.dot(X_fut, betas)
        forecast_list.append(forecast)
        real_list.append(real)
        err_h.append(np.abs(forecast - real))

    mae = np.mean(err_h)
    print(f"MAE for {T}-day window: {mae}")
    forecasts_window_lists.append(forecast_list)

#TASK 2
# Evaluate the performance of the following forecast combinations:
# AW(364,728)
# AW(28,364)
# AW(28,728)
# Compare them with the results from Task 1.

list_28, list_364, list_728 = forecasts_window_lists
# print(len(real_list))

#AW(364,728)
forecast = [(x + y) / 2 for x, y in zip(list_364, list_728)]
forecast_array = np.array(forecast)
real_array = np.array(real_list)
error_a = np.abs(forecast_array - real_array)
mae = np.mean(error_a)
print(f"MAE for AW(364,728)-day window: {mae}")

#AW(28,364)
forecast = [(x + y) / 2 for x, y in zip(list_28, list_364)]
forecast_array = np.array(forecast)
real_array = np.array(real_list)
error_b = np.abs(forecast_array - real_array)
mae = np.mean(error_b)
print(f"MAE for AW(28,364)-day window: {mae}")

#AW(28,728)
forecast = [(x + y) / 2 for x, y in zip(list_28, list_728)]
forecast_array = np.array(forecast)
real_array = np.array(real_list)
error_c = np.abs(forecast_array - real_array)
mae = np.mean(error_c)
print(f"MAE for AW(28,728)-day window: {mae}")

#Task 3 & 4
# Use the forecasts from naive #1 and naive #2 models to obtain
# the 50% and 90% prediction intervals for days 366 to 1082. Use
# the 365-day rolling calibration window and treat each hour
# separately. Calculate the empirical coverage.

# 4 Average obtained probabilistic predictions and calculate the
# empirical coverage. Compare it with the coverage of base models
# from Task 3.

T = 365

forecast_errors = []
prediction_intervals = []

nominal_coverage = 0.9
# nominal_coverage = 0.5

coverage_h_rolling = []
coverage_h_rolling_1 = []
coverage_h_rolling_2 = []

for h in range(24):
    p_hour = d[d[:, 1] == h, 2]

    coverage_h_rolling_h_avg = []  # Initialize coverage list for this hour
    coverage_h_rolling_h_1 = []
    coverage_h_rolling_h_2 = []

    for t in range(T, len(p_hour)):
        # Filter data for the rolling calibration window
        p_hour_rolling = p_hour[(t - T):t]

        # 1st Naive Forecast for the rolling window
        pf_naive1_rolling = np.roll(p_hour_rolling, 1)
        # Calculate the forecast errors
        forecast_errors_rolling_1 = p_hour_rolling[:(T - 1)] - pf_naive1_rolling[:(T - 1)]
        # Calculate the quantiles for the forecast errors
        q1_rolling = np.quantile(forecast_errors_rolling_1, (1 - nominal_coverage) / 2)
        q2_rolling = np.quantile(forecast_errors_rolling_1, (1 + nominal_coverage) / 2)

        # Compute lower and upper bounds for the prediction interval
        lower_bound_rolling_1 = pf_naive1_rolling[T - 1] + q1_rolling  # indeksowanie od 0
        upper_bound_rolling_1 = pf_naive1_rolling[T - 1] + q2_rolling

        real_rolling_1 = p_hour_rolling[T - 1]

        # 2nd Naive Forecast for the rolling window
        pf_naive2_rolling = np.roll(p_hour_rolling, 7)
        forecast_errors_rolling_2 = p_hour_rolling[:(T - 7)] - pf_naive2_rolling[:(T - 7)]
        q1_rolling_2 = np.quantile(forecast_errors_rolling_2, (1 - nominal_coverage) / 2)
        q2_rolling_2 = np.quantile(forecast_errors_rolling_2, (1 + nominal_coverage) / 2)

        # Compute lower and upper bounds for the prediction interval
        lower_bound_rolling_2 = pf_naive2_rolling[T - 7] + q1_rolling_2  # indeksowanie od 0
        upper_bound_rolling_2 = pf_naive2_rolling[T - 7] + q2_rolling_2

        real_rolling_2 = p_hour_rolling[T - 7]

        upper_bound_rolling_avg = (upper_bound_rolling_1 + upper_bound_rolling_2)/2
        lower_bound_rolling_avg = (lower_bound_rolling_1 + lower_bound_rolling_2)/2

        # Check if the real value falls within the prediction interval
        if real_rolling_1 < upper_bound_rolling_1 and real_rolling_1 > lower_bound_rolling_1:
            coverage_h_rolling_h_1.append(1)
        else:
            coverage_h_rolling_h_1.append(0)

        if real_rolling_2 < upper_bound_rolling_2 and real_rolling_2 > lower_bound_rolling_2:
            coverage_h_rolling_h_2.append(1)
        else:
            coverage_h_rolling_h_2.append(0)

        if real_rolling_1 < upper_bound_rolling_avg and real_rolling_1 > lower_bound_rolling_avg:
            coverage_h_rolling_h_avg.append(1)
        else:
            coverage_h_rolling_h_avg.append(0)

    coverage_h_rolling_1.append(np.mean(coverage_h_rolling_h_1))
    coverage_h_rolling_2.append(np.mean(coverage_h_rolling_h_2))
    coverage_h_rolling.append(np.mean(coverage_h_rolling_h_avg))

print("Average:", np.mean(coverage_h_rolling))
print("#1:", np.mean(coverage_h_rolling_1))
print("#2:", np.mean(coverage_h_rolling_2))

plt.figure()
plt.plot(coverage_h_rolling, label='AVG')
plt.plot(coverage_h_rolling_1, label='#1')
plt.plot(coverage_h_rolling_2, label='#2')
plt.title('Coverage')
plt.legend()
plt.show()

#Average coverage is much better for both 50% and 90% nominal coverage.