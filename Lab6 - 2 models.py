import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# TASK 1 - fixed calibration window
# Obtain forecasts using the following regression models:
# ˆ Pd,h = β0 + β1Pd−1,h + β2Pd−7,h + β3Ld,h,
# ˆ Pd,h = β0 + β1Pd−2,h + β2Pd−4,h + β3Ld-1,h,
window = 365
# Calculate MAE of obtained predictions.

d = np.loadtxt('GEFCOM.txt')
data = d[:, 2]
data_L = d[:, 4]

# TASK 1 - One window (365) and two models
T = window
forecast_list_m1 = []
forecast_list_m2 = []
real_list = []
err_h_m1 = []
err_h_m2 = []

for h in range(24):
    p_hour = data[h::24]
    p_hour_L = data_L[h::24]

    cal_data = p_hour[-T:]  # Last T days for calibration
    cal_data_L = p_hour_L[-T:]
    X2 = cal_data[0:T - 7]
    X1 = cal_data[6:T - 1]
    X0 = np.ones(np.shape(X1))
    Y = cal_data[7:T]
    X3 = cal_data_L[7:T]

    X = np.column_stack([X0, X1, X2, X3])
    betas_m1 = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

    X_fut = np.column_stack([np.ones((len(p_hour) - T,)), p_hour[T - 1:len(p_hour) - 1], p_hour[T - 7:len(p_hour) - 7],p_hour_L[T:]])
    real = p_hour[T:]
    real_list.append(real)
    forecast_m1 = np.dot(X_fut, betas_m1)
    forecast_list_m1.append(forecast_m1)
    err_h_m1.append(np.abs(forecast_m1 - real))

    # Second model
    X1 = cal_data[2:T - 2]
    X2 = cal_data[0:T - 4]
    Y = cal_data[4:T]
    X3 = cal_data_L[3:T-1]
    X0 = np.ones(np.shape(X1))
    # print(len(X1), len(X2), len(X3), len(Y))
    X = np.column_stack([X0, X1, X2, X3])
    betas_m2 = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

    X_fut = np.column_stack([np.ones((len(p_hour) - T,)), p_hour[T - 2:len(p_hour) - 2], p_hour[T - 4:len(p_hour) - 4], p_hour_L[T - 1: len(p_hour_L)-1]])
    forecast_m2 = np.dot(X_fut, betas_m2)
    forecast_list_m2.append(forecast_m2)
    err_h_m2.append(np.abs(forecast_m2 - real))

mae_m1 = np.mean(err_h_m1)
print(f"MAE for Model 1 with {T}-day window: {mae_m1}")

mae_m2 = np.mean(err_h_m2)
print(f"MAE for Model 2 with {T}-day window: {mae_m2}")

# TASK 2 - Evaluate the performance of the forecast combinations
forecast_array_m1 = np.array(forecast_list_m1)
forecast_array_m2 = np.array(forecast_list_m2)

forecast_combination = (forecast_array_m1 + forecast_array_m2) / 2
error_combination = np.abs(forecast_combination - real_list)
mae_combination = np.mean(error_combination)
print(f"MAE for AW(Model 1, Model 2): {mae_combination}")
