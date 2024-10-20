import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the datasets
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')

# Merge the datasets on 'ID'

df_merged = pd.merge(pd.merge(df1[['ID', 'gender']], df2, on='ID'), df3, on='ID')

# Split data by gender male
df_male = df_merged[df_merged['gender'] == 1]

# Select well-being columns only (adjust the column range based on your dataset)
df_male['male_wellbeing_mean'] = df_male.iloc[:, 11:].mean(axis=1)

# Display the DataFrame with the new column
print(df_male[['ID', 'C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk', 'male_wellbeing_mean']])
# df_male[['ID', 'C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk', 'male_wellbeing_mean']].to_csv('male_wellbeing.csv', index=False)

# import dataset male well-being into dataframe
df = pd.read_csv('male_wellbeing.csv')

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:, 1:-1].values  # Select all columns except the first and last
y = df.iloc[:,-1].values #male_wellneing_mean # normally we access the value of y from the last column # response variables

# Split dataset into 80% training and 20% test sets
x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                                    test_size=0.2,
                                                    random_state=0)

# Build a linear regression model
model = LinearRegression()

# Train (fit) the linear regression model using the training set
model.fit(x_train, y_train)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: %.4f" % model.intercept_)
print("Coefficient: ", model.coef_)

print("b_1: %.4f" % model.coef_[0])
print("b_2: %.4f" % model.coef_[1])
print("b_3: %.4f" % model.coef_[2])
print("b_4: %.4f" % model.coef_[3])
print("b_5: %.4f" % model.coef_[4])
print("b_6: %.4f" % model.coef_[5])
print("b_7: %.4f" % model.coef_[6])
print("b_8: %.4f" % model.coef_[7])

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(x_test)

#######################################################################################################################
# # I Have used two methods but got different answers
# # Method 1
# # Plotting Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Best Fit Line (y=x)')
plt.xlabel('Explanatory Variable')
plt.ylabel('Response Variable')
plt.title('Actual vs Predicted Values for Multiple Linear Regression Model')
plt.legend()
plt.show()

###########################################################################################################################

# # Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:
# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred))  # OR # rmse = math.sqrt(mse)
# Normalised Root Mean Square Error
y_max = y.max()
y_min = y.min()
rmse_norm = rmse / (y_max - y_min) # normalised rmse
# R-Squared
r_2 = metrics.r2_score(y_test, y_pred)

# print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
 
 
 # """
# COMPARE THE PERFORMANCE OF THE LINEAR REGRESSION MODEL
# VS.
# A DUMMY MODEL (BASELINE) THAT USES MEAN AS THE BASIS OF ITS PREDICTION
# """
# Compute mean of values in (y) training set
y_base = np.mean(y_train)

# Replicate the mean values as many times as there are values in the test set
y_pred_base = [y_base] * len(y_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_base_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_base})
print(df_base_pred)

# Compute standard performance metrics of the baseline model:
mae = metrics.mean_absolute_error(y_test, y_pred_base)  ## Mean Absolute Error
mse = metrics.mean_squared_error(y_test, y_pred_base)  # Mean Squared Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred_base))  ## Root Mean Square Error
# Normalised Root Mean Square Error
y_max = y.max()
y_min = y.min()
rmse_norm = rmse / (y_max - y_min)
# R-Squared
r_2 = metrics.r2_score(y_test, y_pred_base)

print("Baseline model's performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)