import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

# Load the CSV files into DataFrames
df1 = pd.read_csv('Screentime Average.csv')
df2 = pd.read_csv('Wellbeing Score.csv')

# Merge the two DataFrames based on the matching 'id' column
merged_df = pd.merge(df1, df2, on='ID')

# Select the features (X) and the target variable (y)
# Replace 'feature_column' and 'target_column' with actual column names
X = merged_df[['average_column']].values  # Independent variable (features)
y = merged_df['Wellbeing Score'].values     # Dependent variable (target)

# Split the dataset into 65% training and 35% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Predict the target variable for the test data
y_pred = model.predict(X_test)

# Calculate R-squared, MAE, MSE, RMSE, and NRMSE
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Calculate NRMSE - Normalized RMSE using the range of the actual values
nrmse = rmse / (y_test.max() - y_test.min())  # You can normalize by mean or range

# Calculate Adjusted R-squared
n = len(y_test)  # Number of observations in the test set
p = X_train.shape[1]  # Number of independent variables (features)
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

# Print the performance metrics
print(f'\nR-squared: {r2:.4f}')
print(f'Adjusted R-squared: {adjusted_r2:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'NRMSE: {nrmse:.4f}')

# Rounding Wellbeing Scores to the nearest integer for classification (1 to 5 scale)
merged_df['Wellbeing Score'] = merged_df['Wellbeing Score'].round()

# Calculate the percentage for each wellbeing category
wellbeing_counts = merged_df['Wellbeing Score'].value_counts(normalize=True) * 100

# Print the percentage of wellbeing scores
print("Percentage of people's wellbeing scores based on the scale:")
for score, percentage in wellbeing_counts.sort_index().items():
    print(f"Score {int(score)}: {percentage:.2f}%")

# Create a scatter plot of the test data and the regression line
plt.scatter(X_test, y_test, color='blue', label='Actual values')
plt.plot(X_test, y_pred, color='red', label='Predicted values')

plt.title('Scatter Plot with Regression Line')
plt.xlabel('Screentime')
plt.ylabel('Wellbeing Score')
plt.legend()
plt.show()
