import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('dataset2.csv')

# Specify the columns to average
columns_to_average = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']  # Replace with your column names

# Compute the average of the selected columns for each row
df['average_column'] = df[columns_to_average].mean(axis=1)

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_file.csv', index=False)

print(df.head())  # To check the first few rows of the DataFrame
