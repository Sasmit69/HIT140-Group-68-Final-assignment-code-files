import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('dataset3.csv')

# Specify the columns to average
columns_to_average = ['Optm','Usef','Relx','Intp','Engs','Dealpr','Thcklr','Goodme','Clsep','Conf','Mkmind','Loved','Intthg','Cheer']  # Replace with your column names

# Compute the average of the selected columns for each row
df['Wellbeing Score'] = df[columns_to_average].mean(axis=1)

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_file.csv', index=False)

print(df.head())  # To check the first few rows of the DataFrame
