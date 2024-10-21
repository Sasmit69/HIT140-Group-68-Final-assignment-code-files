import pandas as pd

# Load the two CSV files into DataFrames
df1 = pd.read_csv('dataset1.csv')  # File with gender, minority status, and deprived status
df2 = pd.read_csv('Screentime Average.csv')  # File with screentime and ID

# Merge the two DataFrames based on the matching 'ID' column
merged_df = pd.merge(df1, df2, on='ID')

# Calculate average screentime by gender (1 for male, 0 for female)
avg_screentime_by_gender = merged_df.groupby('gender')['average_column'].mean()

# Calculate average screentime by minority status (0 for majority, 1 for minority)
avg_screentime_by_minority = merged_df.groupby('minority')['average_column'].mean()

# Calculate average screentime by deprived status (1 for deprived, 0 for non-deprived)
avg_screentime_by_deprived = merged_df.groupby('deprived')['average_column'].mean()

# Calculate the percentage of screentime for deprived status
screentime_by_deprived_percentage = merged_df.groupby('deprived')['average_column'].sum() / merged_df['average_column'].sum() * 100

# Print the results
print(f"Average Screen Time by Gender:\nMale: {avg_screentime_by_gender.get(1, 0):.2f}\nFemale: {avg_screentime_by_gender.get(0, 0):.2f}")

print(f"\nAverage Screen Time by Minority Status:\nMajority: {avg_screentime_by_minority.get(0, 0):.2f}\nMinority: {avg_screentime_by_minority.get(1, 0):.2f}")

print(f"\nAverage Screen Time by Deprived Status:\nNon-Deprived: {avg_screentime_by_deprived.get(0, 0):.2f}\nDeprived: {avg_screentime_by_deprived.get(1, 0):.2f}")

print(f"\nPercentage of Total Screen Time by Deprived Status:\nNon-Deprived: {screentime_by_deprived_percentage.get(0, 0):.2f}%\nDeprived: {screentime_by_deprived_percentage.get(1, 0):.2f}%")
