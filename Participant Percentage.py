import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('dataset1.csv')

# Calculate the percentage for gender (1 for male, 0 for female)
gender_counts = df['gender'].value_counts(normalize=True) * 100  # Normalize to get percentages
male_percentage = gender_counts.get(1, 0)  # 1 for male, get 0 if not present
female_percentage = gender_counts.get(0, 0)  # 0 for female, get 0 if not present

# Calculate the percentage for minority status (0 for majority, 1 for minority)
minority_counts = df['minority'].value_counts(normalize=True) * 100
majority_percentage = minority_counts.get(0, 0)  # 0 for majority
minority_percentage = minority_counts.get(1, 0)  # 1 for minority

# Calculate the percentage for deprived status (1 for deprived, 0 for non-deprived)
deprivation_counts = df['deprived'].value_counts(normalize=True) * 100
non_deprived_percentage = deprivation_counts.get(0, 0)  # 0 for non-deprived
deprived_percentage = deprivation_counts.get(1, 0)  # 1 for deprived

# Rounding Wellbeing Scores to the nearest integer for classification (1 to 5 scale)
merged_df['Wellbeing Score'] = merged_df['Wellbeing Score'].round()

# Calculate the percentage for each wellbeing category
wellbeing_counts = merged_df['Wellbeing Score'].value_counts(normalize=True) * 100

# Print the percentage of wellbeing scores
print("Percentage of people's wellbeing scores based on the scale:")
for score, percentage in wellbeing_counts.sort_index().items():
    print(f"Score {int(score)}: {percentage:.2f}%")
    
# Print the results
print(f"Gender percentages:\nMale: {male_percentage:.2f}%\nFemale: {female_percentage:.2f}%")
print(f"Minority status percentages:\nMajority: {majority_percentage:.2f}%\nMinority: {minority_percentage:.2f}%")
print(f"Deprived status percentages:\nNon-Deprived: {non_deprived_percentage:.2f}%\nDeprived: {deprived_percentage:.2f}%")
