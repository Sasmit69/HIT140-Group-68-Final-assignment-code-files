import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns

# Load the datasets
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')

# # Display the first few rows to inspect the data
# print(df1.head())
# print(df2.head())
# print(df3.head())

# Summary statistics for demographics (df1)
print(df1.describe())

# Summary statistics for screen time (df2)
print(df2.describe())

# Summary statistics for well-being scores (df3)
print(df3.describe())

#######################################################################################################################
# Histograms for screen time variables (df2)
df2[['C_wk', 'G_wk', 'S_wk', 'T_wk']].hist(bins=20, figsize=(10,8))
plt.suptitle('Screen Time Distribution (Weekdays)')
plt.show()

# Histograms for well-being scores (e.g., df3)
df3.iloc[:, 1:].hist(bins=10, figsize=(10,10))
plt.suptitle('Well-being Scores Distribution')
plt.show()

# Histograms for weekends screen time variables (df2)
df2[['C_we', 'G_we', 'S_we', 'T_we']].hist(bins=20, figsize=(10,8))
plt.suptitle('Screen Time Distribution (Weekends)')
plt.show()

# Histograms for well-being scores (e.g., df3)
df3.iloc[:, 1:].hist(bins=10, figsize=(10,10))
plt.suptitle('Well-being Scores Distribution')
plt.show()

###########################################################################################################################################
# Merge screen time (df2) with well-being (df3) data on ID
df_merged23 = pd.merge(df2, df3, on='ID')

# Merge df1 (demographics) with df2 (screen time) and df3 (well-being)
df_merged12 = pd.merge(df1[['ID', 'gender']], df2, on='ID')  # Include gender in the merge
df_merged123 = pd.merge(df_merged12, df3, on='ID')
##########################################################
# HItmap which shows ID as well. we don't need this for our report
# Correlation matrix
# correlation_matrix = df_merged23.corr()
variables = ['C_wk', 'G_wk', 'S_wk', 'T_wk', 'C_we', 'G_we', 'S_we', 'T_we', 'T_we', 'Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
correlation_matrix = df_merged123[variables].corr()
# Plot heatmap of correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix between Screen Time and Well-being Scores')
plt.xticks(rotation=45)
plt.show()
######################################################################################################################################################
# Grouping by gender and calculating the mean for screen time variables (weekdays)
mean_values = df_merged123.groupby('gender')[['C_wk', 'G_wk', 'S_wk', 'T_wk']].mean()

# Plot the bar chart and capture the BarContainer object
ax = mean_values.plot(kind='bar', figsize=(10,6))

plt.title('Average Screen Time by Gender (Weekdays)')
plt.ylabel('Hours of Screen Time')
plt.xticks(rotation=0)
plt.legend(title="Screening time indicators")

# Add labels to each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3)

plt.show()

# Grouping by gender and calculating the mean for screen time variables (weekends)
mean_values = df_merged123.groupby('gender')[['C_we', 'G_we', 'S_we', 'T_we']].mean()

# Plot the bar chart and capture the BarContainer object
ax = mean_values.plot(kind='bar', figsize=(10,6))

plt.title('Average Screen Time by Gender (Weekends)')
plt.ylabel('Hours of Screen Time')
plt.xticks(rotation=0)
plt.legend(title="Screening time indicators")

# Add labels to each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3)

plt.show()


# Grouped analysis by deprivation (df1 and df_merged23)
df_deprived = pd.merge(df1[['ID', 'deprived']], df_merged23, on='ID')
mean_values = df_deprived.groupby('deprived').mean()[["Optm", "Usef", "Relx", "Intp", "Engs", "Dealpr", "Thcklr", "Goodme", "Clsep", "Conf",
                     "Mkmind", "Loved", "Intthg", "Cheer"]]
# Define parameters for bar positions and width
x = np.arange(len(mean_values.index))  # the label locations (deprivation status)
width = 0.25  # the width of the bars
colors = ["blue", "orange", "green", "yellow", "purple", "red", "brown", "grey", "skyblue", "pink", "lightblue", "salmon", "lightgreen", "black"]
# Plot the bar chart and capture the axes object
ax = mean_values.plot(kind='bar', width=0.9, color=colors, figsize=(30, 6))
plt.title('Well-being Scores by Deprivation Status')
plt.ylabel('Average Well-being Score')
plt.xticks(rotation=0)
plt.legend(title="Well-being Indicators")

# Add labels to each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3)

plt.show()

#############################################################################################################################################################################
# # Outliers Detection
# Selecting the weekday screen time columns
sample_wk = df2[['C_wk', 'G_wk', 'S_wk', 'T_wk']]

# Melt the DataFrame to a long format suitable for Seaborn
sample_wk_melted = sample_wk.melt(var_name='Screen Time Type', value_name='Hours')

# Create a boxplot with labels
sns.boxplot(x='Screen Time Type', y='Hours', data=sample_wk_melted)
plt.title("Boxplot of Screen Time by Activity (Weekdays)")
plt.ylabel("Hours of Screen Time")
plt.xlabel("Screen Time Type")
plt.show()


############################################################################################################################################
# # Function to identify outliers based on IQR # need extra task or code to be job done
# def separate_outliers(data, threshold=1.5):
#     outliers_dict = {}
#     non_outliers_dict = {}

#     for column in data.columns:
#         # Calculate Q1 (25th percentile) and Q3 (75th percentile)
#         Q1 = data[column].quantile(0.25)
#         Q3 = data[column].quantile(0.75)
#         IQR = Q3 - Q1

#         # Calculate the bounds for outliers
#         lower_bound = Q1 - threshold * IQR
#         upper_bound = Q3 + threshold * IQR

#         # Separate outliers and non-outliers
#         outliers = data[column][(data[column] < lower_bound) | (data[column] > upper_bound)].tolist()
#         non_outliers = data[column][(data[column] >= lower_bound) & (data[column] <= upper_bound)].tolist()

#         # Store results in dictionaries
#         outliers_dict[column] = outliers
#         non_outliers_dict[column] = non_outliers

#     return outliers_dict, non_outliers_dict

# # Apply the function to separate outliers and non-outliers
# outliers, non_outliers = separate_outliers(sample_wk)

# # Print the results
# for col in sample_wk.columns:
#     print(f"\nOutliers in {col} (unique values): {outliers[col]}")
#     print(f"Non-outliers in {col}: {non_outliers[col]}")
###################################################################################################################################################################


#  Selecting the weekends screen time columns
sample_we = df2[['C_we', 'G_we', 'S_we', 'T_we']]
# Melt the DataFrame to a long format suitable for Seaborn
sample_we_melted = sample_we.melt(var_name='Screen Time Type', value_name='Hours')

# Create a boxplot with labels
sns.boxplot(x='Screen Time Type', y='Hours', data=sample_we_melted)
plt.title("Boxplot of Screen Time by Activity (Weekdays)")
plt.ylabel("Hours of Screen Time")
plt.xlabel("Screen Time Type")
plt.show()

# Box plot for well-being indicators
df3.iloc[:, 1:].plot(kind='box', figsize=(10,6))
plt.title('Well-being Scores Outliers')
plt.show()

##########################################################################
# Create total screen time for weekdays and weekends
df2['total_weekday_screen_time'] = df2['C_wk'] + df2['G_wk'] + df2['S_wk'] + df2['T_wk']
df2['total_weekend_screen_time'] = df2['C_we'] + df2['G_we'] + df2['S_we'] + df2['T_we']

# Merge with well-being scores and demographic data
df_final = pd.merge(pd.merge(df1, df2, on='ID'), df3, on='ID')

# Check correlation with new features
print(df_final[['total_weekday_screen_time', 'total_weekend_screen_time']].corr())
