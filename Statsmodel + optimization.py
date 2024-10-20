import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the male well-being dataset
df = pd.read_csv('male_wellbeing.csv')

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
fig1.tight_layout()

""" FIGURE 1 """
ax1.scatter(x = df['C_we'], y = df['male_wellbeing_mean'])
ax1.set_xlabel("Number of hours using computers per day on weekends")
ax1.set_ylabel("Mean of self-reported well-being indicators")

ax2.scatter(x = df['C_wk'], y = df['male_wellbeing_mean'])
ax2.set_xlabel("Number of hours using computers per day on weekdays")
ax2.set_ylabel("Mean of self-reported well-being indicators")

ax3.scatter(x = df['G_we'], y = df['male_wellbeing_mean'])
ax3.set_xlabel("Number of hours playing video games per day on weekends")
ax3.set_ylabel("Mean of self-reported well-being indicators")

ax4.scatter(x = df['G_wk'], y = df['male_wellbeing_mean'])
ax4.set_xlabel("Number of hours playing video games per day on weekdays")
ax4.set_ylabel("Mean of self-reported well-being indicators")

plt.show()

""" FIGURE 2 """

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
fig1.tight_layout()
ax1.scatter(x = df['S_we'], y = df['male_wellbeing_mean'])
ax1.set_xlabel("Number of hours using a smartphone per day on weekends")
ax1.set_ylabel("Mean of self-reported well-being indicators")

ax2.scatter(x = df['S_wk'], y = df['male_wellbeing_mean'])
ax2.set_xlabel("Number of hours using a smartphone per day on weekdays")
ax2.set_ylabel("Mean of self-reported well-being indicators")

ax3.scatter(x = df['T_we'], y = df['male_wellbeing_mean'])
ax3.set_xlabel("Number of hours watching TV per day on weekends")
ax3.set_ylabel("Mean of self-reported well-being indicators")

ax4.scatter(x = df['T_wk'], y = df['male_wellbeing_mean'])
ax4.set_xlabel("Number of hours watching TV per day on weekdays")
ax4.set_ylabel("Mean of self-reported well-being indicators")

plt.show()
############################################################################################################################################################

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:, 1:-1]
y = df.iloc[:,-1] # do not add .values beacause we want every values for this multiple linear regression

# Build and evaluate the linear regression model using statsmodels
# to diagonize or overall performance of the the model. so we dont split data into training and test 
x = sm.add_constant(x) ## for intercept to be calculated
model = sm.OLS(y,x).fit() # For distance
pred = model.predict(x)
model_details = model.summary()
print(model_details)

# """Non-linear transformation of T_wk variable"""
# Re-read dataset into a DataFrame
df = pd.read_csv('male_wellbeing.csv')

# Apply non-linear transformation  i.e, log-transform on the T_wk variable
# we will do this to make more linear towards the response variable, so will get much better model
df["LOGSTAT"] = df["T_wk"].apply(np.log)
print(df.info())
print(df["T_wk"].head()) # this is how they look like following the transformation
print(df.info())

# Rearrange the variables so that male_wellbeing_mean appears as the last column
df = df[['ID', 'C_we','C_wk','G_we','G_wk','S_we','S_wk','T_we', 'T_wk', 'LOGSTAT','male_wellbeing_mean']]

# RE-RUN THE LINEAR REGRESSION MODEL WITH A TRANSFORMED VARIABLE
# Drop the original T_wk variable
df = df.drop("T_wk", axis=1)
#####################################################################################################
# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:, 1:-1]
y = df.iloc[:, -1] # do not add .values beacause we want every values for this multiple linear regression
# Build and evaluate the linear regression model using statsmodels
# to diagonize or overall performance of the the model. so we dont split data into training and test 
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)

# # OR
# from sklearn.preprocessing import StandardScaler
# # Standardize the explanatory variables
# scaler = StandardScaler()
# x_scaled = scaler.fit_transform(x)

# # Add a constant for the intercept
# x = sm.add_constant(x_scaled)

# # Build and fit the model
# model = sm.OLS(y, x_scaled).fit()
# print(model.summary())

#######################################################################################################
# Visualise the effect of the transformation
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plt.scatter(df["T_wk"], df['male_wellbeing_mean'], color="green")
plt.title("Original T_wk")
plt.xlabel("T_wk")
plt.ylabel('male_wellbeing_mean')
plt.plot([0,7],[7,0])

plt.subplot(1,2,2)
plt.scatter(df["LOGSTAT"], df['male_wellbeing_mean'], color="red")
plt.title("Log Transformed T_wk")
plt.xlabel("LOGSTAT")
plt.ylabel('male_wellbeing_mean')
plt.plot([0,7],[7,0])

plt.show()

# ##############################################################
# """
# SINCE LOGSTAT WORKS BETTER THAN LSTAT FOLLOWING THE LINEAR TRANSFORMATION:
# 1. WRITE THE TRANSFORMED DATAFRAME INTO A .CSV FILE
# 2. RE-RUN THE LINEAR REGRESSION MODEL USING SKLEARN
# """

# # write dataframe to .csv
# df.to_csv("male_wellbeing_logstat.csv", index=False)

import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load the female well-being dataset
df = pd.read_csv('female_wellbeing.csv')

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
fig1.tight_layout()

""" FIGURE 1 """
ax1.scatter(x = df['C_we'], y = df['female_wellbeing_mean'])
ax1.set_xlabel("Number of hours using computers per day on weekends")
ax1.set_ylabel("Mean of self-reported well-being indicators")

ax2.scatter(x = df['C_wk'], y = df['female_wellbeing_mean'])
ax2.set_xlabel("Number of hours using computers per day on weekdays")
ax2.set_ylabel("Mean of self-reported well-being indicators")

ax3.scatter(x = df['G_we'], y = df['female_wellbeing_mean'])
ax3.set_xlabel("Number of hours playing video games per day on weekends")
ax3.set_ylabel("Mean of self-reported well-being indicators")

ax4.scatter(x = df['G_wk'], y = df['female_wellbeing_mean'])
ax4.set_xlabel("Number of hours playing video games per day on weekdays")
ax4.set_ylabel("Mean of self-reported well-being indicators")

plt.show()

""" FIGURE 2 """

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
fig1.tight_layout()
ax1.scatter(x = df['S_we'], y = df['female_wellbeing_mean'])
ax1.set_xlabel("Number of hours using a smartphone per day on weekends")
ax1.set_ylabel("Mean of self-reported well-being indicators")

ax2.scatter(x = df['S_wk'], y = df['female_wellbeing_mean'])
ax2.set_xlabel("Number of hours using a smartphone per day on weekdays")
ax2.set_ylabel("Mean of self-reported well-being indicators")

ax3.scatter(x = df['T_we'], y = df['female_wellbeing_mean'])
ax3.set_xlabel("Number of hours watching TV per day on weekends")
ax3.set_ylabel("Mean of self-reported well-being indicators")

ax4.scatter(x = df['T_wk'], y = df['female_wellbeing_mean'])
ax4.set_xlabel("Number of hours watching TV per day on weekdays")
ax4.set_ylabel("Mean of self-reported well-being indicators")

plt.show()
############################################################################################################################################################

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:, 1:-1]
y = df.iloc[:,-1] # do not add .values beacause we want every values for this multiple linear regression

# Build and evaluate the linear regression model using statsmodels
# to diagonize or overall performance of the the model. so we dont split data into training and test 
x = sm.add_constant(x) ## for intercept to be calculated
model = sm.OLS(y,x).fit() # For distance
pred = model.predict(x)
model_details = model.summary()
print(model_details)

# """Non-linear transformation of T_wk variable"""
# Re-read dataset into a DataFrame
df = pd.read_csv('female_wellbeing.csv')

# Apply non-linear transformation  i.e, log-transform on the T_wk variable
# we will do this to make more linear towards the response variable, so will get much better model
df["LOGSTAT"] = df["T_wk"].apply(np.log)
print(df.info())
print(df["T_wk"].head()) # this is how they look like following the transformation
print(df.info())

# Rearrange the variables so that female_wellbeing_mean appears as the last column
df = df[['ID', 'C_we','C_wk','G_we','G_wk','S_we','S_wk','T_we', 'T_wk', 'LOGSTAT','female_wellbeing_mean']]

# RE-RUN THE LINEAR REGRESSION MODEL WITH A TRANSFORMED VARIABLE
# Drop the original T_wk variable
df = df.drop("T_wk", axis=1)
#####################################################################################################
# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:, 1:-1]
y = df.iloc[:, -1] # do not add .values beacause we want every values for this multiple linear regression
# Build and evaluate the linear regression model using statsmodels
# to diagonize or overall performance of the the model. so we dont split data into training and test 
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)

# # OR
# from sklearn.preprocessing import StandardScaler
# # Standardize the explanatory variables
# scaler = StandardScaler()
# x_scaled = scaler.fit_transform(x)

# # Add a constant for the intercept
# x = sm.add_constant(x_scaled)

# # Build and fit the model
# model = sm.OLS(y, x_scaled).fit()
# print(model.summary())

#######################################################################################################
# Visualise the effect of the transformation
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plt.scatter(df["T_wk"], df['male_wellbeing_mean'], color="green")
plt.title("Original T_wk")
plt.xlabel("T_wk")
plt.ylabel('male_wellbeing_mean')
plt.plot([0,7],[7,0])

plt.subplot(1,2,2)
plt.scatter(df["LOGSTAT"], df['male_wellbeing_mean'], color="red")
plt.title("Log Transformed T_wk")
plt.xlabel("LOGSTAT")
plt.ylabel('male_wellbeing_mean')
plt.plot([0,7],[7,0])

plt.show()

# ##############################################################
# """
# SINCE LOGSTAT WORKS BETTER THAN LSTAT FOLLOWING THE LINEAR TRANSFORMATION:
# 1. WRITE THE TRANSFORMED DATAFRAME INTO A .CSV FILE
# 2. RE-RUN THE LINEAR REGRESSION MODEL USING SKLEARN
# """

# # write dataframe to .csv
# df.to_csv("female_wellbeing_logstat.csv", index=False)
