import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

"""
BUILD AND EVALUATE LINEAR REGRESSION USING STATSMODELS
"""

# read dataset into a DataFrame
df = pd.read_csv("male_wellbeing.csv")

# separate explanatory variables (x) from the response variable (y)
x = df.iloc[:, 1:-1]
y = df.iloc[:,-1]

# build and evaluate the linear regression model
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)


"""
APPLY Z-SCORE STANDARDISATION
"""
scaler = StandardScaler()

# Drop the previously added constant
x = x.drop(["const"], axis=1)

# Apply z-score standardisation to all explanatory variables
std_x = scaler.fit_transform(x.values)

# Restore the column names of each explanatory variable
std_x_df = pd.DataFrame(std_x, index=x.index, columns=x.columns)
print(std_x_df)
print(std_x_df.head())


"""
REBUILD AND REEVALUATE LINEAR REGRESSION USING STATSMODELS
USING STANDARDISED EXPLANATORY VARIABLES
"""

# Build and evaluate the linear regression model
std_x_df = sm.add_constant(std_x_df)

print(std_x_df)
model = sm.OLS(y,std_x_df).fit()
pred = model.predict(std_x_df)
model_details = model.summary()
print(model_details)


import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

"""
BUILD AND EVALUATE LINEAR REGRESSION USING STATSMODELS
"""

# read dataset into a DataFrame
df = pd.read_csv("female_wellbeing.csv")

# separate explanatory variables (x) from the response variable (y)
x = df.iloc[:, 1:-1]
y = df.iloc[:,-1]

# build and evaluate the linear regression model
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)


"""
APPLY Z-SCORE STANDARDISATION
"""
scaler = StandardScaler()

# Drop the previously added constant
x = x.drop(["const"], axis=1)

# Apply z-score standardisation to all explanatory variables
std_x = scaler.fit_transform(x.values)

# Restore the column names of each explanatory variable
std_x_df = pd.DataFrame(std_x, index=x.index, columns=x.columns)
print(std_x_df)
print(std_x_df.head())


"""
REBUILD AND REEVALUATE LINEAR REGRESSION USING STATSMODELS
USING STANDARDISED EXPLANATORY VARIABLES
"""

# Build and evaluate the linear regression model
std_x_df = sm.add_constant(std_x_df)

print(std_x_df)
model = sm.OLS(y,std_x_df).fit()
pred = model.predict(std_x_df)
model_details = model.summary()
print(model_details)

