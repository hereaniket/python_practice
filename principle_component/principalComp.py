# Importing pandas and seaborn libraries for data manipulation and charting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Creating a pipeline to obtain principal components for Elbow rule
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Import data from CSV file
data = pd.read_csv('../data/customer_churn/Data.csv')


# Function to clean column names
def column_name_remove_space(df):
    for x in df.columns:
        if " " in x:
            df = df.rename(columns={x: x.replace(" ", "_").replace("(", "")
                           .replace(")", "").replace(",", "_").replace("/", "_")})
    return df


# Creating a copy of the data frame with existing categorical variables
data_cat = data[['Gender', 'Churn', 'Techie', 'Contract', 'Port_modem', 'Area', 'Tablet', 'Phone',
                 'OnlineSecurity', 'Multiple', 'OnlineBackup', 'TechSupport', 'DeviceProtection', 'StreamingTV',
                 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod', 'Marital', 'InternetService']].copy()

# Following data columns are not important for data analysis (categorical data, geolocation data etc)
to_drop = ['City', 'County', 'Zip', 'Job', 'TimeZone',
           'Lat', 'Lng', 'UID', 'Customer_id', 'Interaction', 'CaseOrder', 'State', 'Gender',
           'Churn', 'Techie', 'Contract', 'Port_modem', 'Area', 'Tablet', 'Phone',
           'OnlineSecurity', 'Multiple', 'OnlineBackup', 'TechSupport', 'DeviceProtection', 'StreamingTV',
           'StreamingMovies', 'PaperlessBilling', 'PaymentMethod', 'Marital', 'InternetService']
# Fix all columns
data = column_name_remove_space(data)
data.drop(columns=to_drop, inplace=True)

# Check for null values
data.isna().any(axis=0).any()

# Checking number of unique values in each column
data.nunique()

# Check data by printing first few rows
data.head()

# Check data by printing last few rows
data.tail()

data.to_csv('clean_data.csv')

sns.heatmap(data.corr())

pipe1 = Pipeline([('scaler', StandardScaler()), ('reducer', PCA())])
pc = pipe1.fit_transform(data)

# Print principal components
pc

# Obtained variances for scree plot
variances = pipe1.steps[1][1].explained_variance_ratio_ * 100
variances

# Cumulative sum of variances
np.cumsum(variances)

# Components
pipe1.steps[1][1].components_

data.columns

# PCA components are orthogonal, their dot product is zero
print(np.dot(pipe1.steps[1][1].components_[1], pipe1.steps[1][1].components_[2]).round(3))
print(np.dot(pipe1.steps[1][1].components_[2], pipe1.steps[1][1].components_[3]).round(3))
print(np.dot(pipe1.steps[1][1].components_[1], pipe1.steps[1][1].components_[3]).round(3))
print(np.dot(pipe1.steps[1][1].components_[1], pipe1.steps[1][1].components_[4]).round(3))
print(np.dot(pipe1.steps[1][1].components_[2], pipe1.steps[1][1].components_[4]).round(3))
print(np.dot(pipe1.steps[1][1].components_[3], pipe1.steps[1][1].components_[4]).round(3))

# Scree plot, elbow rule
plt.plot(variances)

# Applying PCA without normalization can give highly flawed results
# Here it without normalization, it looks like the first component accounts for close
# to 80% variability in the data

pipe1x = Pipeline([('reducer', PCA())])
pcx = pipe1x.fit_transform(data)
plt.plot(pipe1x.steps[0][1].explained_variance_ratio_)

# Component coffecient have a big range of orders of magnitude
# Never use PCA without normalization or Standardization of the field values

pipe1x.steps[0][1].components_

# Based on scree plot creating PCA with appropriate number of components
pipe2 = Pipeline([('scaler', StandardScaler()), ('reducer', PCA(n_components=2))])
pc = pipe2.fit_transform(data)
cols = data_cat.columns

data_cat['pc1'] = pc[:, 0]
data_cat['pc2'] = pc[:, 1]
cols

# Printing components along with the field names for the 2 principal components
pc1 = dict(zip(data.columns, pipe2.steps[1][1].components_[0]))
print(dict(sorted(pc1.items(), key=lambda item: item[1])))
pc2 = dict(zip(data.columns, pipe2.steps[1][1].components_[1]))
print(dict(sorted(pc2.items(), key=lambda item: item[1])))

# Creating plots for principal components, colored (legend) based on the other categorical variables
for col in cols:
    sns.scatterplot(data=data_cat, x='pc1', y='pc2', hue=col)
    plt.show()

# Creating train and test split data


y = data_cat['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.25)

# Printing train test split variables
print(X_train.head(5))
print(X_test.head(5))
print(y_train.head(5))
print(y_test.head(5))

# Using LogisticRegression along with PCA to compute churn prediction accuracy


for i in range(2, 12):
    pipe2 = Pipeline(
        [('scaler', StandardScaler()), ('reducer', PCA(n_components=i)), ('classifier', LogisticRegression())])
    pipe2.fit(X_train, y_train)
    print(i, pipe2.score(X_test, y_test))

# Using LogisticRegression to compute churn prediction accuracy without PCA
pipe4 = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression())])
pipe4.fit(X_train, y_train)
pipe4.score(X_test, y_test)

for i in range(2, 12):
    pipe5 = Pipeline(
        [('scaler', StandardScaler()), ('reducer', PCA(n_components=i)), ('classifier', RandomForestClassifier())])
    pipe5.fit(X_train, y_train)
    print(i, pipe5.score(X_test, y_test))

# Using RandomForestClassifier to compute churn prediction accuracy without PCA
pipe6 = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier())])
pipe6.fit(X_train, y_train)
pipe6.score(X_test, y_test)
