import pandas as pd # data processing
import numpy as np # working with arrays
import itertools # construct specialized tools
import matplotlib.pyplot as plt # visualizations
from matplotlib import rcParams # plot size customization
from termcolor import colored as cl # text customization
from sklearn.model_selection import train_test_split # splitting the data
from sklearn.linear_model import LogisticRegression # model algorithm
from sklearn.preprocessing import StandardScaler # data normalization
# from sklearn.metrics import jaccard_similarity_score as jss # evaluation metric
from sklearn.metrics import precision_score # evaluation metric
from sklearn.metrics import classification_report # evaluation metric
from sklearn.metrics import confusion_matrix # evaluation metric
from sklearn.metrics import log_loss # evaluation metric
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

data_type = {
    "customer_id": "string",
    "gender": "string",
    "senior_citizen": "string",
    "partner": "string",
    "dependents": "string",
    "tenure": "string",
    "phone_service": "string",
    "multiple_lines": "string",
    "internet_service": "string",
    "online_security": "string",
    "online_backup": "string",
    "device_protection": "string",
    "tech_support": "string",
    "streaming_service": "string",
    "streaming_movies": "string",
    "contract": "string",
    "paperless_billing": "string",
    "payment_method": "string",
    "monthly_charges": "string",
    "total_charges": "string",
    "churn": "string"
}

filepath = "data/Data_Formatted.csv"

data = pd.read_csv(filepath_or_buffer=filepath, dtype=data_type)
# print(data.info())

# Generating the subset with three dependent variables
sub_set = data[['tenure', 'monthly_charges', 'total_charges', 'churn']]
print(sub_set['total_charges'].describe())

# Converting to numbers
sub_set['monthly_charges'] = pd.to_numeric(sub_set['monthly_charges'])
sub_set['total_charges'] = pd.to_numeric(sub_set['total_charges'])
# print(sub_set.info())

# Converting the required columns to integer
for i in sub_set.columns:
    sub_set[i] = sub_set[i].astype(int)
# print(sub_set.info())

numerical_features = sub_set[['tenure', 'monthly_charges', 'total_charges']]
target = 'churn'

print(numerical_features.describe())

numerical_features.hist(bins=30, figsize=(10, 7))

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
sub_set[sub_set.churn == "No"][numerical_features].hist(bins=30, color="blue", alpha=0.5, ax=ax)
sub_set[sub_set.churn == "Yes"][numerical_features].hist(bins=30, color="red", alpha=0.5, ax=ax)

ROWS, COLS = 4, 4
fig, ax = plt.subplots(ROWS, COLS, figsize=(18, 18))
row, col = 0, 0
for i, categorical_feature in enumerate(categorical_features):
    if col == COLS - 1:
        row += 1
    col = i % COLS
    sub_set[categorical_feature].value_counts().plot('bar', ax=ax[row, col]).set_title(categorical_feature)

sub_set[target].value_counts().plot('bar').set_title('churned')


from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)
pipeline.fit(df_train, df_train[target])
pred = pipeline.predict(df_test)
















