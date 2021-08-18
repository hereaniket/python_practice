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
# print(sub_set['total_charges'].describe())

# Converting to numbers
sub_set['monthly_charges'] = pd.to_numeric(sub_set['monthly_charges'])
sub_set['total_charges'] = pd.to_numeric(sub_set['total_charges'])
# print(sub_set.info())

# Converting the required columns to integer
for i in sub_set.columns:
    sub_set[i] = sub_set[i].astype(int)
# print(sub_set.info())

# taking them to X and Y, X is being independent vars and Y is dependent
X_var = np.asarray(sub_set[['tenure', 'monthly_charges', 'total_charges']])
# print(cl('X_var samples : ', attrs=['bold']), X_var[:5])
y_var = np.asarray(sub_set['churn'])
# print(cl('y_var samples : ', attrs=['bold']), y_var[:5])

# Doing normalization
X_var = StandardScaler().fit(X_var).transform(X_var)
# print(cl(X_var[:5], attrs=['bold']))


X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.3, random_state=4)
# print(cl('X_train samples : ', attrs=['bold']), X_train[:5])
# print(cl('X_test samples : ', attrs=['bold']), X_test[:5])
# print(cl('y_train samples : ', attrs=['bold']), y_train[:10])
# print(cl('y_test samples : ', attrs=['bold']), y_test[:10])


# Modelling
lr = LogisticRegression(C=0.1, solver='liblinear')
lr.fit(X_train, y_train)

print(cl(lr, attrs=['bold']))

# Predictions
yhat = lr.predict(X_test)
yhat_prob = lr.predict_proba(X_test)

print(cl('yhat samples : ', attrs=['bold']), yhat[:10])
print(cl('yhat_prob samples : ', attrs=['bold']), yhat_prob[:10])

# Precision Score
print(cl('Precision Score of our model is {}'.format(precision_score(y_test, yhat).round(2)), attrs=['bold']))
