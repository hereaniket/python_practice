import pandas as pd # data processing
import numpy as np # working with arrays
from termcolor import colored as cl # text customization
from sklearn.model_selection import train_test_split # splitting the data
from sklearn.linear_model import LogisticRegression # model algorithm
from sklearn.preprocessing import StandardScaler # data normalization
# from sklearn.metrics import jaccard_similarity_score as jss # evaluation metric
from sklearn.metrics import precision_score # evaluation metric

# Data set for STA and AGE
data = pd.read_csv(filepath_or_buffer="data/CST-570-RS-ICU.csv")[['STA', 'AGE']]

print(data.columns)
print(data.info())

# taking them to X and Y, X is being independent vars and Y is dependent
X_var = np.asarray(data[['AGE']])
y_var = np.asarray(data['STA'])


# Doing normalization
X_var = StandardScaler().fit(X_var).transform(X_var)

X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.3, random_state=4)

# Modelling
lr = LogisticRegression(C=0.1, solver='liblinear')
lr.fit(X_train, y_train)

print(cl(lr, attrs=['bold']))

# Predictions
yhat = lr.predict(X_test)
yhat_prob = lr.predict_proba(X_test)

print(cl('yhat samples : ', attrs=['bold']), yhat[:500])
print(cl('yhat_prob samples : ', attrs=['bold']), yhat_prob[:500])

# Precision Score
print(cl('Precision Score of our model is {}'.format(precision_score(y_test, yhat).round(2)), attrs=['bold']))



