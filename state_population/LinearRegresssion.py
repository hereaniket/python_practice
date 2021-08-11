import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression

filepath = "/Users/aniketpathak/Documents/IMPORTANT/MS/CST-570/Topic-2/out.csv"
dataTypes = {
    "state_name": "string",
    "year": "int64",
    "population": "int64",
}
raw_data = pd.read_csv(filepath_or_buffer=filepath, dtype=dataTypes)

#print(raw_data['state_name'].rank(method='max'))

data = raw_data[['year', 'population']]  # load data set
X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
Y = data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions


plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()