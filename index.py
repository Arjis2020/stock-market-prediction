import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
def prepare_data(df, forecast_col, forecast_out, test_size):
    # creating new column called label with the last 5 rows are nan
    label = df[forecast_col].shift(-forecast_out)
    X = np.array(df[[forecast_col]])  # creating the feature array
    X = preprocessing.scale(X)  # processing the feature array
    # creating the column i want to use later in the predicting method
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]  # X that will contain the training and testing
    label.dropna(inplace=True)  # dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=test_size)  # cross validation

    response = [X_train, X_test, Y_train, Y_test, X_lately]
    return response


df = pd.read_csv("prices.csv")  # loading the csv file
df = df[df.symbol == 'GOOG']  # choosing stock symbol


forecast_col = 'close'  # choosing which column to forecast
forecast_out = 5  # how far to forecast
test_size = 0.2  # the size of my test set

# calling the method were the cross validation and data preperation is in
X_train, X_test, Y_train, Y_test, X_lately = prepare_data(
    df, forecast_col, forecast_out, test_size)

# initializing linear regression model
learner = linear_model.LinearRegression()

learner.fit(X_train, Y_train)  # training the linear regression model
score = learner.score(X_test, Y_test)  # testing the linear regression model

# set that will contain the forecasted data
forecast = learner.predict(X_lately)

response = {}  # creting json object
response['accuracy_percent'] = score * 100
response['forecast_set'] = forecast

print(response)
