import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler


# feature selection
def select_features(X_train, y_train, X_test, k):
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k=k)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


PATH2 = "C:\\datasets\\crime\\"
CSV_DATA = "vanCensus.csv"
sc_x = StandardScaler()
sc_y = StandardScaler()

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv(PATH2 + CSV_DATA)

# Remove columns with a sum of 0.0
col_name = list(df.columns)
for i in range(0, len(col_name)):
    if df[col_name[i]].sum() == 0.0:
        del df[col_name[i]]

# Remove columns with a sum < 200.0
col_name = list(df.columns)
for i in range(0, len(col_name)):
    if df[col_name[i]].sum() <= 200.0:
        del df[col_name[i]]

# Imputing null values
imputer = KNNImputer(n_neighbors=10)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Get X and y variables
columns = list(df.columns)
y = df["total_crimes_2016"]
X = df[columns]
del X["Dissemination Number"]
del X["total_crimes_2016"]

columns2 = []
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, 'all')

# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
    if fs.scores_[i] > 20.0:
        columns2.append(list(X.columns)[i])

# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

X = X[columns2]
print(columns2)

# MODEL 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

y_train = sc_y.fit_transform(np.array(y_train).reshape(-1, 1))

model = sm.OLS(y_train, X_train, hasconst=True).fit()
unscaledPredictions = model.predict(X_test)  # make the predictions by the model
yhat = sc_y.inverse_transform(np.array(unscaledPredictions).reshape(-1, 1))
rmse = np.sqrt(mean_squared_error(y_test, yhat))
print(model.summary())
print(rmse)

# Choose the variables with p-value < 0.05
columns3 = []
for i in range(0, len(model.pvalues)):
    if model.pvalues[i] <= 0.05:
        columns3.append(columns2[i])
print(columns3)

# MODEL 2
X = X[columns3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

y_train = sc_y.fit_transform(np.array(y_train).reshape(-1, 1))

model = sm.OLS(y_train, X_train, hasconst=True).fit()
unscaledPredictions = model.predict(X_test)  # make the predictions by the model
yhat = sc_y.inverse_transform(np.array(unscaledPredictions).reshape(-1, 1))
rmse = np.sqrt(mean_squared_error(y_test, yhat))
print(model.summary())
print(rmse)

# Choose the variables with p-value < 0.05
columns4 = []
for i in range(0, len(model.pvalues)):
    if model.pvalues[i] <= 0.05:
        columns4.append(columns3[i])

print(columns4)

# FINAL MODEL
X = X[columns4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

y_train = sc_y.fit_transform(np.array(y_train).reshape(-1, 1))

model = sm.OLS(y_train, X_train, hasconst=True).fit()
unscaledPredictions = model.predict(X_test)  # make the predictions by the model
yhat = sc_y.inverse_transform(np.array(unscaledPredictions).reshape(-1, 1))
rmse = np.sqrt(mean_squared_error(y_test, yhat))
print(model.summary())
print(rmse)