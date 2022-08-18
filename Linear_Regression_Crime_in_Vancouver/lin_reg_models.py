import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import metrics
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def getUnfitModels():
    models = list()
    models.append(ElasticNet())
    models.append(SVR(gamma='scale'))
    models.append(DecisionTreeRegressor())
    models.append(AdaBoostRegressor())
    models.append(RandomForestRegressor(n_estimators=10))
    models.append(ExtraTreesRegressor(n_estimators=10))
    return models


def evaluateModel(y_test, predictions, model):
    mse = mean_squared_error(y_test, predictions)
    rmse = round(np.sqrt(mse), 3)
    print(" RMSE:" + str(rmse) + " " + model.__class__.__name__)


def fitBaseModels(X_train, y_train, X_test, models):
    dfPredictions = pd.DataFrame()

    # Fit base model and store its predictions in dataframe.
    for i in range(0, len(models)):
        models[i].fit(X_train, y_train)
        predictions = models[i].predict(X_test)
        colName = str(i)
        # Add base model predictions to column of data frame.
        dfPredictions[colName] = predictions
    return dfPredictions, models


def fitStackedModel(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


PATH2 = "C:\\datasets\\crime\\"
CSV_DATA = "vanCensus.csv"
sc_x = StandardScaler()
sc_y = StandardScaler()
count = 0

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv(PATH2 + CSV_DATA)

# Imputing null values
imputer = KNNImputer(n_neighbors=10)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Get X and y variables
columns = list(df.columns)
y = df["total_crimes_2016"]
X = df[columns]
del X["Dissemination Number"]
del X["total_crimes_2016"]

columns = ['Multiple Aboriginal and non-Aboriginal ancestries', 'No bedrooms', '1991 to 2000.1', 'Punjabi (Panjabi).2']
X = df[columns]
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

###################################################################################################
# Original Model
###################################################################################################
model = sm.OLS(y_train, X_train, hasconst=True).fit()
predictions = model.predict(X_test)  # make the predictions by the model
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(model.summary())
print(rmse)

###################################################################################################
# Model with scaling
###################################################################################################
scalers = [RobustScaler(), StandardScaler(), MinMaxScaler()]
scaler_name = ["Robust", "Standard", "MinMax"]
r2 = []
adj_r2 = []
aic = []
bic = []
rmse_list = []

for s in range(0, len(scalers)):
    sc_x = scalers[s]
    sc_y = scalers[s]

    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    X_train = sc_x.fit_transform(X_train)
    X_test = sc_x.transform(X_test)
    y_train = sc_y.fit_transform(np.array(y_train).reshape(-1, 1))

    model = sm.OLS(y_train, X_train).fit()
    unscaledPredictions = model.predict(X_test)  # make the predictions by the model
    yhat = sc_y.inverse_transform(np.array(unscaledPredictions).reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_test, yhat))

    r2.append(model.rsquared)
    adj_r2.append(model.rsquared_adj)
    aic.append(model.aic)
    bic.append(model.bic)
    rmse_list.append(rmse)
    print(model.summary())
    print(rmse)

d = {'Scaler': scaler_name, 'R2': r2, 'R2_Adj': adj_r2, 'AIC': aic, 'BIC': bic, 'rmse': rmse_list}
scaler_stats = pd.DataFrame(data=d)
print(scaler_stats)

###################################################################################################
# Model with Stacking
###################################################################################################

# Split data into train, test and validation sets.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

# Get base models.
unfitModels = getUnfitModels()

# Fit base and stacked models.
dfPredictions, models = fitBaseModels(X_train, y_train, X_test, unfitModels)
stackedModel = fitStackedModel(dfPredictions, y_test)

# Evaluate base models with validation data.
print("\n** Evaluate Base Models **")
dfValidationPredictions = pd.DataFrame()
for i in range(0, len(models)):
    predictions = models[i].predict(X_val)
    colName = str(i)
    dfValidationPredictions[colName] = predictions
    evaluateModel(y_val, predictions, models[i])

# Evaluate stacked model with validation data.
stackedPredictions = stackedModel.predict(dfValidationPredictions)
print("\n** Evaluate Stacked Model **")
evaluateModel(y_val, stackedPredictions, stackedModel)

###################################################################################################
# Model with K-Fold Validation
###################################################################################################
# prepare cross validation with three folds.
kfold = KFold(n_splits=8, shuffle=True)
rmseList = []
bicList = []
rsquareLst = []
count = 1

for train_index, test_index in kfold.split(X):
    X_train = X.loc[X.index.isin(train_index)]
    X_test = X.loc[X.index.isin(test_index)]
    y_train = y.loc[y.index.isin(train_index)]
    y_test = y.loc[y.index.isin(test_index)]

    # Perform linear regression.
    model = sm.OLS(y_train, X_train).fit()
    print(model.summary())

    y_pred = model.predict(X_test)  # make the predictions by the model
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    rmseList.append(rmse)
    bic = model.bic
    bicList.append(bic)
    rsqr = model.rsquared
    rsquareLst.append(rsqr)

    print("\n***K-fold: " + str(count))
    print("RMSE:     " + str(rmse))
    print("BIC:      " + str(bic))
    print("R^2:      " + str(rsqr))

    count += 1

# Show averages of scores over multiple runs.
print("*********************************************")
print("\nScores for all folds:")
print("*********************************************")
print("RMSE Average :   " + str(np.mean(rmseList)))
print("RMSE SD:         " + str(np.std(rmseList)))
print("BIC Average :    " + str(np.mean(bicList)))
print("BIC SD:          " + str(np.std(bicList)))
print("RSQ Average :    " + str(np.mean(rsquareLst)))
print("RSQ SD:          " + str(np.std(rsquareLst)))

###################################################################################################
# Final Model
###################################################################################################
scalers = RobustScaler()
scaler_name = ["Robust", "Standard", "MinMax"]

sc_x = scalers
sc_y = scalers

X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
y_train = sc_y.fit_transform(np.array(y_train).reshape(-1, 1))

# prepare cross validation with three folds.
kfold = KFold(n_splits=8, shuffle=True)
rmseList = []
bicList = []
rsquareLst = []
count = 1

for train_index, test_index in kfold.split(X):
    X_train = X.loc[X.index.isin(train_index)]
    X_test = X.loc[X.index.isin(test_index)]
    y_train = y.loc[y.index.isin(train_index)]
    y_test = y.loc[y.index.isin(test_index)]

    # Perform linear regression.
    model = sm.OLS(y_train, X_train).fit()
    print(model.summary())

    y_pred = model.predict(X_test)  # make the predictions by the model
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    rmseList.append(rmse)
    bic = model.bic
    bicList.append(bic)
    rsqr = model.rsquared
    rsquareLst.append(rsqr)

    print("\n***K-fold: " + str(count))
    print("RMSE:     " + str(rmse))
    print("BIC:      " + str(bic))
    print("R^2:      " + str(rsqr))

    count += 1

# Show averages of scores over multiple runs.
print("*********************************************")
print("\nScores for all folds:")
print("*********************************************")
print("RMSE Average :   " + str(np.mean(rmseList)))
print("RMSE SD:         " + str(np.std(rmseList)))
print("BIC Average :    " + str(np.mean(bicList)))
print("BIC SD:          " + str(np.std(bicList)))
print("RSQ Average :    " + str(np.mean(rsquareLst)))
print("RSQ SD:          " + str(np.std(rsquareLst)))

