import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold

# Functions
def plotPredictionVsActual(title, y_test, predictions):
    plt.scatter(y_test, predictions)
    b, a = np.polyfit(y_test, predictions, deg=1)
    plt.legend(loc='best')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title('Predicted (Y) vs. Actual (X): ' + title)
    xseq = np.linspace(0, 100, num=100)
    plt.plot(xseq, a + b*xseq, '-o', color='orange')
    plt.show()

FOLDER = "C:\\datasets\\"
FILE = 'exams.csv'

# Create DataFrame.
df = pd.read_csv(FOLDER + FILE)

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Explore the data, check for missing values
print(df.describe())

# Create an "average score" column
df['average_score'] = round(((df['math score'] + df['reading score'] + df['writing score']) / 3), 1)

# Create X and y
y = df['average_score']

X = df.copy()
del X['math score']
del X['reading score']
del X['writing score']
del X['average_score']
del X['id']

# Create dummy variables
X = pd.get_dummies(X, columns=['gender', 'lunch', 'test preparation course'])
del X['gender_male']
del X['lunch_free/reduced']
del X['test preparation course_none']

X = pd.get_dummies(X, columns=['race/ethnicity', 'parental level of education'])

# Removed the following variables as they were not statistically significant
del X['parental level of education_some college']
del X['parental level of education_high school']

# Adding an intercept
X = sm.add_constant(X)

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

    plotPredictionVsActual("Grades", y_test, y_pred)

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

