import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# Functions
def plotPredictionVsActual(title, xlabel, ylabel, y_test, predictions):
    plt.scatter(y_test, predictions)
    b, a = np.polyfit(y_test, predictions, deg=1)
    plt.legend(loc='best')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    xseq = np.linspace(0, 1, num=100)
    plt.plot(xseq, a + b*xseq, '-', color='orange')
    plt.show()

FOLDER = "C:\\datasets\\"  # Windows
FILE = 'exams.csv'

# Create DataFrame.
df = pd.read_csv(FOLDER + FILE)

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Create an "average score" column
df['average_score'] = round(((df['math score'] + df['reading score'] + df['writing score']) / 3), 1)


# Create dummy variables
df = pd.get_dummies(df, columns=['gender', 'race/ethnicity', 'parental level of education',
                               'lunch', 'test preparation course'])

count = 0
while count < 2:
    if count == 0:
        X = df['test preparation course_completed']
    if count == 1:
        X = df['gender_female']
    y = df['average_score']
    x = X.copy()

    # Adding an intercept
    X = sm.add_constant(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Build and evaluate model.
    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model
    print(model.summary())
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    if count == 0:
        plotPredictionVsActual('The Effect of Test Prep Completion on Exam Marks'
                               , 'Test Prep Completion (0 = not completed  1 = completed)'
                               , 'Marks', x, y)
    if count == 1:
        plotPredictionVsActual('The Effect of Gender on Exam Marks'
                               , 'Gender (0 = male  1 = female)'
                               , 'Marks', x, y)
    count += 1