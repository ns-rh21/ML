# ML
Machine learning codes
# Classification Algorithms in Python


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
#%pip install mplfinance
import mplfinance as fplt
import os

## Understanding Business Question

Algorithmic Trading

Billionaire Mathematician - Jim Simons

https://www.youtube.com/watch?v=gjVDqfUhXOY

## Read Data from Yahoo Finance API

#Get data directly from Yahoo Finance website: 
#https://uk.finance.yahoo.com/most-active

#Install Pandas Module needed to Connect to Yahoo Finance API
#%pip install pandas-datareader
#%pip install --upgrade pandas
#%pip install --upgrade pandas-datareader
import pandas_datareader as pdr

#Load data from Yahoo Finance API
#data = pdr.get_data_yahoo("^GSPC", start = "2017-01-01", end = "2021-12-31")
os.chdir('E:\\داده\\project')
#data = pd.read_csv('weekly.csv')

## Data Visualization of Financial Data

Time series / date functionality in Pandas

https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

#data.loc['2017-01-04', :]

#data.loc[data['Year']==1990,:]


Technical Analysis Library in Python

https://technical-analysis-library-in-python.readthedocs.io/en/latest/index.html

#%pip install ta
#import ta

#Simple Moving Average(SMA)
# https://www.investopedia.com/terms/s/sma.asp
sma = ta.trend.sma_indicator(data['Volume'], 20)

sma.head(21)

sma.tail()

#Exponential Moving Average(EMA)
# https://www.investopedia.com/terms/e/ema.asp
#ema = ta.trend.ema_indicator(data['Volume'], 20)
%pip install cv2
import cv2
rio=cv2.selectROI(data['Today'])

#Relative Strength Index (RSI)
# https://www.investopedia.com/terms/r/rsi.asp
#rsi = ta.momentum.rsi(data['Today'], 3)

## Data Preparation¶
 

#data['rsi'] = rsi
data.info()

#Calculate daily return
#data['d_return'] = 0

import datetime as dt
df = pd.DataFrame({'date':pd.date_range(start='1/1/1990',end='31/12/2010',freq='W')})
df["week"]=df['date'].dt.isocalendar().week
df["Year"]=df['date'].dt.isocalendar().year

df.drop([0,1,2,3,4],inplace=True)
#data.shape,df.shape
df.reset_index(drop=True,inplace=True)
total=pd.merge(data,df,left_index=True,right_index=True)
total.shape
#total.head()
total.to_csv('total.csv')

#data=pd.read_csv('total.csv')

#data.set_index("date",inplace=True)
#del data["Year_y"]
#del data["rsi"]
#del data["Unnamed: 0"]
#data.head()
#Calculate volume change
#data['volume_change'] = 0
data.head()

for i in range(1, data.shape[0]):
    data.iloc[i, 10] = data.iloc[i, 6] / data.iloc[i - 1, 6] - 1
data.head()

#Plot daily return
data['volume_change'].plot()

#Plot daily return
sns.histplot(data['Today'], stat = 'probability', 
             kde = True, alpha = 0.7, color = 'green',
             bins = np.linspace(min(data.Today), max(data.Today), 20))

#daily return lags: lag 1
data['r_lag1'] = data['d_return'].shift(1)
#data.head()

#daily return lags: lag 2
data['r_lag2'] = data['d_return'].shift(2)

#daily return lags: lag 3
data['r_lag3'] = data['d_return'].shift(3)

#daily return lags: lag 4
data['r_lag4'] = data['d_return'].shift(4)

#daily return lags: lag 5
data['r_lag5'] = data['d_return'].shift(5)
data.head(10)

#volume change lag: lag 1
data['v_lag1'] = data['volume_change'].shift(1)
data.head()

#volume change lag: lag 2
data['v_lag2'] = data['volume_change'].shift(2)

#volume change lag: lag 3
data['v_lag3'] = data['volume_change'].shift(3)

#volume change lag: lag 4
data['v_lag4'] = data['volume_change'].shift(4)

#volume change lag: lag 5
data['v_lag5'] = data['volume_change'].shift(5)
data.head(10)

#Market trend
#This method is good for complex conditions(here we have simple conditions!)
#create a list of conditions
conditions = [data['Direction'] =='Down',
              data['Direction'] =='Up']

#create a list of the values needed to assign for each condition
values = [0, 1]

#create a new column and use np.select to assign values to it using the lists as arguments
data['trend'] = np.select(conditions, values)
data.head(10)

#remove first 6 rows
data.drop(index = data.index[list(range(6))], inplace = True)
data.head()

## Correlation Analysis¶ 

#corr btw daily return and rsi & d_return lags
corr_table1 = round(data.iloc[:, [1,2,3,4,5,6,7]].corr(method = 'pearson'), 2)
corr_table1

#corr btw daily return and v_change lags
corr_table2 = round(data.iloc[:, [11,12,13,14,7,16]].corr(method = 'pearson'), 2)
corr_table2

#Save data for next session
#Export Data to CSV File
data.to_csv('sp500_data.csv', index = True)

## Read Data from File

data = pd.read_csv('sp500_data.csv')

#data.isna().sum()

## Divide Dataset into Train and Test and Real 

#data.set_index(pd.to_datetime(data['Date'], format = '%Y-%m-%d'), inplace = True) 

train = data.loc['1990':'2007', ]
train

test = data.loc['2008':'2010', ]
test

#real = data.loc[data['Date'] >= '2019', ]
#real

## Prediction Models 

### Logistic Regression 

train.info()

#Define the feature set X 
X_train = train.loc[:, ['Today', 
                        'Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5',
                        'v_lag1', 'v_lag2', 'v_lag3', 'v_lag4', 'v_lag5']]
X_train = sm.add_constant(X_train) #adding a constant

#Define response variable
y_train = train.loc[:, 'trend']

X_train.head()

y_train.head()

import statsmodels.api as sm
model_lr = sm.Logit(y_train, X_train).fit()
print(model_lr.summary())

#Define the feature set X 
X_train = train.loc[:, ['Today', 
                        'Lag1', 'Lag2', 'Lag3', 'Lag4','Lag5']]

X_train = sm.add_constant(X_train) #adding a constant
X_train.head()

model_lr = sm.Logit(y_train, X_train).fit()
print(model_lr.summary())

#Prediction on train
y_prob_train = model_lr.predict(X_train)
y_prob_train

y_pred_train = [1 if _ > 0.5 else 0 for _ in y_prob_train] 
y_pred_train

#Accuracy
sum(y_pred_train == y_train) / len(y_train) * 100

#Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_train, y_pred_train) * 100

#Confusion matrix for train dataset
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_train, y_pred_train)
print(confusion_matrix)

#Prediction on test
X_test = test.loc[:, ['rsi', 
                      'Lag1', 'Lag2', 'Lag3', 'Lag4',
                      'Lag5']]

X_test = sm.add_constant(X_test) #adding a constant
X_test.head()

y_test = test.loc[:, 'trend']
y_test.head()

y_prob_test = model_lr.predict(X_test)
y_prob_test

y_pred_test = pd.Series([1 if _ > 0.5 else 0 for _ in y_prob_test], index = y_prob_test.index)
y_pred_test

#Accuracy
sum(y_pred_test == y_test) / len(y_test) * 100

#Confusion matrix for test dataset
from sklearn.metrics import confusion_matrix
confusion_matrix_lr = confusion_matrix(y_test, y_pred_test)
print(confusion_matrix_lr)

#Model evaluation
#Accuracy = TP + TN / Total
#TP = 76, TN = 68
(76 + 68) / len(y_test) * 100

#Precision = TP / TP + FP
#TP = 76, FP = 3
76 / (76 + 3) * 100

#Sensitivity 
#TP / TP + FN
#TP = 76, FN = 8
76 / (76 + 8) * 100

#Specificity
#TN / TN + FP
#TN = 59, FP = 60
68 / (68 + 3) * 100

### Random Forest 

#Define the feature set X 
X_train = train.loc[:, ['Today', 
                        'Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5',
                        'v_lag1', 'v_lag2', 'v_lag3', 'v_lag4', 'v_lag5']]

X_train.head()

y_train.head()

from sklearn.ensemble import RandomForestClassifier
#Random Forest: Using 4 Features
model_rf = RandomForestClassifier(max_features = 4, random_state = 123, n_estimators = 500).fit(X_train, y_train)

#Prediction on test
X_test = test.loc[:, ['Today', 
                      'Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5',
                      'v_lag1', 'v_lag2', 'v_lag3', 'v_lag4', 'v_lag5']]

X_test.head()

y_pred_rf = pd.Series(model_rf.predict(X_test), index = y_test.index)
y_pred_rf

#Accuracy
sum(y_pred_rf == y_test) / len(y_test) * 100

#Confusion matrix for test dataset
from sklearn.metrics import confusion_matrix
confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print(confusion_matrix_rf)

### Naive Bayes Classifier 

from sklearn.naive_bayes import GaussianNB

model_nb = GaussianNB().fit(X_train, y_train)

#Prediction on test
y_pred_nb = pd.Series(model_nb.predict(X_test), index = y_test.index)
y_pred_nb

#Accuracy
sum(y_pred_nb == y_test) / len(y_test) * 100

#Confusion matrix for test dataset
from sklearn.metrics import confusion_matrix
confusion_matrix_nb = confusion_matrix(y_test, y_pred_nb)
print(confusion_matrix_nb)

### Linear Discriminant Analysis (LDA) 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model_lda = LinearDiscriminantAnalysis().fit(X_train, y_train)

#Prediction on test
y_pred_lda = pd.Series(model_lda.predict(X_test), index = y_test.index)
y_pred_lda

#Accuracy
sum(y_pred_lda == y_test) / len(y_test) * 100

#Confusion matrix for test dataset
from sklearn.metrics import confusion_matrix
confusion_matrix_lda = confusion_matrix(y_test, y_pred_lda)
print(confusion_matrix_lda)

### Support Vector Machines 

from sklearn.svm import SVC
#model_svc = SVC(kernel = 'poly', degree = 2, C = 1E6).fit(X_train, y_train)
#Degree?
#C: Regularization parameter. The strength of the regularization is inversely proportional to C.

#Cross-Validation on Polynomial kernel SVM
from sklearn.model_selection import cross_val_score
degree_grid = [2, 3, 4]
acc_scores = []
for d in degree_grid:
    svc = SVC(kernel = 'poly', degree = d, C = 1E6)
    scores = cross_val_score(svc, X_train, y_train, cv = 5, scoring = 'accuracy')
    acc_scores.append(scores.mean())
print(acc_scores)

#Plot Cross-Validation Results for SVM
plt.plot(degree_grid, acc_scores)
plt.xticks(degree_grid)
plt.xlabel('Degree')
plt.ylabel('Cross-Validated Accuracy')

#Prediction on test
model_svc = SVC(kernel = 'poly', degree = 3, C = 1E6).fit(X_train, y_train)
y_pred_svc = model_svc.predict(X_test)
y_pred_svc

#Accuracy
sum(y_pred_svc == y_test) / len(y_test) * 100

#Confusion matrix for test dataset
from sklearn.metrics import confusion_matrix
confusion_matrix_svc = confusion_matrix(y_test, y_pred_svc)
print(confusion_matrix_svc)

### Summary of results on test data 

Logistic Regression: 53.78

Random Forest: 47.41

Naive Bayes Classifier: 49.00

Linear Discriminant Analysis: 55.78

Support Vector Machines: 49.40239

## Strategy Implementation

#Prediction on real
#Define the feature set X 
X_real = real.loc[:, ['rsi', 
                      'r_lag1', 'r_lag2', 'r_lag3', 'r_lag4', 'r_lag5',
                      'v_lag1', 'v_lag2', 'v_lag3', 'v_lag4', 'v_lag5']]
X_real

y_real = real.loc[:, 'trend']
y_real.head()

y_pred_real = pd.Series(model_lda.predict(X_real), index = y_real.index)
y_pred_real

#Accuracy
sum(y_pred_real == y_real) / len(y_real) * 100

#Confusion matrix for test dataset
from sklearn.metrics import confusion_matrix
confusion_matrix_real = confusion_matrix(y_real, y_pred_real)
print(confusion_matrix_real)

real['pred'] = y_pred_real
real

#Balance over time
real['balance'] = 0
real.head()

#initial deposit : $1000
real.iloc[0, 22] = 1000
real.head()

#Trade Simulation
for i in range(1, real.shape[0]):
    if real.iloc[i, 21] == 1:
        real.iloc[i, 22] = real.iloc[i - 1, 22] * real.iloc[i, 4] / real.iloc[i, 3]
    if real.iloc[i, 21] == 0:
        real.iloc[i, 22] = real.iloc[i - 1, 22] * real.iloc[i, 3] / real.iloc[i, 4]

real

real.loc[:, 'balance'].plot()
plt.axhline(1000, color = 'red', linewidth = 2, linestyle = '--')

Any Suggestion for Improving the algo trading system?

# End of the Code
