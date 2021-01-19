#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 23:16:53 2020

@author: anujhsawant
"""

#Importing all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import plot
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

#Importing the dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head()
df.columns

#Checking for any null values
df.isnull().values.any()

#Description of the dataset
df.describe()

#Count plot of the target varaible Attrition
sns.countplot(df['Attrition'])

#Age vs. Attrition
plt.subplots(figsize=(12,4))
sns.countplot(x='Age',hue='Attrition',data=df)

#Number of Companies worked vs. Attrition
sns.countplot(x="NumCompaniesWorked", hue="Attrition",data = df)

#Dropping unnecessary columns
df=df.drop(['Over18','EmployeeNumber','StandardHours','EmployeeCount'],axis=1)

#Correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=False)

#Salary hike vs. Attrition
hike=df.groupby(['PercentSalaryHike','Attrition']).apply(lambda x:x['DailyRate'].count()).reset_index(name='Counts')
hike_fig = px.line(hike,x='PercentSalaryHike',y='Counts',color='Attrition',title='Count of Hike Percentages people receive in an Organization vs. Attrition')
plot(hike_fig)

#Stock option vs. Attrition
stockOption=df.groupby(['StockOptionLevel','Attrition']).apply(lambda x:x['DailyRate'].count()).reset_index(name='Counts')
stocks_fig = px.bar(stockOption,x='StockOptionLevel',y='Counts',color='Attrition',title='Stock options given to people in an Organization vs. Attrition')
plot(stocks_fig)

#Years vs Attrition
yearsAtJob=df.groupby(['YearsInCurrentRole','Attrition']).apply(lambda x:x['DailyRate'].count()).reset_index(name='Counts')
yearsAtJob_fig = px.line(yearsAtJob,x='YearsInCurrentRole',y='Counts',color='Attrition',title='Count of People working for # of years in an Organization vs. Attrition')
plot(yearsAtJob_fig)

#Work Exp vs attrition
workExp=df.groupby(['NumCompaniesWorked','Attrition']).apply(lambda x:x['DailyRate'].count()).reset_index(name='Counts')
workExp_fig = px.area(workExp,x='NumCompaniesWorked',y='Counts',color='Attrition',title='Work Experience level of people in an Organization vs. Attrition')
plot(workExp_fig)

#Monthly income vs. Attrition
monthlyIncomeRate=df.groupby(['MonthlyIncome','Attrition']).apply(lambda x:x['MonthlyIncome'].count()).reset_index(name='Counts')
monthlyIncomeRate['MonthlyIncome']=round(monthlyIncomeRate['MonthlyIncome'],-3)
monthlyIncomeRate=monthlyIncomeRate.groupby(['MonthlyIncome','Attrition']).apply(lambda x:x['MonthlyIncome'].count()).reset_index(name='Counts')
monthly_fig=px.line(monthlyIncomeRate,x='MonthlyIncome',y='Counts',color='Attrition',title='Monthly Income of people in an Organization vs. Attrition')
plot(monthly_fig)

#Age vs. Attrition
age_group=df.groupby(['Age','Attrition']).apply(lambda x:x['DailyRate'].count()).reset_index(name='Counts')
age_group_fig = px.line(age_group,x='Age',y='Counts',color='Attrition',title='Age vs Attrition')
plot(age_group_fig)

#Department wise attrition
dept=df.groupby(['Department','Attrition']).apply(lambda x:x['DailyRate'].count()).reset_index(name='Counts')
dept_fig=px.bar(dept,x='Department',y='Counts',color='Attrition',title='Department wise Attrition')
plot(dept_fig)

#MODELLING:

y = df['Attrition']
X = df
X.pop('Attrition')

y.unique()

df.head()

#Using label binarizer
le = preprocessing.LabelBinarizer()
y = le.fit_transform(y)

y.shape

df.info()

#Sorting features with dtype object
df.select_dtypes(['object'])

#Converting object (categorical) dtypes to numerical
ind_BusinessTravel = pd.get_dummies(df['BusinessTravel'], prefix='BusinessTravel')
ind_Department = pd.get_dummies(df['Department'], prefix='Department')
ind_EducationField = pd.get_dummies(df['EducationField'], prefix='EducationField')
ind_Gender = pd.get_dummies(df['Gender'], prefix='Gender')
ind_JobRole = pd.get_dummies(df['JobRole'], prefix='JobRole')
ind_MaritalStatus = pd.get_dummies(df['MaritalStatus'], prefix='MaritalStatus')
ind_OverTime = pd.get_dummies(df['OverTime'], prefix='OverTime')

ind_BusinessTravel.head()

df['BusinessTravel'].unique()

#Concatinating the converted features into a new df
df1 = pd.concat([ind_BusinessTravel, ind_Department, ind_EducationField, ind_Gender, 
                 ind_JobRole, ind_MaritalStatus, ind_OverTime])

df.select_dtypes(['int64'])

#Concatinating both the new df and the remaining features
df1 = pd.concat([ind_BusinessTravel, ind_Department, ind_EducationField, ind_Gender, 
                 ind_JobRole, ind_MaritalStatus, ind_OverTime, df.select_dtypes(['int64'])], axis=1)



#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df1, y)

# Decision Tree
clf = DecisionTreeClassifier(random_state=42, max_depth=3)
clf.fit(X_train, y_train)

#Function for accuracy, classification report and confusion matrix
def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    '''
    print the accuracy score, classification report and confusion matrix of classifier
    '''
    if train:
        '''
        training performance
        '''
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))
        plt.figure(figsize=(4,4))
        sns.heatmap(confusion_matrix(y_train, clf.predict(X_train)),annot=True)

        res = cross_val_score(clf, X_train, y_train.ravel(), cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        '''
        test performance
        '''
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))
        plt.figure(figsize=(4,4))
        sns.heatmap(confusion_matrix(y_test, clf.predict(X_test)),annot=True)
        

print_score(clf, X_train, y_train, X_test, y_test, train=True)
print_score(clf, X_train, y_train, X_test, y_test, train=False)


# Random Forest
rf_clf = RandomForestClassifier(max_depth=3)
rf_clf.fit(X_train, y_train.ravel())

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)

#RFC Feature Importance plot
pd.Series(rf_clf.feature_importances_, 
         index=X_train.columns).sort_values(ascending=False).plot(kind='bar', figsize=(12,6));

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train.ravel())

predicted= log_reg.predict(X_test)

print_score(log_reg, X_train, y_train, X_test, y_test, train=True)
print_score(log_reg, X_train, y_train, X_test, y_test, train=False)

# XGBoost
xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train.ravel())

print_score(xgb_clf, X_train, y_train, X_test, y_test, train=True)
print_score(xgb_clf, X_train, y_train, X_test, y_test, train=False)

#XGBoost Feature Importance plot
pd.Series(xgb_clf.feature_importances_, 
         index=X_train.columns).sort_values(ascending=False).plot(kind='bar', figsize=(12,6));