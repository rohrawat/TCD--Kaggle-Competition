# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:38:00 2019

@author: Rohit
"""
#Importing all the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
#Reading train and test csv files
salary = pd.read_csv(r'D:\download\tcd-ml-1920-group-income-train.csv\tcd-ml-1920-group-income-train.csv')
salarytest = pd.read_csv(r'D:\download\tcd-ml-1920-group-income-test.csv\tcd-ml-1920-group-income-test.csv')
#Delete unnecessary columns from train and test files
salary.drop('Instance',axis = 1,inplace = True)
salarytest.drop(['Instance','Total Yearly Income [EUR]'],axis = 1,inplace = True)
#Exploratory Data Analysis and removing null values
salary.describe()
salary.isnull().sum()
sns.heatmap(salary.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')
salary['Gender'] = salary['Gender'].fillna('unknown')
salary['Gender'] = salary['Gender'].replace('0','unknown')
salary['Gender'] = salary['Gender'].replace('other','Other')
salary['Gender'] = salary['Gender'].replace('f','female')
salarytest['Gender'] = salarytest['Gender'].fillna('unknown')
salarytest['Gender'] = salarytest['Gender'].replace('0','unknown')
salarytest['Gender'] = salarytest['Gender'].replace('other','Other')
salarytest['Gender'] = salarytest['Gender'].replace('f','female')
#salarytest['Gender'] = salarytest['Gender'].replace('0',np.NaN)
salary['University Degree'] = salary['University Degree'].fillna('No')
salary['University Degree'] = salary['University Degree'].replace('0','No')
salarytest['University Degree'] = salarytest['University Degree'].fillna('No')
salarytest['University Degree'] = salarytest['University Degree'].replace('0','No')
#salarytest['University Degree'] = salarytest['University Degree'].replace('0',np.NaN)
salary['Hair Color'] = salary['Hair Color'].fillna('other')
salary['Hair Color'] = salary['Hair Color'].replace('0','other')
salary['Hair Color'] = salary['Hair Color'].replace('Unknown','other')
salarytest['Hair Color'] = salarytest['Hair Color'].fillna('other')
salarytest['Hair Color'] = salarytest['Hair Color'].replace('0','other')
salarytest['Hair Color'] = salarytest['Hair Color'].replace('Unknown','other')
#salarytest['Hair Color'] = salarytest['Hair Color'].replace('0',np.NaN)
#salary['Year of Record'] = salary['Year of Record'].replace(np.nan,0)
#salary['Age'] = salary['Age'].replace(np.nan,0)
salary['Year of Record'] = salary['Year of Record'].fillna(salary['Year of Record'].mode()[0])
salary['Age'] = salary['Age'].fillna(salary['Age'].mean())
salary['Profession'] = salary['Profession'].fillna('other')
salarytest['Year of Record'] = salarytest['Year of Record'].fillna(salarytest['Year of Record'].mode()[0])
#salarytest['Age'] = salarytest['Age'].replace(np.nan,0)
#salarytest['Year of Record'] = salarytest['Year of Record'].replace(0,np.median(salary['Year of Record']))
salarytest['Age'] = salarytest['Age'].fillna(salarytest['Age'].mean())
salarytest['Profession'] = salarytest['Profession'].fillna('other')
salary['Housing Situation'] = salary['Housing Situation'].replace(0, 'Homeless')
salary['Housing Situation'] = salary['Housing Situation'].replace('nA','Homeless')
salarytest['Housing Situation'] = salarytest['Housing Situation'].replace(0, 'Homeless')
salarytest['Housing Situation'] = salarytest['Housing Situation'].replace('nA','Homeless')
salary['Work Experience in Current Job [years]'] = salary['Work Experience in Current Job [years]'].replace('#NUM!',np.NaN) 
salary['Work Experience in Current Job [years]'] = salary['Work Experience in Current Job [years]'].fillna(salary['Work Experience in Current Job [years]'].median())
salarytest['Work Experience in Current Job [years]'] = salarytest['Work Experience in Current Job [years]'].replace('#NUM!',np.NaN) 
salarytest['Work Experience in Current Job [years]'] = salarytest['Work Experience in Current Job [years]'].fillna(salarytest['Work Experience in Current Job [years]'].median())
salary['Satisfation with employer'] = salary['Satisfation with employer'].fillna('Average')
salarytest['Satisfation with employer'] = salarytest['Satisfation with employer'].fillna('Average')
salary['Yearly Income in addition to Salary (e.g. Rental Income)'] = salary['Yearly Income in addition to Salary (e.g. Rental Income)'].map(lambda x:x.rstrip('EUR'))
salarytest['Yearly Income in addition to Salary (e.g. Rental Income)'] = salarytest['Yearly Income in addition to Salary (e.g. Rental Income)'].map(lambda x:x.rstrip('EUR'))
salary['Work Experience in Current Job [years]'] = salary['Work Experience in Current Job [years]'].astype(float)
salarytest['Work Experience in Current Job [years]'] = salarytest['Work Experience in Current Job [years]'].astype(float)
salary['Yearly Income in addition to Salary (e.g. Rental Income)'] = salary['Yearly Income in addition to Salary (e.g. Rental Income)'].astype(float)
salarytest['Yearly Income in addition to Salary (e.g. Rental Income)'] = salarytest['Yearly Income in addition to Salary (e.g. Rental Income)'].astype(float)
#Target Encoding for categorical columns
country = salary['Country'].value_counts()
countrytest = salarytest['Country'].value_counts()
profession = salary['Profession'].value_counts()
professiontest = salarytest['Profession'].value_counts()
satisfaction = salary['Satisfation with employer'].value_counts()
satisfactiontest = salarytest['Satisfation with employer'].value_counts()
housing = salary['Housing Situation'].value_counts()
housingtest = salarytest['Housing Situation'].value_counts()
deg = salary['University Degree'].value_counts()
degtest = salarytest['University Degree'].value_counts()
gender = salary['Gender'].value_counts()
gendertest = salarytest['Gender'].value_counts()
salary['newcountry'] = 0
salary['newprofession'] = 0
salarytest['newcountry'] = 0
salarytest['newprofession'] = 0
salary['newSatisfation with employer'] = 0
salarytest['newSatisfation with employer'] = 0
salary['newHousing Situation'] = 0
salarytest['newHousing Situation'] = 0
salary['newUniversity Degree'] = 0
salarytest['newUniversity Degree'] = 0
salary['newGender'] = 0
salarytest['newGender'] = 0
for i in country.index:
    temp = salary[salary['Country']==i]
    #a = sum(temp['Income in EUR'])/temp.shape[0]
    salary.loc[salary['Country']==i, 'newcountry'] = ((np.mean(temp['Total Yearly Income [EUR]']) * len(temp)) +(48 * np.mean(salary['Total Yearly Income [EUR]'])))/(len(temp)+48)
    salarytest.loc[salarytest['Country']==i,'newcountry'] = ((np.mean(temp['Total Yearly Income [EUR]']) * len(temp)) +(48 * np.mean(salary['Total Yearly Income [EUR]'])))/(len(temp)+48)

for i in satisfaction.index:
    temp = salary[salary['Satisfation with employer']==i]
    #a = sum(temp['Income in EUR'])/temp.shape[0]
    salary.loc[salary['Satisfation with employer']==i, 'newSatisfation with employer'] = ((np.mean(temp['Total Yearly Income [EUR]']) * len(temp)) +(48 * np.mean(salary['Total Yearly Income [EUR]'])))/(len(temp)+48)
    salarytest.loc[salarytest['Satisfation with employer']==i,'newSatisfation with employer'] = ((np.mean(temp['Total Yearly Income [EUR]']) * len(temp)) +(48 * np.mean(salary['Total Yearly Income [EUR]'])))/(len(temp)+48)

for i in housing.index:
    temp = salary[salary['Housing Situation']==i]
    #a = sum(temp['Income in EUR'])/temp.shape[0]
    salary.loc[salary['Housing Situation']==i, 'newHousing Situation'] = ((np.mean(temp['Total Yearly Income [EUR]']) * len(temp)) +(48 * np.mean(salary['Total Yearly Income [EUR]'])))/(len(temp)+48)
    salarytest.loc[salarytest['Housing Situation']==i,'newHousing Situation'] = ((np.mean(temp['Total Yearly Income [EUR]']) * len(temp)) +(48 * np.mean(salary['Total Yearly Income [EUR]'])))/(len(temp)+48)

for i in deg.index:
    temp = salary[salary['University Degree']==i]
    #a = sum(temp['Income in EUR'])/temp.shape[0]
    salary.loc[salary['University Degree']==i, 'newUniversity Degree'] = ((np.mean(temp['Total Yearly Income [EUR]']) * len(temp)) +(48 * np.mean(salary['Total Yearly Income [EUR]'])))/(len(temp)+48)
    salarytest.loc[salarytest['University Degree']==i,'newUniversity Degree'] = ((np.mean(temp['Total Yearly Income [EUR]']) * len(temp)) +(48 * np.mean(salary['Total Yearly Income [EUR]'])))/(len(temp)+48)

for i in gender.index:
    temp = salary[salary['Gender']==i]
    #a = sum(temp['Income in EUR'])/temp.shape[0]
    salary.loc[salary['Gender']==i, 'newGender'] = ((np.mean(temp['Total Yearly Income [EUR]']) * len(temp)) +(48 * np.mean(salary['Total Yearly Income [EUR]'])))/(len(temp)+48)
    salarytest.loc[salarytest['Gender']==i,'newGender'] = ((np.mean(temp['Total Yearly Income [EUR]']) * len(temp)) +(48 * np.mean(salary['Total Yearly Income [EUR]'])))/(len(temp)+48)

for i in profession.index:
    temp = salary[salary['Profession']==i]
    #a = sum(temp['Income in EUR'])/temp.shape[0]
    salary.loc[salary['Profession']==i,'newprofession'] = ((np.mean(temp['Total Yearly Income [EUR]']) * len(temp)) + (48 * np.mean(salary['Total Yearly Income [EUR]'])))/(len(temp)+48)
    salarytest.loc[salarytest['Profession']==i,'newprofession'] = ((np.mean(temp['Total Yearly Income [EUR]']) * len(temp)) + (48 * np.mean(salary['Total Yearly Income [EUR]'])))/(len(temp)+48)

salary.drop(['Country','Profession','University Degree','Gender','Housing Situation','Satisfation with employer', 'Hair Color'],axis = 1, inplace = True)
salarytest.drop(['Country','Profession','University Degree','Gender','Housing Situation','Satisfation with employer','Hair Color'],axis = 1, inplace = True)
salarytest['newcountry'] = salarytest['newcountry'].replace(0,np.mean(salarytest['newcountry']))
salarytest['newprofession'] = salarytest['newprofession'].replace(0,np.mean(salarytest['newprofession']))
salarytest['newUniversity Degree'] = salarytest['newUniversity Degree'].replace(0,np.mean(salarytest['newUniversity Degree']))
salarytest['newGender'] = salarytest['newGender'].replace(0,np.mean(salarytest['newGender']))
salarytest['newHousing Situation'] = salarytest['newHousing Situation'].replace(0,np.mean(salarytest['newHousing Situation']))
salarytest['newSatisfation with employer'] = salarytest['newSatisfation with employer'].replace(0,np.mean(salarytest['newSatisfation with employer']))
#Function for min max scaling
def normalize(x):
    return (x - np.min(x))/(np.max(x)-np.min(x))
#separate features and label from dataset and take log of Income column
salary['Income'] = np.log(salary['Total Yearly Income [EUR]'])
Y = salary['Income']
X = salary.drop(['Total Yearly Income [EUR]','Wears Glasses','Income'],axis = 1)
Xtest = salarytest.drop(['Wears Glasses'], axis  = 1)
#apply min-max function to all the features
X = X.apply(normalize)
Xtest = Xtest.apply(normalize)
#split data in train and test in the ratio of 80:20
X_train, X_test,Y_train, Y_test = train_test_split(X,Y,train_size = 0.8)
#train the model using Catboost with two different learning rate 0.01 and 0.009 and take mean of the results.
cbr = CatBoostRegressor(iterations=12000,
                        learning_rate = 0.009,
                        depth = 10,
                         eval_metric = 'MAE',
                         bagging_temperature = 0.2,
                         od_wait = 100)
cbr.fit(X_train,Y_train,
        eval_set = (X_test,Y_test),
        use_best_model = True,
        verbose = True)
cbr1 = CatBoostRegressor(iterations=12000,
                        learning_rate = 0.01,
                        depth = 10,
                         eval_metric = 'MAE',
                         bagging_temperature = 0.2,
                         od_wait = 100)
cbr1.fit(X_train,Y_train,
        eval_set = (X_test,Y_test),
        use_best_model = True,
        verbose = True)
#predict income of test data
ypred  = cbr.predict(Xtest)
ypred1 = cbr1.predict(Xtest)
ypred = pd.DataFrame(ypred)
ypred1 = pd.DataFrame(ypred1)
#take exponential of income predicted
ypred = np.exp(ypred)
ypred1 = np.exp(ypred1)
#calculate mean of income from two model
result = pd.concat([ypred,ypred1],axis = 1)
result = result.mean(axis = 1)
#save result to csv files
result.to_csv(r'D:\download\result17.csv')
