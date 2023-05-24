# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries .
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4. Apply label encoder to the non-numerical column inoreder to convert into numerical values.
5. Determine training and test data set.
6. Apply decision tree regression on to the dataframe.
7. Get the values of Mean square error, r2 and data prediction.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Pravinrajj G.K
RegisterNumber:  212222240080
*/

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
y = data["Salary"]
y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
### data.head()
![240526707-28c6a7c9-e262-4ebc-ae4a-bdb804234370](https://github.com/Pravinrajj/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/117917674/3b090317-7639-4a3a-aef9-f89a18e73fa4)
### data.info()
![240549459-a8e42ffb-8c98-4499-8d78-72cf7232af96](https://github.com/Pravinrajj/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/117917674/63f6ca58-2307-42ac-b695-796aae4d9e4b)
### isnull() and sum()
![240526928-63bcba94-bf2c-4a8d-98db-b3ad0e485a10](https://github.com/Pravinrajj/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/117917674/cb64eb02-102a-4895-89d6-2b0afc6f5197)
### data.head() for salary
![240527027-e1916232-103c-4e61-9525-28f8023dc52c](https://github.com/Pravinrajj/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/117917674/3b0dae9a-6fef-4124-ac42-db6c1a33c4fa)
### MSR value
![240527095-de3d3ea5-f16b-401a-acec-c5b541a09c60](https://github.com/Pravinrajj/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/117917674/364ae628-d50d-4984-b7c7-3a89edcc355f)
### r2 value
![240527166-0a263b34-f482-4370-a052-49428cd1d50b](https://github.com/Pravinrajj/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/117917674/a1ad7b20-03c3-44f6-8234-fe52294a0534)
### Data Prediction
![240527586-39891e38-299c-4b27-b625-33cb0334a6c9](https://github.com/Pravinrajj/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/117917674/7d9b4df8-6e94-40be-af97-43c59c9db7b7)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
