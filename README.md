# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Employee.csv dataset and display the first few rows.

2.Check dataset structure and find any missing values.

3.Display the count of employees who left vs stayed.

4.Encode the "salary" column using LabelEncoder to convert it into numeric values.

5.Define features x with selected columns and target y as the "left" column.

6.Split the data into training and testing sets (80% train, 20% test).

7.Create and train a DecisionTreeClassifier model using the training data.

8.Predict the target values using the test data.

9.Evaluate the model’s accuracy using accuracy score.

10.Predict whether a new employee with specific features will leave or not.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Rajamanikandan R
RegisterNumber:  212223220082
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
data

data.head()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]
y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier (criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### Value of df:
![image](https://github.com/user-attachments/assets/c788ffa9-7136-4627-be6c-f62cbff1651c)


### df.head()
![image](https://github.com/user-attachments/assets/c6d96ebf-0063-4912-aa1b-83c1945d77cc)


### df.describe()
![image](https://github.com/user-attachments/assets/16a460ea-a04b-4631-b8ad-fcf0eba6bf5e)


### df.info()
![image](https://github.com/user-attachments/assets/ca63162e-4105-40c2-ba2c-21680811ac01)

### df.isnull().sum()
![image](https://github.com/user-attachments/assets/e22e1288-8032-4f53-a3e2-0e35b8d27266)  ![image](https://github.com/user-attachments/assets/66edd8ee-7dad-480d-9e7c-ac796ce9d9e3)![image](https://github.com/user-attachments/assets/be462836-5adc-4d4f-8d33-56a0be553dc7)




### Value of x.head() and y

![image](https://github.com/user-attachments/assets/aa92d499-bc47-4297-9f4d-8707a5fd8df0)    ![image](https://github.com/user-attachments/assets/05c6d866-9932-4c26-8d3b-7ef2e8938d9c)

![image](https://github.com/user-attachments/assets/29d36c4d-bdc9-4117-b165-4c86c5968b3d)




### Value of Accuracy,Confusion_matrix and data prediction
![image](https://github.com/user-attachments/assets/ecb48f84-e6c9-4435-a450-3ebaf686df22)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
