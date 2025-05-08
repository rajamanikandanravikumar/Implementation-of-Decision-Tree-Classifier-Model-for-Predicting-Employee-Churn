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
![image](https://github.com/user-attachments/assets/15e83408-2bf9-44fc-9e6f-eb58669783f8)

### df.head() and df.isnull().sum()
![image](https://github.com/user-attachments/assets/a65b7b7e-9b64-4810-8837-375bb51d5a2a)

### df.describe()
![image](https://github.com/user-attachments/assets/707c5a6b-3fb0-4072-a159-0061f54eac51)

### df.info() and Value counts
![image](https://github.com/user-attachments/assets/9afc341d-4ed8-4674-a40c-6ce29833c058)

### df.head()
![image](https://github.com/user-attachments/assets/b75300a1-1f49-48d4-81fc-bb88435723a0)

### Value of x.head() and y

![image](https://github.com/user-attachments/assets/4f908e6a-a5af-4959-a30a-0cfa2136056a)

### Value of Accuracy,Confusion_matrix and data prediction

![image](https://github.com/user-attachments/assets/d5c914fb-5283-4ed8-9ae1-64984260b5a3)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
