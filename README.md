# LS_Project

## Overview
This repository is for my personal project. In this repository multiple Data science tools will be used , Examples being :
+ SQL
+ Python
+ Machiene learning
+ Power bi
----

### SQL
For these SQL queries,The retail_sales_dataset which can be found in the description was used 

#### Queries
``` sql
SELECT * FROM retail_sales_dataset;
```
``` sql
---for top-selling products---
SELECT 
Product_Category,SUM(Quantity) AS total_quantity_sold
FROM retail_sales_dataset
GROUP BY Product_Category
ORDER BY total_quantity_sold DESC;
```
``` sql
--- total quantity by gender ---
SELECT gender ,SUM(quantity) AS total_quantity_sold
FROM retail_sales_dataset
GROUP BY gender
ORDER BY total_quantity_sold DESC;
```
``` sql
--- for count of gender in product category ---
SELECT Product_Category, gender, COUNT(*) AS gender_count
FROM retail_sales_dataset
GROUP BY Product_Category, gender
ORDER BY gender_count DESC;
```
``` sql
--- age range (18-30) ---
SELECT gender, age
FROM retail_sales_dataset
WHERE age BETWEEN 18 AND 30;
```
``` sql
--- age range (30-40) ---
SELECT gender, age
FROM retail_sales_dataset
WHERE age BETWEEN 30 AND 40;
```
``` sql
--- age range (40-50) ---
SELECT gender, age
FROM retail_sales_dataset
WHERE age BETWEEN 40 AND 50;
```
``` sql
--- age range (50-64) ---
SELECT gender, age
FROM retail_sales_dataset
WHERE age BETWEEN 50 AND 64;
```
----
### Python 
Under the python portion of this project i used pandas to clean and preprocess the data with pandas,While also perfroming EDA with Matplotlib/Seaborn
> These codes are carried out with the "retail_sales_dataset.csv" as found in the description

> Note : All codes are carried out on Google colab
----
#### Codes
``` python
***Importing libraries***
```
``` python
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
```
``` python
***Importing dataset***
```
``` python
df = pd.read_csv("retail_sales_dataset.csv")
```
``` python
***Data cleaning & preprocessing***
```
``` python
df.info()
```
``` python
df.describe()
```
``` python
df.head()
```
``` python
print(df.isnull().sum)
```
``` python
# converting data types
df["Date"] = pd.to_datetime(df["Date"])
```

``` python
***EDA***
```
``` python
print(df.describe(include="all"))
```
``` python
***Visualisation with seaborn/matplotlib***
```
``` python
# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```
``` python
# Distribution of a numeric column
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()
```
``` python
# Boxplot for numeric vs categorical
plt.figure(figsize=(8,5))
sns.boxplot(x="Gender", y="Total Amount", data=df)
plt.title("Income by Gender")
plt.show()
```
``` python
# Count plot of a categorical column
plt.figure(figsize=(8,5))
sns.countplot(x="Gender", data=df)
plt.title("Gender Distribution")
plt.show()
```
``` python
# Scatter plot for relationships
plt.figure(figsize=(8,5))
sns.scatterplot(x="Age", y="Total Amount", hue="Gender", data=df)
plt.title("Age vs Income by Gender")
plt.show()
```
### Machine learning
Under machiene learning wth the aid of python,I created a classification model with the aim of predicting customer churn 
> For the creation of this model i used the "Churn_Modelling.csv" as found in the description

> Note : All codes are carried out on Google colab
#### Codes
``` python
> ***Importing the libraries***
```
``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
``` python
> ***Importing the dataset***
```
``` python
line = pd.read_csv("Churn_Modelling.csv")
line.head()
```
``` python
print(line.isnull().sum())
```
``` python
line.info()
```
``` python
> ***Droping insignificant figures***
```
``` python
line.drop(columns=["CustomerId","Surname"],inplace=True)
```
``` python
> ***Divide into X and y***
```
``` python
X = line.iloc [:,1:-1].values
y = line.iloc [:,-1].values
```
``` python
print(X)
```
``` python
print(y)
```
``` python
> ***One hot encoding***
```
``` python
# for changing the independent variable to numeric representation

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[1,2])],remainder= "passthrough")
X = ct.fit_transform (X)
```
``` python
print(X)
```
``` python
> ***Training and testing the data***
```
``` python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42)
```
``` python
print(X_train)
```
``` python
print(X_test)
```
``` python
print(y_train)
```
``` python
print(y_test)
```
``` python
> ***Feature Scaling***
```
``` python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,3:-1] = sc.fit_transform(X_train)[:,3:-1]
X_test[:,3:-1] = sc.transform(X_test)[:,3:-1]
```
``` python
print(X_train[:,3:-1] )
```
``` python
print(X_test[:,3:-1] )
```
``` python
> ***Building the model***
```
``` python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10,random_state=42)
model.fit(X_train,y_train)
```
``` python
> ***Predicting a new result***
```
``` python
y_pred = model.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
```
``` python
> ***Evaluation***
```
``` python
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)
```
### Power bi
This is the last portion of this repository, With the aid of PowerBi i created a dashboard showing the 
+ Profit by country
+ profit by product category
+ profit by Time (e.g Month,year,Quarters)

> Note: The visuals were created with the "accessories sales" csv file as found in the description

> Note: The visuals and dashboard were uploaded in the description as "LS_academy.pdf" 





