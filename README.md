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
Under the python portion of this project i will be using pandas to clean and preprocess the data with pandas,While also perfroming EDA with Matplotlib/Seaborn
> These codes are carried out with the "retail_sales_dataset.csv" as found in the description 
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
###Machine learning

