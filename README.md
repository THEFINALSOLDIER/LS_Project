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

