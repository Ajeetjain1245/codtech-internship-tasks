# Task 1: Data Pipeline for Data Preprocessing, Transformation & Loading
## 📌 Objective:
Create a Python script or notebook that performs:

Data extraction and loading

Data preprocessing and cleaning

Data transformation using tools like Pandas and Scikit-learn

## 🛠 Technologies Used:
Python

Pandas

Scikit-learn

## 🧪 Sample Output:
'''Original Data Snapshot:
Age, Salary, Department
25, 50000, Sales
30, 60000, Marketing
NaN, 52000, Sales
35, NaN, IT

Processed Data Snapshot:
Age, Salary, Department, Age_scaled, Salary_scaled, Department_encoded
25, 50000, Sales, 0.0, 0.2, 1
30, 60000, Marketing, 0.5, 1.0, 0
28, 52000, Sales, 0.4, 0.4, 1
35, 56000, IT, 1.0, 0.6, 2
'''
