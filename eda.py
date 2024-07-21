# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#loading the dataset
data_set= pd.read_csv('Attrition Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Summary statistics for numerical features
print(data_set.describe())

# Summary statistics for categorical features
print(data_set.describe(include='object'))

# Check for missing values
print(data_set.isnull().sum())

#--------------------------------DATA VISUALISATION--------------------------------------------

# Histograms for numerical features
data_set.hist(bins=30, figsize=(20, 15))
plt.show()

# Box plots for numerical features
plt.figure(figsize=(15, 10))
sns.boxplot(data=data_set)
plt.xticks(rotation=90)
plt.show()

# Bar charts for categorical features
for column in data_set.select_dtypes(include=['object']).columns:
    data_set[column].value_counts().plot(kind='bar')
    plt.title(column)
    plt.show()

# Heatmap for correlation analysis
plt.figure(figsize=(12, 8))
sns.heatmap(data_set.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.show()

# Compare features between employees who stayed and those who left
attrition = data_set[data_set['Attrition'] == 'Yes']
no_attrition = data_set[data_set['Attrition'] == 'No']