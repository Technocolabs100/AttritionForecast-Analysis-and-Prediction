
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"E:\project\AttritionForecast-Analysis-and-Prediction\Attrition Dataset\EmployeeData.csv"
df = pd.read_csv(file_path)

# Display the first few rows and column information
print("First few rows of the dataset:")
print(df.head())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Information about the dataset
print("\nDataset information:")
print(df.info())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check unique values for categorical columns
print("\nUnique values for categorical columns:")
for col in df.select_dtypes(include=['object']):
    print(f"{col}: {df[col].unique()}")

# Check value counts for target variable 'Attrition'
print("\nValue counts for 'Attrition':")
print(df['Attrition'].value_counts())

# Visualizations

# Histogram for numeric features
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols].hist(bins=15, figsize=(15, 10))
plt.suptitle('Histograms of Numeric Features')
plt.show()

# Count plots for categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=col)
    plt.title(f'Count Plot for {col}')
    plt.xticks(rotation=45)
    plt.show()

# Correlation heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
# Drop columns with excessive missing values
threshold = 0.3 * len(df)
df = df.dropna(thresh=threshold, axis=1)

# Drop unnecessary columns
drop_columns = ['EmployeeNumber', 'EmployeeCount', 'StandardHours']
df = df.drop(columns=drop_columns)

# Handle missing values in specific columns if necessary
# Example: Fill missing values with median for numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Check again for missing values
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Encode categorical variables using label encoding or one-hot encoding
from sklearn.preprocessing import LabelEncoder

# Example: Encode 'Attrition' column (target variable)
label_encoder = LabelEncoder()
df['Attrition'] = label_encoder.fit_transform(df['Attrition'])

# Example: One-hot encode other categorical columns
df = pd.get_dummies(df, columns=['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime'], drop_first=True)

# Display the updated dataframe
print("\nEncoded dataset:")
print(df.head())

# Example: Binning numeric data into categories
# This step is optional and depends on your specific analysis needs

# Example: Bin 'Age' into categories
bins = [18, 25, 35, 45, 55, 65]
labels = ['18-25', '26-35', '36-45', '46-55', '56-65']
df['AgeCategory'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Display the updated dataframe with labels
print("\nUpdated dataset with labels:")
print(df.head())

