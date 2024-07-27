# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib

# Loading the dataset
data_set = pd.read_csv('Attrition Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Summary statistics for numerical features
print(data_set.describe())

# Summary statistics for categorical features
print(data_set.describe(include='object'))

# Check for missing values
print(data_set.isnull().sum())
'''
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

#--------------------------------ATTRITION ANALYSIS--------------------------------------------

# Compare features between employees who stayed and those who left
attrition = data_set[data_set['Attrition'] == 'Yes']
no_attrition = data_set[data_set['Attrition'] == 'No']

# List of numerical features to compare
numerical_features = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'TotalWorkingYears']

for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(attrition[feature], label='Attrition', shade=True)
    sns.kdeplot(no_attrition[feature], label='No Attrition', shade=True)
    plt.title(f'{feature} Distribution by Attrition Status')
    plt.legend()
    plt.show()

# List of categorical features to compare
categorical_features = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']

for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data_set, x=feature, hue='Attrition')
    plt.title(f'{feature} Count by Attrition Status')
    plt.xticks(rotation=45)
    plt.show()
'''
#-----------------------------------DATA PREPROCESSING--------------------------------------
# Remove duplicates
data_set = data_set.drop_duplicates()

# Handling missing values
for column in data_set.columns:
    if data_set[column].dtype in [np.float64, np.int64]:  # Check if the column is numeric
        data_set[column].fillna(data_set[column].mean(), inplace=True)

# Label Encoding for binary categorical variables
label_encoder = LabelEncoder()
data_set['Attrition'] = label_encoder.fit_transform(data_set['Attrition'])

# One-Hot Encoding for multi-class categorical variables
data_set = pd.get_dummies(data_set, columns=['Department', 'EducationField', 'JobRole', 'MaritalStatus', 'OverTime', 'BusinessTravel'])

# Feature Scaling
scaler = StandardScaler()
numerical_features = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
                      'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
                      'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                      'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 
                      'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
                      'YearsSinceLastPromotion', 'YearsWithCurrManager']

data_set[numerical_features] = scaler.fit_transform(data_set[numerical_features])

# Feature Engineering
data_set['Age_Income'] = data_set['Age'] * data_set['MonthlyIncome']
data_set['Experience_Level'] = data_set['TotalWorkingYears'] / (data_set['Age'] - 18)

# Save the preprocessed data
data_set.to_csv('preprocessed_data.csv', index=False)

#--------------------------------DATA SPLITTING----------------------------


# Load the preprocessed data
data_path = 'preprocessed_data.csv'
data = pd.read_csv(data_path)

# Display the first few rows of the dataset
print(data.head())

# Separate features and target variable
X = data.drop(columns=['Attrition'])
y = data['Attrition']

# Identify remaining categorical features
categorical_features = X.select_dtypes(include=['object', 'bool']).columns

# Apply one-hot encoding
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Perform train-test split again with the updated data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save the train and test sets as CSV files
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# Display the shape of the resulting datasets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#----------------------------------------------MODEL BUILDING and training and evaluation and hyperparameter tuning-------------------------
# Function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
    return accuracy, precision, recall, f1, roc_auc, conf_matrix

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Neural Network': MLPClassifier(max_iter=5000)
}

# Define hyperparameter grids for each model
param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'solver': ['liblinear']
    },
    'Decision Tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'Neural Network': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001]
    }
}

# Perform initial model training and evaluation
trained_models = {}
evaluation_results = {}
confusion_matrices = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    accuracy, precision, recall, f1, roc_auc, conf_matrix = evaluate_model(model, X_test, y_test)
    evaluation_results[name] = [accuracy, precision, recall, f1, roc_auc]
    confusion_matrices[name] = conf_matrix

# Save evaluation results to a CSV file
results = []
for name, metrics in evaluation_results.items():
    accuracy, precision, recall, f1, roc_auc = metrics
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })
results_df = pd.DataFrame(results)
results_df.to_csv('initial_model_evaluation_results.csv', index=False)

# Save correct and incorrect predictions
for model_name, model in trained_models.items():
    test_data[f'{model_name}_Pred'] = model.predict(X_test)
    test_data[f'{model_name}_Correct'] = test_data['Attrition'] == test_data[f'{model_name}_Pred']
test_data.to_csv('initial_test_data_with_predictions.csv', index=False)

# Perform hyperparameter tuning and evaluation for each model
best_models = {}
evaluation_results = {}
confusion_matrices = {}

for name, model in models.items():
    print(f"Tuning hyperparameters for {name}...")
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_models[name] = best_model
    accuracy, precision, recall, f1, roc_auc, conf_matrix = evaluate_model(best_model, X_test, y_test)
    evaluation_results[name] = [accuracy, precision, recall, f1, roc_auc]
    confusion_matrices[name] = conf_matrix

# Display evaluation results for tuned models
print("Evaluation Results for Tuned Models:")
for name, metrics in evaluation_results.items():
    accuracy, precision, recall, f1, roc_auc = metrics
    print(f"{name}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}, ROC-AUC={roc_auc:.4f}")

# Convert evaluation results to a DataFrame for better comparison
results_df = pd.DataFrame(evaluation_results, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']).T

# Calculate the average score across all metrics
results_df['Average Score'] = results_df.mean(axis=1)

# Print the DataFrame with average scores
print("\nComparison of Model Performance:")
print(results_df)

# Declare the best model based on the highest average score
best_model_name = results_df['Average Score'].idxmax()
best_model = best_models[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"Performance: \n{results_df.loc[best_model_name]}")

# Save the confusion matrices
conf_matrix_path = 'confusion_matrices.joblib'
joblib.dump(confusion_matrices, conf_matrix_path)

# Save the best model
model_path = 'best_model.joblib'
joblib.dump(best_model, model_path)

# Load the model for future use
loaded_model = joblib.load(model_path)

# Make predictions
predictions = loaded_model.predict(X_test)
print(predictions)

# Load and print the confusion matrices
loaded_confusion_matrices = joblib.load(conf_matrix_path)
for model_name, conf_matrix in loaded_confusion_matrices.items():
    print(f"\nConfusion Matrix for {model_name}:\n{conf_matrix}")