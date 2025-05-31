import pandas as pd
import numpy as np

# Custom Titanic-like dataset
data = {
    'PassengerId': [11, 12, 13, 14, 15],
    'Survived': [1, 0, 0, 1, 1],
    'Pclass': [1, 3, 2, 1, 3],
    'Name': [
        "Smith, Mr. John",
        "Johnson, Mrs. Emily",
        "Williams, Miss. Olivia",
        "Brown, Mr. James",
        "Davis, Miss. Emma"
    ],
    'Sex': ['male', 'female', 'female', 'male', 'female'],
    'Age': [45, np.nan, 29, 34, 15],
    'SibSp': [0, 1, 2, 1, 0],
    'Parch': [0, 0, 1, 0, 0],
    'Ticket': ['123456', '654321', '112233', '445566', '778899'],
    'Fare': [100.5, 7.8, 15.0, 120.0, 8.2],
    'Cabin': [np.nan, np.nan, "E44", "B20", "Unknown"],
    'Embarked': ['C', 'S', np.nan, 'S', 'Q']
}

# Creating DataFrame
df = pd.DataFrame(data)

# Check for missing values before processing
print("Initial Missing Values:\n", df.isnull().sum(), "\n")

# Step 1: Fill missing values using .loc to avoid chained assignment warnings
df.loc[:, 'Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.loc[:, 'Age'] = df['Age'].fillna(df['Age'].median())
df.loc[:, 'Cabin'] = df['Cabin'].fillna("Not Assigned")

# Step 2: Binning Age
def categorize_age(age):
    if age <= 12:
        return "Child"
    elif age <= 19:
        return "Teen"
    elif age <= 35:
        return "Adult"
    elif age <= 60:
        return "Middle-Aged"
    else:
        return "Senior"

df.loc[:, 'AgeCategory'] = df['Age'].apply(categorize_age)

# Step 3: Extracting Surnames
df.loc[:, 'Surname'] = df['Name'].apply(lambda x: x.split(',')[0])

# Final cleaned data preview
print("Missing Values After Cleaning:\n", df.isnull().sum(), "\n")
print("Processed Data Sample:\n", df[['PassengerId', 'Survived', 'Age', 'AgeCategory', 'Embarked', 'Cabin', 'Surname']])