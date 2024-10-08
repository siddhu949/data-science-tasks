# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
titanic_data = pd.read_csv('tested.csv')

# Data cleaning and preprocessing
# Convert 'Sex' and 'Embarked' to numerical values
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data['Embarked'] = titanic_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Fill missing Age values with the median
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)

# Drop Cabin as it has too many missing values
titanic_data.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

# Fill missing Embarked values with the most frequent value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Split the dataset into features and labels
X = titanic_data.drop(['Survived', 'PassengerId'], axis=1)  # Features
y = titanic_data['Survived']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# --- Additional Data Analysis and Visualization ---

# 1. Survival rate by Gender
plt.figure(figsize=(8, 6))
sns.barplot(x='Sex', y='Survived', data=titanic_data, palette='coolwarm')
plt.title('Survival Rate by Gender')
plt.xlabel('Gender (0 = Male, 1 = Female)')
plt.ylabel('Survival Rate')
plt.show()

# 2. Survival rate by Passenger Class (Pclass)
plt.figure(figsize=(8, 6))
sns.barplot(x='Pclass', y='Survived', data=titanic_data, palette='coolwarm')
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Pclass (1 = 1st, 2 = 2nd, 3 = 3rd)')
plt.ylabel('Survival Rate')
plt.show()

# 3. Age distribution of passengers who survived vs those who didn't
plt.figure(figsize=(10, 6))
sns.histplot(titanic_data[titanic_data['Survived'] == 1]['Age'], bins=30, kde=False, color='green', label='Survived')
sns.histplot(titanic_data[titanic_data['Survived'] == 0]['Age'], bins=30, kde=False, color='red', label='Did Not Survive')
plt.title('Age Distribution of Survived vs Did Not Survive')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()

# 4. Survival rate by number of siblings/spouses aboard (SibSp)
plt.figure(figsize=(8, 6))
sns.barplot(x='SibSp', y='Survived', data=titanic_data, palette='coolwarm')
plt.title('Survival Rate by Siblings/Spouses Aboard')
plt.xlabel('Siblings/Spouses Aboard (SibSp)')
plt.ylabel('Survival Rate')
plt.show()

# 5. Survival rate by number of parents/children aboard (Parch)
plt.figure(figsize=(8, 6))
sns.barplot(x='Parch', y='Survived', data=titanic_data, palette='coolwarm')
plt.title('Survival Rate by Parents/Children Aboard')
plt.xlabel('Parents/Children Aboard (Parch)')
plt.ylabel('Survival Rate')
plt.show()

# 6. Correlation Heatmap to show relationships between features
plt.figure(figsize=(10, 8))
sns.heatmap(titanic_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Titanic Dataset Features')
plt.show()
