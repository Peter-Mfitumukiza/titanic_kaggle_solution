import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# 1.  Loading data
train_data = pd.read_csv("/kaggle/input/titanic-data/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic-data/test.csv")

train_data

# 2. Analyse data

print(train_data.head())

# Get a summary of the dataset
print(train_data.info())

# Get descriptive statistics
print(train_data.describe())

# 3. Clean data

# fill missing age with the median age
train_data["Age"].fillna(train_data["Age"].median(), inplace=True)
test_data["Age"].fillna(test_data["Age"].median(), inplace=True)

# Fill missing values for the 'Embarked' feature with the most common port
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)

# Fill missing values for fare with median fare in test data
test_data['Fare'].fillna(test_data['Fare'].mode()[0], inplace=True)

# Convert 'Sex' feature to numerical
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' feature to numerical
train_data['Embarked'] = train_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
test_data['Embarked'] = test_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Drop irrelevant features
train_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Create a new feature 'FamilySize' from 'SibSp' and 'Parch'
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

# Create a new feature 'IsAlone'
train_data['IsAlone'] = np.where(train_data['FamilySize'] > 1, 0, 1)
test_data['IsAlone'] = np.where(test_data['FamilySize'] > 1, 0, 1)

# 4.  Prepare the data for training

X = train_data.drop(['Survived'], axis=1)
y = train_data['Survived']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the validation set
y_pred = clf.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation set accuracy: {accuracy:.2f}')

# 5. Finetuning the Model

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}

# Perform grid search
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f'Best parameters found by grid search: {best_params}')

# Train the classifier with the best parameters
best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)

# Predict on the validation set
y_pred = best_clf.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation set accuracy after tuning: {accuracy:.2f}')

# 6. Predict on the test set
test_pred = best_clf.predict(test_data)

# Prepare the submission file
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_pred
})

submission.to_csv('titanic_submission.csv', index=False)
submission