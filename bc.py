# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# 2. Load Dataset

df = pd.read_csv("data.csv")

print("Original Dataset Shape:", df.shape)
print(df.head())


# 3. Data Preprocessing

# Remove duplicate rows
df = df.drop_duplicates()

# Clean column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Remove columns that contain only missing values
df = df.dropna(axis=1, how="all")

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check dataset information
print("\nDataset Information:")
print(df.info())


# 4. Separate Features and Target

# Remove ID and diagnosis from input features
X = df.drop(columns=["id", "diagnosis"])

# Diagnosis is the target variable
y = df["diagnosis"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nDiagnosis Distribution:")
print(y.value_counts())


# 5. Split Dataset into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# 6. Define Machine Learning Models

# Pipelines apply scaling correctly for Logistic Regression and SVM.
# Decision Tree does not require scaling.

models = {
    "Logistic Regression": (
        make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000)
        ),
        X_train,
        X_test
    ),

    "Decision Tree": (
        DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        ),
        X_train,
        X_test
    ),

    "SVM": (
        make_pipeline(
            StandardScaler(),
            SVC(kernel="linear")
        ),
        X_train,
        X_test
    )
}


# 7. Train Models and Evaluate Performance

for name, (model, xtrain, xtest) in models.items():

    print("\n================================")
    print(name)
    print("================================")

    # Train the model
    model.fit(xtrain, y_train)

    # Make predictions
    y_pred = model.predict(xtest)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="M"
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="M"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="M"
    )

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Confusion Matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["M", "B"]
    )

    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Cross Validation
    cv_scores = cross_val_score(
        model,
        xtrain,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    print("\nCross Validation Scores:")
    print(cv_scores)

    print("Mean CV Accuracy:", cv_scores.mean())


# 8. Hyperparameter Tuning using GridSearchCV

# Decision Tree parameters to test

param_grid = {
    "max_depth": [1, 2, 3, 4, 5, None]
}

# Create Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Apply GridSearchCV
grid_search = GridSearchCV(
    estimator=dt_model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

# Train GridSearchCV
grid_search.fit(X_train, y_train)

# Display best parameters
print("\n================================")
print("GRID SEARCH RESULTS")
print("================================")

print("Best Parameters:", grid_search.best_params_)

print("Best Cross Validation Accuracy:", grid_search.best_score_)


# 9. Evaluate Tuned Decision Tree Model

best_model = grid_search.best_estimator_

# Predictions using tuned model
y_pred_best = best_model.predict(X_test)

# Evaluation metrics
print("\nTUNED DECISION TREE PERFORMANCE")

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred_best)
)

print(
    "Precision:",
    precision_score(y_test, y_pred_best, pos_label="M")
)

print(
    "Recall:",
    recall_score(y_test, y_pred_best, pos_label="M")
)

print(
    "F1 Score:",
    f1_score(y_test, y_pred_best, pos_label="M")
)

# Confusion Matrix
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred_best,
        labels=["M", "B"]
    )
)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))


# 10. Visualize Model Comparison

model_names = [
    "Logistic Regression",
    "Decision Tree",
    "SVM"
]

accuracy_values = []

for name, (model, xtrain, xtest) in models.items():

    model.fit(xtrain, y_train)

    y_pred = model.predict(xtest)

    accuracy_values.append(
        accuracy_score(y_test, y_pred)
    )


plt.figure(figsize=(8, 5))

plt.bar(model_names, accuracy_values)

plt.title("Machine Learning Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")

plt.ylim(0, 1)

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()