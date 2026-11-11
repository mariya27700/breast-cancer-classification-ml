import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)
os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/model_results", exist_ok=True)

print("\n========== BREAST CANCER INTEGRATED PROJECT ==========")


df = pd.read_csv("data.csv")

print("\n========== DATASET INFORMATION ==========")

print("Original Dataset Shape:", df.shape)

print("\nFirst Five Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


print("\n========== DATA CLEANING ==========")

# Remove duplicate records
print("Duplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()

# Remove unnecessary columns
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

# Clean column names
df.columns = (
    df.columns
    .str.lower()
    .str.strip()
    .str.replace(" ", "_", regex=False)
)

# Convert diagnosis into consistent labels
df["diagnosis"] = df["diagnosis"].str.strip().str.upper()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# Check data types and numerical columns
for column in df.columns:
    if column != "diagnosis":
        df[column] = pd.to_numeric(df[column], errors="coerce")

# Save cleaned dataset
df.to_csv("outputs/cleaned_data.csv", index=False)

print("\nCleaned Dataset Shape:", df.shape)
print("Cleaned dataset saved successfully.")



print("\n========== EXPLORATORY DATA ANALYSIS ==========")

# Diagnosis distribution
print("\nDiagnosis Distribution:")
print(df["diagnosis"].value_counts())

print("\nDiagnosis Percentage:")
print(
    (df["diagnosis"].value_counts(normalize=True) * 100).round(2)
)

# Average feature values
print("\nAverage Feature Values:")
print(
    df.groupby("diagnosis")[
        ["radius_mean", "area_mean", "perimeter_mean",
         "smoothness_mean", "compactness_mean"]
    ].mean().round(3)
)

# Chart 1: Diagnosis Distribution
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="diagnosis",
    hue="diagnosis",
    palette="Set2",
    legend=False
)

plt.title("Distribution of Benign and Malignant Cases")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Samples")
plt.tight_layout()

plt.savefig("outputs/charts/01_diagnosis_distribution.png")
plt.show()
plt.close()


# Chart 2: Average Tumour Radius
plt.figure(figsize=(7, 5))

sns.barplot(
    data=df,
    x="diagnosis",
    y="radius_mean",
    hue="diagnosis",
    palette="Set2",
    legend=False
)

plt.title("Average Tumour Radius by Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Average Radius")
plt.tight_layout()

plt.savefig("outputs/charts/02_average_radius.png")
plt.show()
plt.close()


# Chart 3: Tumour Area Distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="area_mean",
    hue="diagnosis",
    kde=True,
    bins=30
)

plt.title("Distribution of Tumour Area")
plt.xlabel("Mean Tumour Area")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig("outputs/charts/03_area_distribution.png")
plt.show()
plt.close()


# Chart 4: Tumour Perimeter Box Plot
plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="diagnosis",
    y="perimeter_mean",
    hue="diagnosis",
    palette="Set2",
    legend=False
)

plt.title("Tumour Perimeter by Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Mean Perimeter")
plt.tight_layout()

plt.savefig("outputs/charts/04_perimeter_boxplot.png")
plt.show()
plt.close()


# Chart 5: Radius and Area Relationship
plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=df,
    x="radius_mean",
    y="area_mean",
    hue="diagnosis",
    palette="Set1"
)

plt.title("Relationship Between Tumour Radius and Area")
plt.xlabel("Mean Radius")
plt.ylabel("Mean Area")
plt.tight_layout()

plt.savefig("outputs/charts/05_radius_vs_area.png")
plt.show()
plt.close()


# Chart 6: Correlation Heatmap
selected_features = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean"
]

plt.figure(figsize=(11, 8))

sns.heatmap(
    df[selected_features].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Selected Features")
plt.tight_layout()

plt.savefig("outputs/charts/06_correlation_heatmap.png")
plt.show()
plt.close()


print("\n========== STATISTICAL ANALYSIS ==========")

# Correlation between radius and area
correlation = df["radius_mean"].corr(df["area_mean"])

print("\nCorrelation Between Radius and Area:", round(correlation, 3))

# Hypothesis testing
benign_radius = df.loc[
    df["diagnosis"] == "B", "radius_mean"
].dropna()

malignant_radius = df.loc[
    df["diagnosis"] == "M", "radius_mean"
].dropna()

t_statistic, p_value = ttest_ind(
    benign_radius,
    malignant_radius,
    equal_var=False
)

print("\nIndependent T-Test Results:")
print("T-Statistic:", round(t_statistic, 4))
print("P-Value:", p_value)

if p_value < 0.05:
    print("Result: A statistically significant difference was observed.")
else:
    print("Result: No statistically significant difference was observed.")

# Save statistical results
statistical_results = pd.DataFrame({
    "Measurement": [
        "Radius and Area Correlation",
        "T-Test Statistic",
        "T-Test P-Value"
    ],
    "Value": [
        correlation,
        t_statistic,
        p_value
    ]
})

statistical_results.to_csv(
    "outputs/model_results/statistical_results.csv",
    index=False
)


print("\n========== MACHINE LEARNING ==========")

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"].map({"B": 0, "M": 1})

# 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="linear",
            probability=True,
            random_state=42
        ))
    ])
}

results = []
trained_models = {}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for name, model in models.items():

    print("\nTraining:", name)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    # Cross-validation on training data
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    mean_cv = cv_scores.mean()

    results.append({
        "Model": name,
        "Test Accuracy": accuracy,
        "Precision (M)": precision,
        "Recall (M)": recall,
        "F1 Score (M)": f1,
        "Mean CV Accuracy": mean_cv
    })

    trained_models[name] = model

    print("Test Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision (M):", round(precision, 4))
    print("Recall (M):", round(recall, 4))
    print("F1 Score (M):", round(f1, 4))
    print("Mean CV Accuracy:", round(mean_cv * 100, 2), "%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Malignant"],
            zero_division=0
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malignant"]
    )

    display.plot(cmap="Blues")
    plt.title(name + " - Confusion Matrix")
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")

    plt.savefig(
        f"outputs/charts/{safe_name}_confusion_matrix.png"
    )
    plt.show()
    plt.close()



print("\n========== MODEL COMPARISON ==========")

results_df = pd.DataFrame(results)

print(results_df.round(4))

results_df.to_csv(
    "outputs/model_results/model_comparison.csv",
    index=False
)

# Accuracy comparison chart
plt.figure(figsize=(8, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Test Accuracy",
    hue="Model",
    palette="Set2",
    legend=False
)

plt.title("Machine Learning Model Comparison")
plt.ylabel("Test Accuracy")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig("outputs/charts/07_model_comparison.png")
plt.show()
plt.close()



plt.figure(figsize=(7, 6))

for name, model in trained_models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    auc_score = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc_score:.3f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()

plt.savefig("outputs/charts/08_roc_curve.png")
plt.show()
plt.close()



print("\n========== DECISION TREE TUNING ==========")

tree_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", DecisionTreeClassifier(random_state=42))
])

parameters = {
    "model__max_depth": [2, 3, 4, 5, None]
}

grid_search = GridSearchCV(
    tree_pipeline,
    parameters,
    cv=cv,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", round(grid_search.best_score_, 4))

best_tree = grid_search.best_estimator_

tuned_predictions = best_tree.predict(X_test)

tuned_results = {
    "Accuracy": accuracy_score(y_test, tuned_predictions),
    "Precision (M)": precision_score(
        y_test, tuned_predictions, zero_division=0
    ),
    "Recall (M)": recall_score(
        y_test, tuned_predictions, zero_division=0
    ),
    "F1 Score (M)": f1_score(
        y_test, tuned_predictions, zero_division=0
    )
}

print("\nTuned Decision Tree Results:")
print(pd.Series(tuned_results).round(4))

pd.DataFrame([tuned_results]).to_csv(
    "outputs/model_results/tuned_decision_tree.csv",
    index=False
)

print("\n========== SUMMARY ==========")

best_model_name = results_df.loc[
    results_df["Test Accuracy"].idxmax(),
    "Model"
]

print("Total Samples:", len(df))
print("Total Numerical Features:", X.shape[1])
print("Best Model by Test Accuracy:", best_model_name)

print("\nCorrelation Between Radius and Area:", round(correlation, 3))


