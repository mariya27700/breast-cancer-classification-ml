# 1. IMPORT LIBRARIES

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")

# 2. CREATE OUTPUT FOLDERS

os.makedirs("outputs/charts", exist_ok=True)

# 3. LOAD DATASET

df = pd.read_csv("data.csv")

print("\n========== DATASET INFORMATION ==========")

print("Original Dataset Shape:", df.shape)

print("\nFirst Five Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

# 4. DATA CLEANING

# Remove duplicate rows
df = df.drop_duplicates()

# Remove unnecessary columns
df = df.drop(
    columns=["id", "Unnamed: 32"],
    errors="ignore"
)

# Remove completely empty columns
df = df.dropna(axis=1, how="all")

# Clean column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Convert numerical columns to numeric format
for column in df.columns:
    if column != "diagnosis":
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

# Check missing values
print("\n========== DATA CLEANING ==========")

print("Missing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Remove rows with missing target values
df = df.dropna(subset=["diagnosis"])

# Fill missing numerical values with median
numeric_columns = df.select_dtypes(
    include="number"
).columns

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )

# Standardize diagnosis labels
df["diagnosis"] = (
    df["diagnosis"].astype(str).str.strip().str.upper()
)

# Validate diagnosis values
df = df[df["diagnosis"].isin(["M", "B"])].copy()

print("\nCleaned Dataset Shape:", df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())

# Save cleaned dataset
df.to_csv(
    "outputs/cleaned_data.csv",
    index=False
)

# 5. BASIC EXPLORATORY DATA ANALYSIS

print("\n========== STATISTICAL SUMMARY ==========")

print(df.describe())

print("\nDiagnosis Distribution:")
print(df["diagnosis"].value_counts())

print("\nDiagnosis Percentage:")
print(
    (df["diagnosis"].value_counts(normalize=True) * 100)
    .round(2)
)

# Calculate average scores by diagnosis
features = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean"
]

group_analysis = df.groupby("diagnosis")[features].mean()

print("\nAverage Feature Values by Diagnosis:")
print(group_analysis.round(3))

# 6. VISUALIZATION 1: DIAGNOSIS DISTRIBUTION

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="diagnosis",
    hue="diagnosis",
    palette="Set2",
    legend=False
)

plt.title("Distribution of Benign and Malignant Samples")
plt.xlabel("Diagnosis (B = Benign, M = Malignant)")
plt.ylabel("Number of Samples")

plt.savefig(
    "outputs/charts/01_diagnosis_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 7. VISUALIZATION 2: AVERAGE TUMOR RADIUS

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="diagnosis",
    y="radius_mean",
    hue="diagnosis",
    palette="Set2",
    legend=False
)

plt.title("Average Tumor Radius by Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Average Radius")

plt.savefig(
    "outputs/charts/02_average_tumor_radius.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 8. VISUALIZATION 3: TUMOR AREA DISTRIBUTION

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="area_mean",
    hue="diagnosis",
    bins=30,
    kde=True
)

plt.title("Distribution of Mean Tumor Area")
plt.xlabel("Mean Tumor Area")
plt.ylabel("Frequency")

plt.savefig(
    "outputs/charts/03_tumor_area_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 9. VISUALIZATION 4: TUMOR PERIMETER BOXPLOT

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="diagnosis",
    y="perimeter_mean",
    hue="diagnosis",
    palette="Set2",
    legend=False
)

plt.title("Tumor Perimeter by Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Mean Perimeter")

plt.savefig(
    "outputs/charts/04_tumor_perimeter_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 10. VISUALIZATION 5: RADIUS VS AREA SCATTER PLOT

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="radius_mean",
    y="area_mean",
    hue="diagnosis",
    palette="Set1",
    alpha=0.7
)

plt.title("Relationship Between Tumor Radius and Area")
plt.xlabel("Mean Radius")
plt.ylabel("Mean Area")

plt.savefig(
    "outputs/charts/05_radius_vs_area.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 11. VISUALIZATION 6: CORRELATION HEATMAP

plt.figure(figsize=(14, 10))

correlation = df.select_dtypes(
    include="number"
).corr()

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    linewidths=0.3
)

plt.title("Correlation Between Tumor Features")

plt.savefig(
    "outputs/charts/06_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 12. VISUALIZATION 7: COMPACTNESS VIOLIN PLOT

plt.figure(figsize=(8, 5))

sns.violinplot(
    data=df,
    x="diagnosis",
    y="compactness_mean",
    hue="diagnosis",
    palette="Set2",
    legend=False
)

plt.title("Tumor Compactness by Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Mean Compactness")

plt.savefig(
    "outputs/charts/07_compactness_violinplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 13. VISUALIZATION 8: PAIR PLOT

selected_features = [
    "radius_mean",
    "texture_mean",
    "area_mean",
    "smoothness_mean",
    "diagnosis"
]

pair_plot = sns.pairplot(
    df[selected_features],
    hue="diagnosis",
    corner=True,
    diag_kind="hist",
    plot_kws={"alpha": 0.6, "s": 25}
)

pair_plot.fig.suptitle(
    "Pairwise Relationships Between Tumor Features",
    y=1.02
)

pair_plot.savefig(
    "outputs/charts/08_pairplot.png",
    dpi=300
)

plt.show()

# 14. ADDITIONAL NUMERICAL ANALYSIS

print("\n========== KEY FINDINGS ==========")

# Diagnosis counts
print("\nDiagnosis Counts:")
print(df["diagnosis"].value_counts())

# Average tumor measurements
print("\nAverage Tumor Measurements:")
print(
    df.groupby("diagnosis")[
        ["radius_mean", "area_mean", "perimeter_mean"]
    ].mean().round(3)
)

# Correlation between radius and area
radius_area_corr = df["radius_mean"].corr(
    df["area_mean"]
)

print(
    "\nCorrelation Between Radius and Area:",
    round(radius_area_corr, 3)
)

# Highest and lowest average tumor radius
print(
    "\nMaximum Tumor Radius:",
    df["radius_mean"].max()
)

print(
    "Minimum Tumor Radius:",
    df["radius_mean"].min()
)

# Diagnosis with highest average area
average_area = df.groupby("diagnosis")["area_mean"].mean()

print(
    "\nDiagnosis with Higher Average Tumor Area:",
    average_area.idxmax()
)

# 15. FINAL SUMMARY

print("\n========== PROJECT SUMMARY ==========")

print("Total Samples:", len(df))

print("Total Numerical Features:", len(numeric_columns))

print("Total Visualizations Created: 8")

print("Cleaned Dataset Saved: outputs/cleaned_data.csv")

print("Charts Saved: outputs/charts/")

print("\nWeek 5 Data Visualization Project Completed!")
