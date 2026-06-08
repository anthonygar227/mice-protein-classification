# Load the data
from ucimlrepo import fetch_ucirepo

# Download the Mice Protein Expression dataset
data = fetch_ucirepo(id=342)

# The fetures (X) are the protein measurements
# The targets (y) hold the labels, including the 8-way class
X = data.data.features
y = data.data.targets

# Grab the mouse ID for each row, so we can later split by mouse (not by row).
# The IDs come back from the UCI dataset's "ids" table.
ids = data.data.ids
print("ID columns available:", list(ids.columns))
mouse_id = ids["MouseID"].str.split("_").str[0]		# "309_1" -> "309"

print("Features shape:", X.shape) # (rows, columns)
print("Targets shape:", y.shape)
print()
print("First few features column names:", list(X.columns[:5]))
print("Target column names:", list(y.columns))
print()
print("First 3 rows of features:")
print(X.head(3))
print()
print("--- A closer look ---")

# How mant colums are text vs text?
print("Numeric columns:", X.select_dtypes(include="number").shape[1])
print("Text columns:", list(X.select_dtypes(exclude="number").columns))

# What are the 8 classes, and how many samples each?
print()
print("Class counts:")
print(y["class"].value_counts())

print()
print("--- Step 2: Cleaning ---")

# Keep only the numeric protein columns (drops Genotype, Treatment, Behavior)
X = X.select_dtypes(include="number")
print("Shape after keeping only proteins:", X.shape)

# How many missing values does each protein have? Show the worst offenders.
missing_per_protein = X.isna().sum().sort_values(ascending=False)
print()
print("Proteins with the most missing values:")
print(missing_per_protein.head(10))

print()
print("Total missing values in the whole table:", int(X.isna().sum().sum()))

print()
print("--- Step 2b: Drop proteins with too many missing values ---")

# 15% of our samples is the cutoff. Drop proteins missing more than that.
cutoff = 0.15 * len(X)
print("Cutoff (15% of", len(X), "samples):", cutoff, "missing values")

too_many_missing = X.columns[X.isna().sum() > cutoff]
print("Dropping these proteins: ", list(too_many_missing))

X = X.drop(columns=too_many_missing)
print("Shape after dropping:", X.shape)

print()
print("--- Step 3: Split into train and test---")
from sklearn.model_selection import GroupShuffleSplit

# We need the label column to go along with the features
labels = y["class"]

# GroupShuffleSplit keeps every measuremnet of one mouse entirely on one side.
# This prevents the same mouse appearing in both train and test.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, labels, groups=mouse_id))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])
# Confirm no mouse is in both sets (should print 0)
overlap = set(mouse_id.iloc[train_idx]) & set(mouse_id.iloc[test_idx])
print("Mice appearing in BOTH train and test (should be 0):", len(overlap))
print("Training still has missing values:", int(X_train.isna().sum().sum()))
print("Test still has missing values:", int(X_test.isna().sum().sum()))

print()
print("--- Step 4: Fill missing values (impute) ---")
from sklearn.impute import SimpleImputer

# SimpleImputer with strategy="median" learns each protein's median
imputer = SimpleImputer(strategy="median")

# .fit_transform on TRAINING: learns the medians AND fills training blanks
X_train = imputer.fit_transform(X_train)

# .transform on TEST: fills test blanks using the TRAINING medians (no peeking).
X_test = imputer.transform(X_test)

# Check: there should now be zero missing values in both
import numpy as np
print("Training missing after fill:", int(np.isnan(X_train).sum()))
print("Test missing after fill:", int(np.isnan(X_test).sum()))
print("Training shape:", X_train.shape)

print()
print("--- Step 5: Scale the proteins ---")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# fit_tranform on training: learns each protein's mean & spread, then scales
X_train = scaler.fit_transform(X_train)

# transform on test: scales using the TRAINING mean & spread (no peeking).
X_test = scaler.transform(X_test)

# After scaling, each protein's training values average 0 with spread 1
print("Mean of first protein after scaling (should be 0):", round(X_train[:, 0].mean(), 4))
print("Spread of first protein after scaling (should be 1):", round(X_train[:, 0].std(), 4))

print()
print("--- Step 6: Select the most informative proteins ---")
from sklearn.feature_selection import SelectKBest, f_classif

# Keep the 30 proteins most associated with the class label.
selector = SelectKBest(score_func=f_classif, k=30)

# fit_transform on training: scores protein using TRAINING data, keeps top 30
X_train = selector.fit_transform(X_train, y_train)

# transform on test: keeps the SAME 30 proteins chosen from training.
X_test = selector.transform(X_test)

print("Training shape after selection:", X_train.shape)
print("Test shape after selection:", X_test.shape)

print()
print("--- Step 7: Train the Random Forest and evaluate ---")
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Baseline: always guess the most common class. This is the "do nothing" bar.
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
print("Baseline accuracy (guessing most common class):",
	round(accuracy_score(y_test, dummy.predict(X_test)), 3))

# The Random Forest: 300 tress voting.
rf = RandomForestClassifier(n_estimators=300, random_state=42)
# Cross-validation grouped by mouse so no mouse spans two folds
from sklearn.model_selection import GroupKFold
train_mouse_id = mouse_id.iloc[train_idx]	# mouse IDs for the training rows
gkf = GroupKFold(n_splits=5)
cv_scores = cross_val_score(rf, X_train, y_train, cv=gkf, groups=train_mouse_id)
print("Cross-validated accuracy:", round(cv_scores.mean(), 3),
	"+/-", round(cv_scores.std(), 3))

# Train on all the training data and test on the sealed test set.
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Held-out test accuracy:", round(accuracy_score(y_test, y_pred), 3))

print()
print("Detailed report per class:")
print(classification_report(y_test, y_pred, zero_division=0))

print()
print("--- Step 8: Which proteins drove the predictions? ---")
from sklearn.inspection import permutation_importance
import pandas as pd

# Recover the names of the 30 proteins the selector kept in Step 6
kept_mask = selector.get_support() 	# True/false for each of the 72
all_proteins = X.columns 	# original 72 protein names
kept_proteins = all_proteins[kept_mask] 	# the 30 that survived

# Permutation importance: scramble each protein, measure the accuracy drop.
result = permutation_importance(
	rf, X_test, y_test, n_repeats=20, random_state=42
)

# Pair each protein with its importance score and sort, highest first.
importance = pd.Series(result.importances_mean, index=kept_proteins)
importance = importance.sort_values(ascending=False)

print("Top 15 proteins driving the prediction:")
print(importance.head(15).round(4))
