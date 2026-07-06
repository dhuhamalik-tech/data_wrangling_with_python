# ==========================================================
# Assignment: Data Wrangling with Pandas
# Student Name: Dhuha Malik 
# Course: SkillsU
# ==========================================================

# Import required libraries
import pandas as pd
import numpy as np

# ----------------------------------------------------------
# Create the messy dataset
# ----------------------------------------------------------
data = pd.DataFrame({
    'name': ['John Smith', 'Jane Doe', 'Bob Wilson', 'MARY JONES', 'john smith', np.nan, 'Jane doe'],
    'email': ['john@gmail.com', 'jane@yahoo.com', np.nan, 'mary@gmail.com',
              'john2@gmail.com', 'unknown@test.com', 'jane@yahoo.com'],
    'age': [25, 30, -999, 35, 25, 40, 30],
    'category': ['A', 'B', 'A', 'C', 'A', 'B', 'B'],
    'score': [85.5, 90.0, 77.5, 995.0, 85.5, 88.0, 90.0]
})

print("="*60)
print("Original Dataset")
print("="*60)
print(data)

# ==========================================================
# TASK 1: DATA CLEANING
# ==========================================================

print("\n========== TASK 1: DATA CLEANING ==========")

# Remove duplicate names (case-insensitive)
# Convert names to lowercase temporarily for duplicate checking
data = data.loc[
    ~data['name'].fillna('').str.lower().duplicated()
].copy()

# Replace -999 with NaN
data['age'] = data['age'].replace(-999, np.nan)

# Remove rows with missing names
data = data.dropna(subset=['name'])

print("\nCleaned Data:")
print(data)

# ==========================================================
# TASK 2: STRING MANIPULATION
# ==========================================================

print("\n========== TASK 2: STRING MANIPULATION ==========")

# Convert names to title case
data['name'] = data['name'].str.title()

# Extract email domain
data['domain'] = data['email'].str.split('@').str[1]

# Create Gmail boolean column
data['is_gmail'] = data['domain'] == 'gmail.com'

print(data[['name', 'email', 'domain', 'is_gmail']])

# ==========================================================
# TASK 3: CATEGORICAL DATA
# ==========================================================

print("\n========== TASK 3: CATEGORICAL DATA ==========")

# Convert category to categorical type
data['category'] = data['category'].astype('category')

print("\nCategory Data Type:")
print(data['category'].dtype)

# Create dummy variables
dummy_categories = pd.get_dummies(data['category'], prefix='Category')

print("\nDummy Variables:")
print(dummy_categories)

# Frequency count
print("\nFrequency Count:")
print(data['category'].value_counts())

# ==========================================================
# TASK 4: HANDLING OUTLIERS
# ==========================================================

print("\n========== TASK 4: HANDLING OUTLIERS ==========")

# Calculate mean and standard deviation
mean_score = data['score'].mean()
std_score = data['score'].std()

# Outlier condition
outliers = (
    (data['score'] > mean_score + 2 * std_score) |
    (data['score'] < mean_score - 2 * std_score)
)

print("\nOutlier Scores:")
print(data.loc[outliers, ['name', 'score']])

# Replace outliers with the mean
data.loc[outliers, 'score'] = mean_score

print("\nDescriptive Statistics:")
print(data['score'].describe())

# ==========================================================
# TASK 5: DATA TRANSFORMATION
# ==========================================================

print("\n========== TASK 5: DATA TRANSFORMATION ==========")

# Dictionary mapping
category_map = {
    'A': 1,
    'B': 2,
    'C': 3
}

# Map categories
data['category_num'] = data['category'].map(category_map)

# Sort dataframe
sorted_data = data.sort_values(
    by=['category_num', 'score']
)

print("\nSorted Data:")
print(sorted_data)

print("\n========== FINAL DATAFRAME ==========")
print(sorted_data)
