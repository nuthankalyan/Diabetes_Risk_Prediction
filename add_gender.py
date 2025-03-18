import pandas as pd
import numpy as np

# Read the dataset
df = pd.read_csv('diabetes_dataset.csv')

# Convert WHR to numeric, handling any potential errors
df['WHR'] = pd.to_numeric(df['WHR'], errors='coerce')

# Create gender column based on pregnancies and WHR
def determine_gender(row):
    if row['Pregnancies'] > 0:
        return 'Female'
    else:
        # Use WHR to make an educated guess
        if pd.isna(row['WHR']):
            return 'Unknown'
        if row['WHR'] < 0.85:
            return 'Female'
        else:
            return 'Male'

# Add gender column
df['Gender'] = df.apply(determine_gender, axis=1)

# Save the updated dataset
df.to_csv('diabetes_dataset_with_gender.csv', index=False)

print("Gender column has been added and saved to 'diabetes_dataset_with_gender.csv'")
print("\nGender distribution:")
print(df['Gender'].value_counts()) 