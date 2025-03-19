import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_class_data(class_label, n_samples):
    """
    Generate data for a specific class with distinct feature distributions
    to ensure better class separation
    """
    # Base values that differ by class
    if class_label == 'No':
        # Healthy patients: lower values for risk factors
        age_range = (20, 45)
        bmi_range = (18, 25)
        hba1c_range = (4.0, 5.6)
        chol_range = (120, 180)
        tg_range = (50, 120)
        hdl_range = (45, 70)
        ldl_range = (60, 100)
        vldl_range = (5, 20)
        urea_range = (2.0, 7.0)
        cr_range = (0.5, 1.0)
    elif class_label == 'Possible':
        # Pre-diabetic patients: intermediate values
        age_range = (35, 65)
        bmi_range = (24, 30)
        hba1c_range = (5.5, 6.5)
        chol_range = (160, 220)
        tg_range = (100, 200)
        hdl_range = (35, 55)
        ldl_range = (90, 140)
        vldl_range = (15, 30)
        urea_range = (5.0, 12.0)
        cr_range = (0.8, 1.5)
    else:  # 'Yes'
        # Diabetic patients: higher values for risk factors
        age_range = (50, 90)
        bmi_range = (28, 40)
        hba1c_range = (6.4, 9.0)
        chol_range = (190, 300)
        tg_range = (180, 300)
        hdl_range = (20, 45)
        ldl_range = (120, 190)
        vldl_range = (25, 40)
        urea_range = (10.0, 20.0)
        cr_range = (1.2, 3.0)
    
    # Generate random data within these ranges
    data = {
        'Gender': np.random.choice(['M', 'F'], n_samples),
        'Age': np.round(np.random.uniform(age_range[0], age_range[1], n_samples), 2),
        'Urea': np.round(np.random.uniform(urea_range[0], urea_range[1], n_samples), 2),
        'Cr': np.round(np.random.uniform(cr_range[0], cr_range[1], n_samples), 2),
        'HbA1c': np.round(np.random.uniform(hba1c_range[0], hba1c_range[1], n_samples), 2),
        'Chol': np.round(np.random.uniform(chol_range[0], chol_range[1], n_samples), 2),
        'TG': np.round(np.random.uniform(tg_range[0], tg_range[1], n_samples), 2),
        'HDL': np.round(np.random.uniform(hdl_range[0], hdl_range[1], n_samples), 2),
        'LDL': np.round(np.random.uniform(ldl_range[0], ldl_range[1], n_samples), 2),
        'VLDL': np.round(np.random.uniform(vldl_range[0], vldl_range[1], n_samples), 2),
        'BMI': np.round(np.random.uniform(bmi_range[0], bmi_range[1], n_samples), 2),
        'CLASS': [class_label] * n_samples
    }
    
    return pd.DataFrame(data)

# Generate data for each class
n_samples_per_class = 724  # Equal samples per class
total_samples = n_samples_per_class * 3

print(f"Generating balanced dataset with {total_samples} samples ({n_samples_per_class} per class)")

# Generate data with distinct distributions for each class
df_no = generate_class_data('No', n_samples_per_class)
df_possible = generate_class_data('Possible', n_samples_per_class)
df_yes = generate_class_data('Yes', n_samples_per_class)

# Combine the data
df = pd.concat([df_no, df_possible, df_yes], ignore_index=True)

# Add ID column
df['ID'] = np.arange(1, len(df) + 1)

# Reorder columns to put ID first
cols = df.columns.tolist()
cols = ['ID'] + [col for col in cols if col != 'ID']
df = df[cols]

# Add some noise to make the data more realistic (but maintain separation)
for col in ['Age', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']:
    # Add small amount of noise (5% of range)
    noise = np.random.normal(0, 0.05 * (df[col].max() - df[col].min()), len(df))
    df[col] = df[col] + noise
    # Round to 2 decimal places
    df[col] = df[col].round(2)
    # Ensure no negative values
    df[col] = df[col].clip(lower=0)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
output_path = 'Balanced_Diabetes_Dataset.csv'
df.to_csv(output_path, index=False)

print(f"Balanced dataset created with {len(df)} samples and saved to {output_path}")
print(f"Sample data:")
print(df.head())
print(f"\nClass distribution:")
print(df['CLASS'].value_counts())

# Print some statistics to verify separation
print("\nClass separation statistics:")
for col in ['Age', 'BMI', 'HbA1c', 'Chol', 'TG', 'HDL']:
    print(f"\n{col} by CLASS:")
    print(df.groupby('CLASS')[col].describe()[['mean', 'std', 'min', 'max']]) 