# Ensuring Consistent Model Visualizations

This guide explains how to ensure that model visualizations remain consistent between your local development environment and the deployed application on Render.

## Understanding the Issue

The model training process includes random elements (noise generation, data shuffling, etc.) that can lead to different results each time the model is trained, even with random seeds set. This can cause visualizations to differ between:

- Local development environment
- Deployed application on Render
- Different deployments of the same application

Additionally, differences in file paths, environment variables, and Python versions can cause inconsistencies.

## Solution: Pre-train and Commit Model Files

The most reliable way to ensure consistent visualizations is to pre-train the model locally and commit the model files to your repository. This way, both local and deployed environments will use the exact same model.

### Step 1: Update Your Code

We've made the following changes to ensure consistency:

1. Added a fixed random seed at the beginning of the `train_model()` function
2. Enhanced the `pretrain_model.py` script to create backups and provide clear instructions
3. Modified the `build.sh` script to skip model training if pre-trained model files exist
4. Updated all file paths to use absolute paths based on the current file location
5. Created a debug script to verify model loading and visualizations

### Step 2: Debug the Current State

Run the debug script to check the current state of your model files:

```bash
python debug_model.py
```

This will:
- Check if model files exist and can be loaded
- Calculate hash values for each model file
- Generate and display key visualization data
- Save debug information to a JSON file for comparison

Run this script in both your local environment and on Render (using the Render shell) to compare the results.

### Step 3: Pre-train the Model Locally

Run the following command to pre-train the model:

```bash
python pretrain_model.py
```

This will:
- Train the model with a fixed random seed
- Save the model files to the `predictor/models/` directory
- Create backups of any existing model files
- Provide instructions for committing the files

### Step 4: Commit the Model Files to Your Repository

```bash
git add predictor/models/*.joblib
git commit -m "Add pre-trained model files for consistent visualizations"
git push
```

### Step 5: Deploy to Render

Deploy your application to Render. The `build.sh` script will detect the pre-trained model files and skip the model training step, ensuring that the deployed application uses the same model as your local environment.

### Step 6: Verify Consistency

After deployment, run the debug script again in both environments:

```bash
# Local environment
python debug_model.py

# Render environment (using the Render shell)
python debug_model.py
```

Compare the generated JSON files to verify that the model files are identical and produce the same visualization data.

## Detailed Troubleshooting

If visualizations are still inconsistent, follow these detailed troubleshooting steps:

### 1. Verify Model File Consistency

Compare the hash values of the model files in both environments:

```bash
# Run in both environments
python debug_model.py
```

If the hash values differ, the model files are different. Possible causes:

- Model files were not committed to the repository
- Model files were modified during deployment
- Different versions of the model files are being used

Solution:
- Ensure that the model files are committed to the repository
- Check that the build script is skipping model training
- Verify that the correct model files are being loaded

### 2. Check File Paths

Incorrect file paths can cause the application to load different model files or train a new model. Check the logs for file path errors:

```bash
# Render logs
# Look for messages like "Loading model from: ..."
```

Solution:
- Update all file paths to use absolute paths based on the current file location
- Use `os.path.join()` and `Path()` for cross-platform compatibility
- Print file paths in logs for debugging

### 3. Check Python and Library Versions

Different Python or library versions can cause inconsistencies:

```bash
# Run in both environments
python -c "import sys, sklearn, numpy, pandas; print(f'Python: {sys.version}, scikit-learn: {sklearn.__version__}, numpy: {numpy.__version__}, pandas: {pandas.__version__}')"
```

Solution:
- Specify exact versions in requirements.txt
- Use a virtual environment for local development
- Consider using Docker for development to match the deployment environment

### 4. Check Dataset Loading

Ensure that the same dataset is being used in both environments:

```bash
# Run in both environments
python -c "import pandas as pd; data = pd.read_csv('diabetes_dataset.csv'); print(f'Dataset shape: {data.shape}, First 5 rows hash: {hash(tuple(map(tuple, data.head(5).values.tolist())))}')"
```

Solution:
- Use absolute paths for dataset loading
- Include the dataset in the repository
- Verify that the dataset is not being modified during deployment

### 5. Check Visualization Code

Ensure that the visualization code is identical in both environments:

```bash
# Compare the visualization code files
```

Solution:
- Use version control to ensure code consistency
- Avoid environment-specific code paths
- Use the same templates and static files

## When to Re-train the Model

You should re-train and re-commit the model files when:

1. You make changes to the model architecture or hyperparameters
2. You update the dataset
3. You modify the feature engineering process

Follow the same steps (pre-train locally, commit, deploy) to ensure consistency after any changes.

## Additional Considerations

- Model files can be large. If your repository has size limitations, consider using Git LFS (Large File Storage)
- For very large models, you might need to store them externally (e.g., AWS S3) and download them during deployment
- Always backup your model files before re-training to allow for easy rollback if needed
- Consider adding a version number to your model files to track changes
- Add logging to your application to track which model files are being loaded and used 