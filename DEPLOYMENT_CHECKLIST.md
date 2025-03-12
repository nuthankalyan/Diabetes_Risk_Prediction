# Deployment Checklist for Render

Use this checklist to ensure you've completed all necessary steps before deploying your Diabetes Prediction application to Render.

## Pre-Deployment Preparation

- [ ] Run `python check_deployment.py` to verify deployment readiness
- [ ] Ensure all required files are present:
  - [ ] `requirements.txt` with all dependencies
  - [ ] `build.sh` with build commands
  - [ ] `Procfile` with correct web process command
  - [ ] `render.yaml` (optional, for Blueprint deployment)
  - [ ] `diabetes_dataset.csv` for model training
- [ ] Verify Django settings are properly configured:
  - [ ] `ALLOWED_HOSTS` includes `.render.com` and `*.onrender.com`
  - [ ] `ALLOWED_HOSTS` includes your specific app domain (e.g., `your-app-name.onrender.com`)
  - [ ] `CSRF_TRUSTED_ORIGINS` includes Render domains and your specific app domain
  - [ ] `STATIC_ROOT` is configured
  - [ ] WhiteNoise middleware is enabled
- [ ] Consider pre-training the model:
  - [ ] Run `python pretrain_model.py` to generate model files
  - [ ] Commit the generated model files to your repository

## Git Repository Setup

- [ ] Initialize Git repository (if not already done):
  ```
  git init
  ```
- [ ] Add all files to Git:
  ```
  git add .
  ```
- [ ] Commit changes:
  ```
  git commit -m "Prepare for Render deployment"
  ```
- [ ] Create a GitHub repository
- [ ] Add the remote repository:
  ```
  git remote add origin https://github.com/yourusername/your-repo-name.git
  ```
- [ ] Push to GitHub:
  ```
  git push -u origin main
  ```

## Render Deployment

### Option 1: Using the Render Dashboard

- [ ] Sign in to your Render account
- [ ] Go to the Dashboard
- [ ] Click on "New" and select "Web Service"
- [ ] Connect your GitHub account if not already connected
- [ ] Select your repository
- [ ] Configure the service:
  - [ ] Name: diabetes-prediction (or your preferred name)
  - [ ] Environment: Python
  - [ ] Region: Choose the closest to your users
  - [ ] Branch: main (or your default branch)
  - [ ] Build Command: `./build.sh`
  - [ ] Start Command: `gunicorn diabetes_prediction.wsgi:application`
- [ ] Add environment variables:
  - [ ] `DEBUG`: False
  - [ ] `SECRET_KEY`: (generate a secure random string or use Render's auto-generation)
  - [ ] `SECURE_SSL_REDIRECT`: True
- [ ] Click "Create Web Service"

### Option 2: Using Blueprint (with render.yaml)

- [ ] Sign in to your Render account
- [ ] Go to the Dashboard
- [ ] Click on "New" and select "Blueprint"
- [ ] Connect your GitHub account if not already connected
- [ ] Select your repository
- [ ] Render will automatically detect the render.yaml file and configure the service

## Post-Deployment Verification

- [ ] Monitor the build logs for any errors
- [ ] Once deployment is complete, visit your application URL
- [ ] Test the application functionality:
  - [ ] Navigate to the home page
  - [ ] Submit a prediction form
  - [ ] Check the dashboard for visualizations
- [ ] Check for any errors in the Render logs
- [ ] Verify that static files are loading correctly
- [ ] Confirm that the model has been trained successfully

## Troubleshooting

If you encounter issues during deployment:

- [ ] Check the Render logs for specific error messages
- [ ] If you see a DisallowedHost error, add your specific app domain to ALLOWED_HOSTS
- [ ] Verify that the dataset file is properly uploaded
- [ ] Ensure all environment variables are set correctly
- [ ] Check that the build script is executing properly
- [ ] Verify that the model training process completed successfully

## Resources

- Detailed instructions are available in the `DEPLOYMENT_GUIDE.md` file
- For specific deployment issues, refer to the Render documentation: https://render.com/docs
- For Django deployment best practices: https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/ 