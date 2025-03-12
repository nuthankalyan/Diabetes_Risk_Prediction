# Deployment Guide for Diabetes Prediction App on Render

This guide provides step-by-step instructions for deploying the Diabetes Prediction application on Render.

## Prerequisites

- A GitHub account with your project repository
- A Render account (sign up at https://render.com if you don't have one)
- Your application code with all necessary files:
  - `requirements.txt`
  - `build.sh`
  - `Procfile`
  - `render.yaml` (optional, for Blueprint deployment)

## Deployment Steps

### Step 1: Prepare Your Repository

Ensure your repository contains all the necessary files:

1. **requirements.txt** - Lists all Python dependencies
2. **build.sh** - Contains build commands for Render
3. **Procfile** - Specifies the command to run your application
4. **render.yaml** - (Optional) Blueprint configuration for Render
5. **diabetes_dataset.csv** - Required for model training

### Step 2: Connect Render to GitHub

1. Sign in to your Render account
2. Go to the Dashboard
3. Click on "New" and select "Blueprint" or "Web Service"
4. Connect your GitHub account if not already connected
5. Select the repository containing your Django application

### Step 3: Configure the Web Service

If using the Render Dashboard (without Blueprint):

1. Select "Web Service" as the service type
2. Choose your repository
3. Configure the service:
   - **Name**: diabetes-prediction (or your preferred name)
   - **Environment**: Python
   - **Region**: Choose the closest to your users
   - **Branch**: main (or your default branch)
   - **Build Command**: `./build.sh`
   - **Start Command**: `gunicorn diabetes_prediction.wsgi:application`
4. Add environment variables:
   - `DEBUG`: False
   - `SECRET_KEY`: (generate a secure random string or use Render's auto-generation)
   - `SECURE_SSL_REDIRECT`: True
5. Click "Create Web Service"

### Step 4: Monitor the Deployment

1. Render will start building and deploying your application
2. Monitor the build logs for any errors
3. The deployment process includes:
   - Installing dependencies
   - Collecting static files
   - Running database migrations
   - Training the machine learning model

### Step 5: Verify the Deployment

1. Once deployment is complete, click on the provided URL to access your application
2. Test the application by:
   - Navigating to the home page
   - Submitting a prediction form
   - Checking the dashboard for visualizations

## Troubleshooting Common Issues

### 400 Bad Request Error

If you encounter a 400 Bad Request error:

1. Check the Render logs for specific error messages
2. Verify that the `ALLOWED_HOSTS` setting includes `.render.com` and `*.onrender.com`
3. Ensure CSRF settings are properly configured:
   ```python
   CSRF_TRUSTED_ORIGINS = ['https://*.render.com', 'https://*.onrender.com']
   CSRF_COOKIE_SECURE = True
   SESSION_COOKIE_SECURE = True
   ```

### Model Training Failures

If the model fails to train during deployment:

1. Check if `diabetes_dataset.csv` is present in the repository
2. Verify the file path in `ml_model.py`
3. Check the build logs for any Python errors during model training
4. Consider pre-training the model and committing the model files to the repository

### Static Files Not Loading

If static files are not loading properly:

1. Ensure `whitenoise` is installed and configured in `settings.py`
2. Verify that `collectstatic` is running in the build process
3. Check that `STATIC_ROOT` and `STATICFILES_DIRS` are properly configured

## Maintenance and Updates

### Updating Your Application

1. Make changes to your code locally
2. Commit and push to GitHub
3. Render will automatically deploy the changes if `autoDeploy` is enabled
4. Otherwise, manually trigger a new deployment from the Render dashboard

### Monitoring

1. Use Render's built-in logs to monitor application performance
2. Set up alerts for application errors or downtime
3. Regularly check the application to ensure it's functioning correctly

## Additional Resources

- [Render Documentation](https://render.com/docs)
- [Django Deployment Checklist](https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/)
- [Gunicorn Configuration](https://docs.gunicorn.org/en/stable/configure.html) 