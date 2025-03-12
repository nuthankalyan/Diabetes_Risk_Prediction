# Diabetes Risk Prediction Application

A Django web application that predicts early-stage diabetes risk based on various health parameters using machine learning.

## Features

- User-friendly web interface for entering patient data
- Machine learning model using stacking ensemble method with cross-validation
- Visualization dashboard showing model performance metrics
- Interactive charts for feature importance, correlation matrix, confusion matrix, and ROC curve

## Technologies Used

- Django 5.0.1
- Python 3.10
- scikit-learn
- pandas
- matplotlib
- seaborn
- Bootstrap 5

## Local Development

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run migrations:
   ```
   python manage.py migrate
   ```
4. Train the model (if not already trained):
   ```
   python predictor/ml_model.py
   ```
5. Run the development server:
   ```
   python manage.py runserver
   ```

## Deployment on Render

### Option 1: Using the Render Dashboard

1. Create a Render account at https://render.com/
2. Connect your GitHub repository
3. Create a new Web Service
4. Select your repository
5. Configure the service:
   - Name: diabetes-prediction (or your preferred name)
   - Environment: Python
   - Region: Choose the closest to your users
   - Branch: main (or your default branch)
   - Build Command: `./build.sh`
   - Start Command: `gunicorn diabetes_prediction.wsgi:application`
6. Add environment variables:
   - `DEBUG`: False
   - `SECRET_KEY`: (generate a secure random string)
   - `SECURE_SSL_REDIRECT`: True
7. Click "Create Web Service"

### Option 2: Using render.yaml (Blueprint)

1. Push the code with the render.yaml file to your GitHub repository
2. Create a Render account and connect your GitHub repository
3. Create a new Blueprint instance
4. Select your repository
5. Render will automatically detect the render.yaml file and configure the service

## Troubleshooting Deployment Issues

If you encounter a 400 Bad Request error:

1. Check the Render logs for specific error messages
2. Ensure the dataset file is properly uploaded to the server
3. Verify that the model training process completed successfully
4. Check that CSRF settings are properly configured
5. Make sure all required environment variables are set

## License

This project is licensed under the MIT License - see the LICENSE file for details. 