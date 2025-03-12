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

## Deployment Options

### Heroku Deployment

1. Create a Heroku account and install the Heroku CLI
2. Login to Heroku:
   ```
   heroku login
   ```
3. Create a new Heroku app:
   ```
   heroku create your-app-name
   ```
4. Set environment variables:
   ```
   heroku config:set SECRET_KEY=your_secret_key
   heroku config:set DEBUG=False
   ```
5. Push to Heroku:
   ```
   git push heroku main
   ```
6. Run migrations:
   ```
   heroku run python manage.py migrate
   ```
7. Train the model on Heroku:
   ```
   heroku run python predictor/ml_model.py
   ```

### PythonAnywhere Deployment

1. Create a PythonAnywhere account
2. Upload your code to PythonAnywhere (via Git or manual upload)
3. Create a virtual environment and install dependencies:
   ```
   mkvirtualenv --python=/usr/bin/python3.10 myenv
   pip install -r requirements.txt
   ```
4. Configure a new web app with manual configuration (Django)
5. Set the WSGI configuration file to point to your Django project
6. Set environment variables in the WSGI configuration file
7. Collect static files:
   ```
   python manage.py collectstatic
   ```
8. Train the model:
   ```
   python predictor/ml_model.py
   ```

### Render Deployment

1. Create a Render account and connect your GitHub repository
2. Create a new Web Service
3. Select your repository
4. Configure the service:
   - Build Command: `pip install -r requirements.txt && python manage.py collectstatic --noinput`
   - Start Command: `gunicorn diabetes_prediction.wsgi:application`
5. Add environment variables:
   - SECRET_KEY
   - DEBUG=False
6. Deploy the service
7. After deployment, use the Render shell to train the model:
   ```
   python predictor/ml_model.py
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 