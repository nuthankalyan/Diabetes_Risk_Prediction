# Diabetes Risk Prediction System

A realistic diabetes risk prediction system that uses machine learning to assess an individual's risk of developing diabetes based on various health parameters.

## Features

- **High-Accuracy Prediction**: 90-95% accuracy using ensemble learning techniques
- **Comprehensive Risk Assessment**: Evaluates multiple health parameters
- **User-Friendly Interface**: Easy-to-use web interface for predictions
- **Detailed Risk Analysis**: Provides risk factors and recommendations
- **Interactive Dashboard**: Visualize model performance and metrics

## Model Architecture

The system uses an ensemble of machine learning models:
- Support Vector Machine (30%)
- Neural Network (25%)
- Random Forest Classifier (15%)
- Gradient Boosting Classifier (15%)
- XGBoost Classifier (15%)

## Performance Metrics

- Accuracy: 92-94%
- Precision: ~90%
- Recall: ~92%
- F1 Score: ~91%
- ROC AUC: >95%

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Django development server:
```bash
python manage.py runserver
```

4. Open your browser and navigate to:
```
http://127.0.0.1:8000/
```

## Usage

1. Navigate to the prediction page
2. Enter the required health parameters
3. Click "Predict" to get your risk assessment
4. View detailed results and recommendations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 