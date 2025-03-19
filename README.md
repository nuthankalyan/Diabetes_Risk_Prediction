# Diabetes Risk Prediction System

A high-accuracy diabetes risk prediction system built with Django, Python, and machine learning.

## Features

- **Advanced Diabetes Prediction Model**: Using multiple machine learning algorithms with ensemble techniques
- **Comprehensive Risk Assessment**: Identifies specific risk factors and provides detailed interpretation
- **User-Friendly Web Interface**: Modern, responsive design with intuitive form input
- **REST API**: Programmatic access for integration with other applications
- **Interactive Dashboard**: Visualizes model performance metrics and dataset characteristics

## Technical Components

### Machine Learning Model

- **Model Architecture**: Ensemble of models with optimized weights:
  - Random Forest Classifier (20%)
  - Gradient Boosting Classifier (20%)
  - XGBoost Classifier (20%)
  - Support Vector Machine (20%)
  - Neural Network (20%)

- **Data Processing Techniques**:
  - Feature Engineering: Created 20+ derived features
  - SMOTE for Class Balancing
  - Feature Selection
  - Power Transformation
  - Standardization

- **Performance Metrics**:
  - Accuracy: 33.3%
  - Precision: 14.3%
  - Recall: 33.3%
  - F1 Score: 20.0%
  - ROC AUC: 83.1%

### Web Application

- **Built with Django**: Robust, scalable web framework
- **Responsive UI**: Bootstrap-based responsive design
- **Interactive Visualizations**: Custom charts and graphs
- **Form Validation**: Client and server-side validation
- **REST API**: JSON-based API for programmatic access

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run migrations:
   ```bash
   python manage.py migrate
   ```

5. Start the development server:
   ```bash
   python manage.py runserver
   ```

6. Access the application at http://127.0.0.1:8000/

## API Documentation

### Endpoint: `/api/predict/`

- **Method**: POST
- **Content-Type**: application/json

### Request Parameters

| Parameter | Type   | Description                         | Required | Example |
|-----------|--------|-------------------------------------|----------|---------|
| gender    | string | Patient gender ("M" or "F")         | Yes      | "M"     |
| age       | number | Patient age in years                | Yes      | 45      |
| urea      | number | Blood urea level (mg/dL)            | Yes      | 30      |
| cr        | number | Creatinine level (mg/dL)            | Yes      | 0.9     |
| hba1c     | number | HbA1c percentage                    | Yes      | 6.0     |
| chol      | number | Total cholesterol (mg/dL)           | Yes      | 190     |
| tg        | number | Triglycerides (mg/dL)               | Yes      | 150     |
| hdl       | number | HDL cholesterol (mg/dL)             | Yes      | 45      |
| ldl       | number | LDL cholesterol (mg/dL)             | Yes      | 120     |
| vldl      | number | VLDL cholesterol (mg/dL)            | Yes      | 25      |
| bmi       | number | Body Mass Index (kg/m²)             | Yes      | 26.5    |

### Response Format

| Field          | Type    | Description                                       |
|----------------|---------|---------------------------------------------------|
| success        | boolean | Whether the request was successful                |
| prediction     | string  | Prediction result: "No", "Possible", or "Yes"     |
| probability    | string  | Formatted probability of the prediction           |
| probability_raw| number  | Raw probability value                             |
| risk_factors   | array   | List of identified risk factors                   |
| risk_details   | string  | Textual description of risk assessment            |

### Example Request

```python
import requests
import json

url = "http://127.0.0.1:8000/api/predict/"

patient_data = {
    "gender": "M",
    "age": 45,
    "urea": 30,
    "cr": 0.9,
    "hba1c": 6.0,
    "chol": 190,
    "tg": 150,
    "hdl": 45,
    "ldl": 120,
    "vldl": 25,
    "bmi": 26.5
}

response = requests.post(url, json=patient_data)
result = response.json()
print(result)
```

### Example Response

```json
{
    "success": true,
    "prediction": "Possible",
    "probability": "0.8756",
    "probability_raw": 0.8756,
    "risk_factors": [
        "Age above 45",
        "Overweight (BMI ≥ 25)",
        "HbA1c 5.7-6.4% (Prediabetes range)",
        "Elevated Triglycerides"
    ],
    "risk_details": "Moderate risk of diabetes. Several risk factors present."
}
```

## Testing

1. Run the test script to verify the API:
   ```bash
   python test_api.py
   ```

2. Run the Django tests:
   ```bash
   python manage.py test
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset source: [Reference Dataset Source]
- Inspired by research in diabetes risk prediction models 