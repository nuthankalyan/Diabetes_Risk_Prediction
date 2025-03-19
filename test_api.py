import requests
import json
import sys

def test_diabetes_prediction_api():
    """
    Test the diabetes prediction API endpoint with various test cases.
    """
    # API endpoint URL (local development server)
    url = "http://127.0.0.1:8000/api/predict/"
    
    # Test case 1: High risk patient
    high_risk_patient = {
        "gender": "M",
        "age": 65,
        "urea": 45,
        "cr": 1.4,
        "hba1c": 8.2,
        "chol": 240,
        "tg": 220,
        "hdl": 32,
        "ldl": 180,
        "vldl": 35,
        "bmi": 31.5
    }
    
    # Test case 2: Moderate risk patient
    moderate_risk_patient = {
        "gender": "F",
        "age": 52,
        "urea": 35,
        "cr": 0.9,
        "hba1c": 6.2,
        "chol": 205,
        "tg": 160,
        "hdl": 38,
        "ldl": 130,
        "vldl": 28,
        "bmi": 27.8
    }
    
    # Test case 3: Low risk patient
    low_risk_patient = {
        "gender": "F",
        "age": 35,
        "urea": 22,
        "cr": 0.7,
        "hba1c": 5.2,
        "chol": 180,
        "tg": 110,
        "hdl": 55,
        "ldl": 95,
        "vldl": 20,
        "bmi": 22.5
    }
    
    test_cases = {
        "High Risk Patient": high_risk_patient,
        "Moderate Risk Patient": moderate_risk_patient,
        "Low Risk Patient": low_risk_patient
    }
    
    # Run all test cases
    success_count = 0
    for name, patient_data in test_cases.items():
        print(f"\n===== Testing {name} =====")
        print(f"Input data: {json.dumps(patient_data, indent=2)}")
        
        try:
            # Make the API request
            response = requests.post(url, json=patient_data, timeout=10)
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                print(f"Status Code: {response.status_code}")
                print(f"Response: {json.dumps(result, indent=2)}")
                
                # Check if the API returned the expected fields
                expected_fields = ['success', 'prediction', 'probability', 'probability_raw', 'risk_factors', 'risk_details']
                missing_fields = [field for field in expected_fields if field not in result]
                
                if missing_fields:
                    print(f"WARNING: Missing expected fields in response: {missing_fields}")
                else:
                    print("SUCCESS: All expected fields present in response")
                    success_count += 1
            else:
                print(f"ERROR: Received status code {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("ERROR: Could not connect to the server. Make sure the Django server is running.")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: An unexpected error occurred: {str(e)}")
    
    # Print summary
    print("\n===== Test Summary =====")
    print(f"Successful tests: {success_count}/{len(test_cases)}")
    if success_count == len(test_cases):
        print("All tests passed successfully!")
    else:
        print(f"WARNING: {len(test_cases) - success_count} tests failed.")

if __name__ == "__main__":
    print("Testing the Diabetes Prediction API...")
    test_diabetes_prediction_api() 