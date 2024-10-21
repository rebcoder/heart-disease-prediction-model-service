from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Load the trained model pipeline (which includes preprocessing and model)
with open('model_pipeline.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Extract the input data from the request
    data = request.get_json()

    # Define the input features (categorical features are not one-hot encoded yet, handled by the model)
    features = [
        data['age'], data['sex'], data['chest_pain_type'],
        data['resting_blood_pressure'], data['cholesterol'],
        data['fasting_blood_sugar'], data['rest_ecg'],
        data['max_heart_rate_achieved'], data['exercise_induced_angina'],
        data['st_depression'], data['st_slope'],
        data['num_major_vessels'], data['thalassemia']
    ]
    col_names = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
                 'cholesterol', 'fasting_blood_sugar', 'rest_ecg',
                 'max_heart_rate_achieved', 'exercise_induced_angina',
                 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia']

    # Convert features to the expected format for the model (2D array)
    features_array = np.array([features]).astype(float)

    # Convert test_features into a DataFrame with the correct column names
    test_features_df = pd.DataFrame(features_array, columns=col_names)

    # Debugging: Print or log the shape and content of features_array
    print("Input feature array:", features_array)
    print("Shape of input array:", features_array.shape)

    # Run prediction using the loaded model (pipeline handles one-hot encoding and prediction)
    prediction = model_pipeline.predict(test_features_df)

    # Return the prediction result as JSON
    return jsonify({"prediction": int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=False)
