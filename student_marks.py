import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = load_model('best_advanced_study_model.keras')

# Initialize the scaler (reuse the same scaler that was fit during training)
scaler = StandardScaler()

# Define the input schema using Pydantic
class StudyData(BaseModel):
    number_courses: int
    time_study: float

# Root endpoint to check if the API is running
@app.get("/")
def read_root():
    return {"message": "Welcome to the Study Prediction API!"}

# Prediction endpoint
@app.post("/predict/")
def predict(data: StudyData):
    try:
        # Create a DataFrame from the input data
        input_data = pd.DataFrame([data.dict()])

        # Generate interaction and polynomial features
        input_data['interaction'] = (
            input_data['number_courses'] * input_data['time_study']
        )
        input_data['number_courses_squared'] = input_data['number_courses'] ** 2
        input_data['time_study_squared'] = input_data['time_study'] ** 2

        # Standardize the input features
        input_scaled = scaler.fit_transform(input_data)

        # Reshape input for LSTM model
        input_reshaped = np.reshape(input_scaled, (1, 1, input_scaled.shape[1]))

        # Make prediction
        prediction = model.predict(input_reshaped)

        # Return the result
        return {"Predicted Marks": float(prediction[0][0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Run the API using: uvicorn filename:app --reload
