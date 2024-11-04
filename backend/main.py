import sys
sys.dont_write_bytecode = True
import json
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request  # Import Request
from fastapi.responses import JSONResponse  # Import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from prediction_model import PropertyPriceModel  # Import the prediction model
from classification_model import predict_status
import numpy as np  # Import numpy
import pandas as pd  # Import pandas
from contextlib import asynccontextmanager  # Import asynccontextmanager
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Import LabelEncoder and StandardScaler
from sklearn.cluster import KMeans  # Import KMeans

# Initialize FastAPI app globally
app = FastAPI()

# CORS configuration to allow requests from http://localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify the frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define Pydantic model for input validation
class PropertyData(BaseModel):
    cbd_distance: float = None
    bedroom: int = None
    bathroom: int = None
    car_garage: int = None
    landsize: float = None
    building_area: float = None
    built_year: int = 2024
    suburb_name: str = 'Other'
    prop_type: str = 'u'

class HouseStatus(BaseModel):
    price: float
    cbd_distance: float
    bedroom: int
    bathroom: int
    car_garage: int
    landsize: float
    re_agency: str
    median_price: float
    median_rental: int

# Initialize the model
property_price_model = PropertyPriceModel()

#db files
PREDICTION_FILE = "predictions.json"
SALES_PREDICTION_FILE = "sale_predictions.json"

# Function to save predictions to a JSON file
def save_prediction_to_json(input_data, predicted_price):
    if os.path.exists(PREDICTION_FILE):
        try:
            with open(PREDICTION_FILE, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []  # Initialize as empty list if file is corrupted or empty
    else:
        data = []

    # Determine ID for each record
    if data:
        last_id = max([entry["id"] for entry in data])
        new_id = last_id + 1
    else:
        new_id = 1

    # Append new prediction data
    new_entry = {
        "id": new_id,
        "house_data": input_data,
        "predicted_price": round(predicted_price, 2)
    }
    data.append(new_entry)

    # Write the updated data back to the JSON file
    with open(PREDICTION_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Function to delete a prediction by ID from the JSON file and reassign IDs
def delete_prediction_from_json(prediction_id: int):
    try:
        if os.path.exists(PREDICTION_FILE):
            with open(PREDICTION_FILE, "r") as f:
                data = json.load(f)
            # Check if the prediction exists
            updated_data = [entry for entry in data if entry["id"] != prediction_id]

            if len(updated_data) == len(data):
                raise HTTPException(status_code=404, detail="Prediction not found")
            # Reassign the IDs to maintain sequential order
            for i, entry in enumerate(updated_data, start=1):
                entry["id"] = i

            # Write the updated data back to the JSON file
            with open(PREDICTION_FILE, "w") as f:
                json.dump(updated_data, f, indent=4)
        else:
            raise HTTPException(status_code=404, detail="Prediction file not found.")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Prediction file not found.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error decoding JSON data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting the prediction: {str(e)}")

# Function to load all stored predictions from the JSON file
def load_predictions_from_json(file):
    try:
        if os.path.exists(file):
            with open(file, "r") as f:
                return json.load(f)
        else:
            return []
    except FileNotFoundError:  # Handle case where file not found
        raise HTTPException(status_code=404, detail="Prediction file not found.")
    except json.JSONDecodeError:  # Handle case where JSON is corrupted
        raise HTTPException(status_code=500, detail="Error decoding JSON data.")
    except Exception as e:
        # General error handling
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

#Function to save sale predictions
# Function to save sale predictions
def save_sale_prediction_to_json(predicted_result, predicted_status, price, median_price, median_rental):
    if os.path.exists(SALES_PREDICTION_FILE):
        try:
            with open(SALES_PREDICTION_FILE, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []  # Initialize as empty list if file is corrupted or empty
    else:
        data = []

    # Determine ID for each record
    if data:
        last_id = max(entry["id"] for entry in data)
        new_id = last_id + 1
    else:
        new_id = 1

    # Append new prediction data
    new_entry = {
        "id": new_id,
        "predicted_result": predicted_result,
        "predicted_status": predicted_status,
        "price": round(price, 2),
        "median_price": round(median_price, 2),
        "median_rental": median_rental
    }
    data.append(new_entry)

    # Write the updated data back to the JSON file
    with open(SALES_PREDICTION_FILE, "w") as f:
        json.dump(data, f, indent=4)

#House price Prediction Endpoints
# Endpoint for predicting house prices
@app.post("/predict")
async def predict_price(property_data: PropertyData):
    input_data = property_data.dict()
    predicted_price = property_price_model.predict_price(input_data)
    save_prediction_to_json(input_data, predicted_price)
    return {"predicted_price": round(predicted_price, 2)}

# Endpoint for fetching all predictions
@app.get("/prediction-history/")
async def get_predictions():
    predictions = load_predictions_from_json(PREDICTION_FILE)
    return {"predictions": predictions}

# Endpoint for deleting a prediction by ID
@app.delete("/delete-prediction/{prediction_id}")
async def delete_prediction(prediction_id: int):
    delete_prediction_from_json(prediction_id)
    return {"message": f"Prediction with ID {prediction_id} has been deleted."}


#House Potential Sell Predict Endpoint
@app.post("/predict-sale-potential")
async def predict_sale_potential(data: HouseStatus):
    input_data = {
        'Price': data.price,
        'CBD Distance': data.cbd_distance,
        'Bedroom': data.bedroom,
        'Bathroom': data.bathroom,
        'Car-Garage': data.car_garage,
        'Landsize': data.landsize,
        'RE Agency': data.re_agency,
        'Median Price': data.median_price,
        'Median Rental': data.median_rental
    }
    # Get the prediction result
    predicted_result = predict_status(input_data)
    if predicted_result < 40:
        predicted_status = "Bad"
    elif predicted_result >= 40 and predicted_result < 80:
        predicted_status = "Average"
    elif predicted_result >= 80:
        predicted_status = "Good"
    save_sale_prediction_to_json(predicted_result, predicted_status, input_data['Price'], input_data['Median Price'], input_data['Median Rental'])
    return {"predicted_status": predicted_status, "predicted_result": predicted_result}

@app.get("/sale-prediction-history/")
async def get_sale_predictions():
    sale_predictions = load_predictions_from_json(SALES_PREDICTION_FILE)
    return {"predictions": sale_predictions}

# Endpoint for deleting a prediction by ID
@app.delete("/delete-sale-prediction/{prediction_id}")
async def delete_prediction(prediction_id: int):
    delete_prediction_from_json(prediction_id)
    return {"message": f"Prediction with ID {prediction_id} has been deleted."}

# Custom 404 error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Entry point for running the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)