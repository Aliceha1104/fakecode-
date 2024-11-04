import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

class PropertyPriceModel:
    def __init__(self):
        self.model = None
        self.suburb_encoder = LabelEncoder()
        self.proptype_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.median_suburb_code = None
        self.load_and_prepare_model()

    def load_and_prepare_model(self):
        # Load the data
        data = pd.read_csv("../dataset/house_features.csv")

        # Drop unnecessary columns
        data = data.drop(['Address', 'Postcode', 'Listing ID', 'RE Agency', 'Status'], axis=1)

        # Convert categorical columns to numerical values
        data['Suburb'] = self.suburb_encoder.fit_transform(data['Suburb'])
        data['PropType'] = self.proptype_encoder.fit_transform(data['PropType'])

        # Calculate median
        self.median_suburb_code = data['Suburb'].median()
        self.median_landsize = data['Landsize'].median()
        self.median_building_area = data['Building Area'].median()

        # Calculate property age
        data['Property Age'] = 2024 - data['Built Year']

        # Log transform the target variable
        data['Price'] = np.log(data['Price'])

        # Feature Engineering
        data['Area-to-Landsize Ratio'] = data['Building Area'] / data['Landsize']
        data['Area-to-Landsize Ratio'] = data['Area-to-Landsize Ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
        data['Total Rooms'] = data['Bedroom'] + data['Bathroom'] + data['Car-Garage']

        # Handle outliers
        data['Landsize'] = np.clip(data['Landsize'], 0, data['Landsize'].quantile(0.99))
        data['Building Area'] = np.clip(data['Building Area'], 0, data['Building Area'].quantile(0.99))

        # Feature Selection
        features = ['CBD Distance', 'Bedroom', 'Bathroom', 'Car-Garage', 'Landsize',
                    'Building Area', 'Property Age', 'Suburb', 'PropType', 'Total Rooms', 'Area-to-Landsize Ratio']
        X = data[features]
        y = data['Price']

        # Scaling for data
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Train the Random Forest Regressor model
        self.model = RandomForestRegressor(max_depth=35, max_leaf_nodes=2600, n_estimators=90, min_samples_split=5, random_state=42, max_features=0.9)
        self.model.fit(X_scaled, y)

    def predict_price(self, input_data):
        if input_data['suburb_name'] == "Other":
            suburb_code = self.median_suburb_code
        else:
            suburb_code = self.suburb_encoder.transform([input_data['suburb_name']])[0]

        landsize = input_data['landsize'] if input_data['landsize'] != 1 else self.median_landsize
        building_area = input_data['building_area'] if input_data['building_area'] != 0 else self.median_building_area
        proptype_code = self.proptype_encoder.transform([input_data['prop_type']])[0]

        # Add new features
        total_rooms = input_data['bedroom'] + input_data['bathroom'] + input_data['car_garage']
        property_age = 2024 - input_data['built_year']
        area_to_landsize = building_area / landsize

        # Prepare the input data as a DataFrame
        new_data = pd.DataFrame({
            'CBD Distance': [input_data['cbd_distance']],
            'Bedroom': [input_data['bedroom']],
            'Bathroom': [input_data['bathroom']],
            'Car-Garage': [input_data['car_garage']],
            'Landsize': [landsize],
            'Building Area': [building_area],
            'Property Age': [property_age],
            'Suburb': [suburb_code],
            'PropType': [proptype_code],
            'Total Rooms': [total_rooms],
            'Area-to-Landsize Ratio': [area_to_landsize]
        })

        # Scale the new data
        new_data_scaled = self.scaler.transform(new_data)
        # Predict the price based on user input
        predicted_price = self.model.predict(new_data_scaled)
        # Convert back from log scale
        return np.exp(predicted_price)[0]
