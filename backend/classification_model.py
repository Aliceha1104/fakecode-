# model.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
import math

# Suppress warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
data = pd.read_csv('../dataset/market_features.csv')
agent_le = LabelEncoder()
label_le = LabelEncoder()
data['label'] = label_le.fit_transform(data['Status'])
data['RE Agency'] = agent_le.fit_transform(data['RE Agency'])

#Calculate median
median_landsize = data['Landsize'].median()
median_price = data['Median Price'].median()
median_rental = data['Median Rental'].median()
median_agency_encoded = data['RE Agency'].median()

# Model initialization and training
features = ['Price', 'CBD Distance', 'Bedroom', 'Bathroom', 'Car-Garage', 'Landsize', 'RE Agency', 'Median Price', 'Median Rental']
x = data[features]
y = data['label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(max_depth=35, max_leaf_nodes=2600, n_estimators=90, min_samples_split=5, random_state=42, max_features=0.9)
model.fit(X_train, y_train)

def calculate_property_score(price, median_price, median_rental, sale_potential, w1=0.4, w2=0.4, w3=0.2):
    # Calculate Price Factor
    price_factor = 1 - abs((price - median_price) / median_price)

    # Calculate Rental Yield Factor
    rental_yield_factor = (median_rental * 12) / price

    # Determine Sale Potential Factor
    sale_potential_factor = 1 if sale_potential == "Sold" else 0.5  # 1 if Sold, 0.5 if On Sale

    # Calculate the unscaled Property Score
    unscaled_property_score = (w1 * price_factor) + (w2 * rental_yield_factor) + (w3 * sale_potential_factor)

    # Apply Sigmoid Function to scale Property Score
    property_score = 1 / (1 + math.exp(-unscaled_property_score))

    return property_score * 100

# Function to make a prediction based on input data
def predict_status(input_data):
    # Replace zero values with median values
    if input_data['Landsize'] == 0:
        input_data['Landsize'] = median_landsize
    if input_data['Median Price'] == 0:
        input_data['Median Price'] = median_price
    if input_data['Median Rental'] == 0:
        input_data['Median Rental'] = median_rental

   # Check if RE Agency is "Other" and assign median encoded value
    if input_data['RE Agency'] == "Other":
        input_data['RE Agency'] = median_agency_encoded  # Assign median encoded value
    else:
        # Process and encode 'RE Agency'
        input_data['RE Agency'] = agent_le.transform([input_data['RE Agency']])[0]
    # Convert to DataFrame
    new_data = pd.DataFrame([input_data])
    # Scale the input data
    new_data_scaled = scaler.transform(new_data)
    # Make prediction
    prediction = model.predict(new_data_scaled)
    # Convert prediction back to original label

    result = ''
    if label_le.inverse_transform(prediction)[0] == "S":
        result = "Sold"
    elif label_le.inverse_transform(prediction)[0] == "NS":
        result = "On Sale"

    property_score = calculate_property_score(input_data["Price"], input_data["Median Price"], input_data["Median Rental"], result)

    return round(property_score,2)
