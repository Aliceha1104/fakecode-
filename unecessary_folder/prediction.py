
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


# Function to receive new data input from the user
def get_user_input(suburb_encoder, proptype_encoder):
    cbd_distance = float(input("Enter CBD Distance (km): "))
    bedroom = int(input("Enter number of Bedrooms: "))
    bathroom = int(input("Enter number of Bathrooms: "))
    car_garage = int(input("Enter number of Car Garages: "))
    landsize = float(input("Enter Landsize (sqm): "))
    building_area = float(input("Enter Building Area (sqm): "))
    built_year = int(input("Enter Built Years: "))
    suburb_name = input("Enter Suburb Name: ")
    prop_type = input("Enter Properties Type (only one letter, ex h for house, u for unit): ")

    # Convert the suburb name and proptype to code using the encoder
    suburb_code = suburb_encoder.transform([suburb_name])[0]
    proptype_code = proptype_encoder.transform([prop_type])[0]

    #add new features
    total_rooms = bedroom + bathroom + car_garage
    property_age = 2024 - built_year
    area_to_landsize = building_area / landsize
    # Prepare the input data as a DataFrame
    new_data = pd.DataFrame({
        'CBD Distance': [cbd_distance],
        'Bedroom': [bedroom],
        'Bathroom': [bathroom],
        'Car-Garage': [car_garage],
        'Landsize': [landsize],
        'Building Area': [building_area],
        'Property Age': [property_age],
        'Suburb': [suburb_code],
        'PropType': [proptype_code],
        'Total Rooms' : [total_rooms],
        'Area-to-Landsize Ratio': [area_to_landsize]
    })

    return new_data

def main():
    # Load the data
    data = pd.read_csv("../dataset/Clean_dataset.csv")

    # Data preprocessing
    data = data.drop(['Address', 'Postcode', 'Listing ID', 'RE Agency', 'Status'], axis=1)

    # Initialize LabelEncoders
    suburb_encoder = LabelEncoder()
    proptype_encoder = LabelEncoder()

    # Convert categorical columns to numerical values
    data['Suburb'] = suburb_encoder.fit_transform(data['Suburb'])
    data['PropType'] = proptype_encoder.fit_transform(data['PropType'])

    # Calculate property age
    data['Property Age'] = 2024 - data['Built Year']

    # Log transform the target variable to reduce the effect of outliers
    data['Price'] = np.log(data['Price'])

    # ---- Feature Engineering ---- #

    # Add new feature: Building Area to Landsize Ratio
    data['Area-to-Landsize Ratio'] = data['Building Area'] / data['Landsize']
    data['Area-to-Landsize Ratio'] = data['Area-to-Landsize Ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Add new feature: Total Rooms (Bedroom + Bathroom + Car-Garage)
    data['Total Rooms'] = data['Bedroom'] + data['Bathroom'] + data['Car-Garage']

    # Handle outliers by capping Landsize and Building Area (remove extremely high values)
    data['Landsize'] = np.clip(data['Landsize'], 0, data['Landsize'].quantile(0.99))
    data['Building Area'] = np.clip(data['Building Area'], 0, data['Building Area'].quantile(0.99))

    # Data Visualization
    #plot_heatmap(data)

    # Feature Selection
    features = ['CBD Distance', 'Bedroom', 'Bathroom', 'Car-Garage', 'Landsize',
                'Building Area', 'Property Age', 'Suburb', 'PropType', 'Total Rooms', 'Area-to-Landsize Ratio']

    X = data[features]
    y = data['Price']


    # Scaling for data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=1)

    # Define the KFold object
    kf = KFold(n_splits=8, random_state=42, shuffle=True)

    # Initialize the Random Forest Regressor model
    model = RandomForestRegressor(max_depth= 35, max_leaf_nodes=2600, n_estimators=90, min_samples_split=5, random_state=42, max_features= 0.9)

    #initialize array to contain performance result
    #train data
    train_mae_scores = []
    train_mse_scores = []
    train_rmse_scores = []
    train_r2_scores = []

    #test data
    test_mae_scores = []
    test_mse_scores = []
    test_rmse_scores = []
    test_r2_scores = []

    # Loop over each split
    for train_index, test_index in kf.split(X_scaled):
        # Split the data
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the model
        model.fit(X_train, y_train)

        # Predict on the test and train data
        y_test_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        # Evaluate on the test data
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)

        # Evaluate on the training data
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)

        # Print the evaluation metrics for each fold
        # print(f"Train MAE: {train_mae}, Test MAE: {test_mae}")
        # print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")
        # print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")
        # print(f"Train R2: {train_r2}, Test R2: {test_r2}")

        #train data
        train_mae_scores.append(train_mae)
        train_mse_scores.append(train_mse)
        train_rmse_scores.append(train_rmse)
        train_r2_scores.append(train_r2)

        #test data
        test_mae_scores.append(test_mae)
        test_mse_scores.append(test_mse)
        test_rmse_scores.append(test_rmse)
        test_r2_scores.append(test_r2)

    # Calculate the average score
    #train score
    average_train_mae = sum(train_mae_scores) / len(train_mae_scores)
    average_train_mse = sum(train_mse_scores) / len(train_mse_scores)
    average_train_rmse = sum(train_rmse_scores) / len(train_rmse_scores)
    average_train_r2 = sum(train_r2_scores) / len(train_r2_scores)
    #test score
    average_test_mae = sum(test_mae_scores) / len(test_mae_scores)
    average_test_mse = sum(test_mse_scores) / len(test_mse_scores)
    average_test_rmse = sum(test_rmse_scores) / len(test_rmse_scores)
    average_test_r2 = sum(test_r2_scores) / len(test_r2_scores)

    #print average score
    #train score
    # print(f"Average Train MAE: {average_train_mae}")
    # print(f"Average Train MSE: {average_train_mse}")
    # print(f"Average Train RMSE: {average_train_rmse}")
    # print(f"Average Train R2: {average_train_r2}")
    # print("---------------------------------")
    #test score
    # print(f"Average Test MAE: {average_test_mae}")
    # print(f"Average Test MSE: {average_test_mse}")
    # print(f"Average Test RMSE: {average_test_rmse}")
    # print(f"Average Test R2: {average_test_r2}")

    # Predict user input
    new_data = get_user_input(suburb_encoder, proptype_encoder)

    # Predict and Print sample data

    # Scale the new data
    new_data_scaled = scaler.transform(new_data) #Replace X_test[0] with new_data if want to use user input

    # Predict the price based on user input
    predicted_price = model.predict(new_data_scaled)

    # Convert back from log scale
    predicted_price = np.exp(predicted_price)

    print(f"\nPredicted Price: ${predicted_price[0]:,.2f}")

if __name__ == "__main__":
    main()
