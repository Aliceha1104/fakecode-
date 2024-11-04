import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import warnings

#from data_visualisation import plot_data, plot_heatmap

# Function to receive new data input from the user
def get_user_input(re_agency_encoder):
    price = float(input("Enter House Price: "))
    cbd_distance = float(input("Enter CBD Distance (km): "))
    bedroom = int(input("Enter number of Bedrooms: "))
    bathroom = int(input("Enter number of Bathrooms: "))
    car_garage = int(input("Enter number of Car Garages: "))
    landsize = float(input("Enter Landsize (sqm): "))
    re_agency = input("Enter Real Estate Agency: ")
    median_price = float(input("Enter Median Price: "))
    median_rental = int(input("Enter Median Rental: "))

    # Normalize the input data
    re_agency = re_agency_encoder.transform([re_agency])[0]

    # Prepare the input data as a DataFrame
    new_data = pd.DataFrame({
        'Price': [price],
        'CBD Distance': [cbd_distance],
        'Bedroom': [bedroom],
        'Bathroom': [bathroom],
        'Car-Garage': [car_garage],
        'Landsize': [landsize],
        'RE Agency': [re_agency],
        'Median Price': [median_price],
        'Median Rental': [median_rental]
    })

    return new_data

def main():
    warnings.filterwarnings('ignore') # Ignore warnings when display the output

    # Load and preprocess data
    data = pd.read_csv('dataset/market_features.csv')

    # Data Preprocessing
    agent_le = LabelEncoder()
    label_le = LabelEncoder()
    data['label'] = label_le.fit_transform(data['Status'])
    data['RE Agency'] = agent_le.fit_transform(data['RE Agency'])

    # Feature and target
    features = ['Price', 'CBD Distance', 'Bedroom', 'Bathroom', 'Car-Garage', 'Landsize', 'RE Agency', 'Median Price', 'Median Rental']
    x = data[features]
    y = data['label']

    # Data Visualization
    # plot_data(data)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    # Convert X_scaled to DataFrame and add y
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)
    X_scaled_df['label'] = y.values

    # plot_heatmap(X_scaled_df)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


    # Initialize the model
    model = RandomForestClassifier(max_depth= 35, max_leaf_nodes=2600, n_estimators=90, min_samples_split=5, random_state=42, max_features= 0.9)

    # Define the KFold object
    kf = KFold(n_splits=8, random_state=42, shuffle=True)

    # Initialize lists to store metrics
    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_f1s = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1s = []

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

        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')

        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='weighted')
        train_recall = recall_score(y_train, y_train_pred, average='weighted')
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')

        # print(f'Test Accuracy: {test_accuracy}  | Train Accuracy: {train_accuracy}')
        # print(f'Test Precision: {test_precision}| Test Precision: {train_precision}')
        # print(f'Test Recall: {test_recall}      | Train Recall: {train_recall}')
        # print(f'Test F1 Score: {test_f1}        | Train F1 Score: {train_f1}')

        # Append metrics to lists
        train_accuracies.append(train_accuracy)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)
        test_accuracies.append(test_accuracy)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1s.append(test_f1)

    # Calculate and print average metrics
    print("\nAverage Metrics:")
    print(f"Average Train Accuracy: {sum(train_accuracies) / len(train_accuracies):.4f} | Average Test Accuracy: {sum(test_accuracies) / len(test_accuracies):.4f}")
    print(f"Average Train Precision: {sum(train_precisions) / len(train_precisions):.4f} | Average Test Precision: {sum(test_precisions) / len(test_precisions):.4f}")
    print(f"Average Train Recall: {sum(train_recalls) / len(train_recalls):.4f} | Average Test Recall: {sum(test_recalls) / len(test_recalls):.4f}")
    print(f"Average Train F1 Score: {sum(train_f1s) / len(train_f1s):.4f} | Average Test F1 Score: {sum(test_f1s) / len(test_f1s):.4f}")

    # Sample Output
    # print("Sample Output")
    # print(data.iloc[0])

    # Get user input
    # new_data = get_user_input(agent_le)

    predicted = model.predict(X_train[0].reshape(1, -1)) # Predict the first row of the training data
    # predicted = model.predict(new_data) # Predict the new data

    # Convert the predicted label back to the original label
    predicted = label_le.inverse_transform(predicted)

    print(f"Predicted: {predicted}")

if __name__ == '__main__':
    main()