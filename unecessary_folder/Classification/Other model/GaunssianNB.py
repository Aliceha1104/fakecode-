import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# from data_visualisation import plot_data, qq_plot_data, plot_heatmap

def main():
    # Load and preprocess data
    data = pd.read_csv('../../dataset/market_features.csv')

    # Data Preprocessing
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['Status'])
    data['RE Agency'] = le.fit_transform(data['RE Agency'])
    data['PropType'] = le.fit_transform(data['PropType'])
    data['Property Age'] = 2024 - data['Built Year']  # Assuming data is from 2024

    # Feature and target
    features = ['PropType', 'Price', 'CBD Distance', 'Bedroom', 'Bathroom', 'Car-Garage', 'Landsize', 'Building Area', 'Property Age', 'Median Price', 'Median Rental', 'PropertyCount', 'RE Agency']
    x = data[features]
    y = data['label']

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    # Data Visualization

    # Convert X_scaled to DataFrame and add y
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)
    X_scaled_df['label'] = y.values
    # plot_heatmap(X_scaled_df)

    # plot_data(X_scaled_df)
    # qq_plot_data(X_scaled_df)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    test_accuracy = round(accuracy_score(y_test, y_test_pred), 4)
    test_precision = round(precision_score(y_test, y_test_pred, average='weighted'), 4)
    test_recall = round(recall_score(y_test, y_test_pred, average='weighted'), 4)
    test_f1 = round(f1_score(y_test, y_test_pred, average='weighted'), 4)

    train_accuracy = round(accuracy_score(y_train, y_train_pred), 4)
    train_precision = round(precision_score(y_train, y_train_pred, average='weighted'), 4)
    train_recall = round(recall_score(y_train, y_train_pred, average='weighted'), 4)
    train_f1 = round(f1_score(y_train, y_train_pred, average='weighted'), 4)

    print(f'Test Accuracy: {test_accuracy}  | Train Accuracy: {train_accuracy}')
    print(f'Test Precision: {test_precision}| Test Precision: {train_precision}')
    print(f'Test Recall: {test_recall}      | Train Recall: {train_recall}')
    print(f'Test F1 Score: {test_f1}        | Train F1 Score: {train_f1}')

    # Sample Output
    # print("Sample Output")
    # print(data.iloc[151])

    # predicted = best_model.predict(X_train[151].reshape(1, -1))
    # print(f"Predicted: {predicted}")

if __name__ == '__main__':
    main()