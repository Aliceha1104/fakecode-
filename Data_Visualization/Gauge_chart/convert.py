import csv
import json

def convert_csv_to_jsx(csv_file_path, jsx_file_path):
    data = []

    # Read the CSV file
    with open(csv_file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            try:
                data.append({
                    "Status": row["Status"],
                    "Property_value": int(row["Property_value"]),
                })
            except KeyError as e:
                print(f"Missing key in CSV row: {e}")
                continue

    # Convert the data to a JavaScript object
    js_data = json.dumps(data, indent=2)

    # Write the data to a JSX file
    with open(jsx_file_path, mode='w') as jsx_file:
        jsx_file.write("const Property_label = ")
        jsx_file.write(js_data)
        jsx_file.write(";\n\n")
        jsx_file.write("export default Property_label;\n")

# Define the file paths
csv_file_path = 'Data_Visualization/Gauge_chart/label_data.csv'
jsx_file_path = 'Data_Visualization/Gauge_chart/label_data.jsx'

# Convert the CSV to JSX
convert_csv_to_jsx(csv_file_path, jsx_file_path)