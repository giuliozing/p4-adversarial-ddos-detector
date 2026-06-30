import pandas as pd
import hickle as hkl
from sklearn.model_selection import train_test_split
import numpy as np

print("Script started")
# Path to the input CSV file
csv_file_path = "../Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"


    # Read the CSV file and extract the required columns
NUM_CHUNKS = 1
for i in range(NUM_CHUNKS):
    data = pd.read_csv(csv_file_path, usecols=[20, 33, 30, 28, 61]) # ack seq len flags label
    print("CSV file read successfully")

    # Process all rows and columns
    rows_to_delete = []
    for index, row in data.iterrows():
        if index % 1000 == 0:
            print(f"Processing row {index}")
        try:
            for col in data.columns:
                value = row[col]
                # Check if the value is a float, int, or string representing a float
                if isinstance(value, (float, int)):
                    # Convert to int
                    data.at[index, col] = int(value)
                elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                    # Convert string representing a float to int
                    data.at[index, col] = int(float(value))
                else:
                    # If it's neither, mark the row for deletion
                    print(f"Invalid value in column {col}: {value}")
                    rows_to_delete.append(index)
                    break  # No need to check other columns for this row
        except ValueError:
            # If any error occurs during processing, mark the row for deletion
            print(f"Error processing value in row {index}")
            rows_to_delete.append(index)

    # Drop rows with invalid values
    data.drop(index=rows_to_delete, inplace=True)
    print("Rows to delete identified")
    # Reset index after deletion
    data.reset_index(drop=True, inplace=True)
    X = data.iloc[:, :4]
    y = data.iloc[:, 4]
    print("y:", y[:10])
    print("Data split into features and labels")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)
    X_test, X_attack, y_test, y_attack = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, stratify=y_test)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_attack = np.array(X_attack)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_attack = np.array(y_attack)


    print("Train size: {}".format(np.unique(y_train, return_counts=True)))

    print("Test size: {}".format(np.unique(y_test, return_counts=True)))


    # save to file
    data = {'xtrain': X_train, 'ytrain': y_train, 'xtest':X_test,'ytest':y_test, 'xattack':X_attack, 'yattack':y_attack}
    # Save the extracted data to an hickle file
    output_file_path = "Edge-IIoT_Chi_Square4NEW"+ str(i)+ ".hkl"
    hkl.dump(data, output_file_path)

    print(f"Data successfully saved to {output_file_path}")