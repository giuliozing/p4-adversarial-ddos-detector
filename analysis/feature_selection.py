import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

print("Script started")
# Path to the input CSV file
csv_file_path = "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
# Non consideriamo:
# - ip address
# - timestamp
# - mqtt message protoname e topic
# - tcp payload e option
# - arp src e dst
# - http method data request uri
data = pd.read_csv(csv_file_path, usecols=[4,5,7,8,9,10,12,16,17,18,19,20, 21,22,23,24,25,26,27,28,29,30,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,52,53,56, 57,58,59,60, 61])
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


X = data.iloc[:, :46].values
y = data.iloc[:, 46].values
print("X:", X[:30, :])
print("Data filtered and split into features and labels")
print(X.shape, y.shape)
print("Fitting SelectKBest")
selettore = SelectKBest(chi2, k="all")
selettore.fit(X, y)
print("Feature selection completed")
print(selettore.scores_)
