import pandas as pd
import numpy as np
import psutil
import os
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def print_memory_usage(label):
    process = psutil.Process(os.getpid())
    print(f"{label} - Memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")

# Initialize the Model
model = SGDRegressor(random_state=42)

# Load the Train Data in Chunks
chunk_size = 5000
chunks = pd.read_csv('data/Train.csv', chunksize=chunk_size)
print_memory_usage("After Initializing Model")

high_cardinality_cols = ['ID_Zindi']  # Identified previously
X_columns = None  # Placeholder for column names to use after processing the first chunk

# Incremental Training
for chunk in chunks:
    print("Processing a new chunk...")
    # Preprocessing each chunk
    chunk = chunk.ffill()
    chunk = chunk.drop(columns=['ID'])

    # Drop high cardinality columns
    chunk = chunk.drop(columns=high_cardinality_cols, errors='ignore')

    # Split into features and target
    X_chunk = chunk.drop(columns=['GT_NO2'])
    y_chunk = chunk['GT_NO2']

    # Ensure features are numeric
    X_chunk = pd.get_dummies(X_chunk)

    # Save columns from the first chunk to ensure consistent alignment for future chunks
    if X_columns is None:
        X_columns = X_chunk.columns

    # Align columns with the first chunk, filling in missing values with 0
    X_chunk = X_chunk.reindex(columns=X_columns, fill_value=0)

    # Fill any NaN values after reindexing
    X_chunk = X_chunk.fillna(0)

    # Incrementally train the model
    model.partial_fit(X_chunk, y_chunk)
    #print_memory_usage("After Training on Chunk")

# Evaluate Model
# (Load a validation set and evaluate)
val_data = pd.read_csv('data/Train.csv', nrows=1000)  # Load a smaller portion for validation
val_data = val_data.ffill().drop(columns=['ID'])
val_data = val_data.drop(columns=high_cardinality_cols, errors='ignore')
X_val = val_data.drop(columns=['GT_NO2'])
y_val = val_data['GT_NO2']
X_val = pd.get_dummies(X_val)
X_val = X_val.reindex(columns=X_columns, fill_value=0)
X_val = X_val.fillna(0)

y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)

print(f'Validation RMSE: {rmse}, MAE: {mae}')

# Load the Test Data
test_data = pd.read_csv('data/Test.csv')
print("Loaded test data")
print_memory_usage("After Loading Test Data")

# Data Preprocessing for Test Data
test_data = test_data.ffill()
test_data_ids = test_data['ID']  # Save 'ID' column for the submission
test_data = test_data.drop(columns=['ID'])
test_data = test_data.drop(columns=high_cardinality_cols, errors='ignore')
X_test = pd.get_dummies(test_data)
X_test = X_test.reindex(columns=X_columns, fill_value=0)
X_test = X_test.fillna(0)

# Predict on Test Data
y_test_pred = model.predict(X_test)

# Create Submission File
submission = pd.DataFrame({'ID': test_data_ids, 'GT_NO2': y_test_pred})
submission.to_csv('result/SampleSubmission.csv', index=False)


# data cleaning + processing 
# how to use Neural networks
# why use algo and not other (exemple sigmoid not relu..)