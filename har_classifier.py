import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter

# Step 1: Load the dataset
# This example assumes you are using one of the pre-processed CSV files,
# for example, 'Time_domain_subsamples.csv' or 'Frequency_features.csv'.
# Make sure to replace 'your_dataset.csv' with the actual file path.
# The raw data is also a great option but requires more complex feature engineering.
print("Loading the dataset...")
try:
    df = pd.read_csv('smartphone_activity_dataset.csv')
except FileNotFoundError:
    print("Error: The file 'smartphone_activity_dataset.csv' was not found. Please make sure to place your dataset file in the same directory as this script and rename it, or update the path.")
    exit()

print("Dataset loaded successfully.")

# New Step: Preprocessing for Raw Time-Series Data
# This section adds functionality to preprocess raw sensor data from smartphone_activity_dataset.csv.
# It assumes the raw data has columns for x, y, and z acceleration, and gyroscope readings.
# This code is generalized and might need adjustments based on the exact format of your raw data.

def butter_lowpass(cutoff, fs, order=5):
    """
    Creates a Butterworth low-pass filter.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def low_pass_filter(data, cutoff, fs, order=5):
    """
    Applies a low-pass filter to the data.
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def segment_and_extract_features(df, window_size=128, overlap=0.5):
    """
    Segments the time-series data and extracts statistical features.
    """
    features = []
    labels = []
    step_size = int(window_size * (1 - overlap))
    
    # We assume the last column is the activity label
    data = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values

    print("Segmenting data and extracting features...")
    for i in range(0, len(data) - window_size, step_size):
        window_data = data[i: i + window_size]
        window_labels = target[i: i + window_size]

        # Ensure a single label for the window
        # This is a simple majority vote; more advanced methods exist.
        label = np.bincount(window_labels).argmax()
        
        # Feature extraction (mean, standard deviation, max, min)
        # You can add more features as needed (e.g., kurtosis, skewness, signal energy)
        window_features = []
        for j in range(window_data.shape[1]):
            column_data = window_data[:, j]
            window_features.extend([
                np.mean(column_data),
                np.std(column_data),
                np.max(column_data),
                np.min(column_data),
            ])
        
        features.append(window_features)
        labels.append(label)

    print("Feature extraction complete.")
    return pd.DataFrame(features), pd.Series(labels)

# Example usage for raw data
# To use this, you must have a raw dataset with time-series sensor data.
# For example, columns like 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'Class_ID'
# Uncomment the following lines if you are working with a raw time-series dataset.

# print("Applying preprocessing pipeline for raw data...")
# # Define parameters
# fs = 50 # Example sampling frequency in Hz, adjust as needed
# cutoff = 3.667 # Example cutoff frequency, adjust as needed
# # Apply low-pass filter to each sensor axis to remove noise
# for col in df.columns[:-1]:
#     df[col] = low_pass_filter(df[col].values, cutoff, fs)
#
# # Segment data and extract features
# X_processed, y_processed = segment_and_extract_features(df, window_size=200, overlap=0.5)
# X = X_processed
# y = y_processed

# Step 2: Data Preprocessing
# The last column is typically the activity label, but this can vary.
# We'll assume the last column is the target variable (activity label) and all others are features.
# You can inspect the column names to be sure: print(df.columns)
# For the 'Time_domain_subsamples' file, the last column is 'Class_ID'.
# For the 'Frequency_features' file, the last column is also 'Class_ID'.

# Drop the last three columns, if present, which might contain metadata like length and serial number,
# as they are not features for classification.
if 'Length_of_subsample' in df.columns:
    df = df.drop(columns=['Length_of_subsample', 'Serial_no_of_subsample'])

# Split the data into features (X) and target (y)
# We assume the last column is the target variable (activity label)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Scale the features for models that are sensitive to feature scaling (like SVM and KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

print("Data split into features and target.")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Step 3: Split data into training and testing sets
# We use a 70/30 split for training and testing.
# The 'stratify' parameter ensures that the proportion of activity labels is the same in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("\nData split into training and testing sets.")
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Step 4: Train a machine learning model
# Uncomment the model you want to use.
# Note: You may need to install additional libraries (e.g., xgboost, lightgbm) if you choose those models.

# Option 1: Random Forest Classifier (current default)
print("\nTraining a Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Option 2: Support Vector Machine (SVC)
# print("\nTraining a Support Vector Classifier (SVC)...")
# model = SVC(kernel='linear', C=1.0) # You can try other kernels like 'rbf'
# model.fit(X_train, y_train)

# Option 3: K-Nearest Neighbors (KNN)
# print("\nTraining a K-Nearest Neighbors (KNN) model...")
# model = KNeighborsClassifier(n_neighbors=5) # You can experiment with different values for n_neighbors
# model.fit(X_train, y_train)

# Option 4: Gradient Boosting Classifier (e.g., XGBoost)
# import xgboost as xgb
# print("\nTraining an XGBoost Classifier...")
# model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
# model.fit(X_train, y_train)


print("Model training complete.")

# Step 5: Evaluate the model
print("\nEvaluating the model...")
y_pred = model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print a detailed classification report
# This shows precision, recall, and f1-score for each activity.
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Further steps
# For a more advanced approach, especially with raw time-series data,
# consider using a deep learning model like an LSTM (Long Short-Term Memory) network.
# LSTMs are specifically designed to handle sequential data effectively.

# Here is a conceptual snippet for an LSTM model using TensorFlow/Keras:
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
#
# # Reshape data for LSTM input (samples, timesteps, features)
# # You would need to convert your data to a 3D array first.
# # X_train_reshaped = X_train.values.reshape(-1, timesteps, features)
# # X_test_reshaped = X_test.values.reshape(-1, timesteps, features)
#
# model_lstm = Sequential()
# model_lstm.add(LSTM(64, input_shape=(timesteps, features)))
# model_lstm.add(Dropout(0.5))
# model_lstm.add(Dense(len(y.unique()), activation='softmax'))
#
# model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# print("\nTraining a conceptual LSTM model...")
# # model_lstm.fit(X_train_reshaped, y_train_one_hot, epochs=50, batch_size=64, validation_split=0.1)
# print("LSTM model conceptual code snippet. Please uncomment and adapt for use.")
