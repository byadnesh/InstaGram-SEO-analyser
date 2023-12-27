import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# List of dataset file paths
dataset_paths = ["add the dataset in csv formate"]  # Add your file paths here

# Initialize an empty DataFrame to store the combined data
combined_df = pd.DataFrame()

# Loop through the list of dataset file paths and concatenate them
for dataset_path in dataset_paths:
    df = pd.read_csv(dataset_path)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Now, combined_df contains the data from all CSV files

# Handle missing values in the 'Hashtags' column by filling them with an empty string
combined_df['Hashtags'].fillna('', inplace=True)
combined_df['Caption'].fillna('', inplace=True)

# Select features and target variable
X = combined_df[['Caption', 'Hashtags']]
y = combined_df['Views']  # Target variable

# TF-IDF vectorization for text features
tfidf = TfidfVectorizer()

# Transform caption and hashtags separately
caption_tfidf = tfidf.fit_transform(X['Caption'])
hashtags_tfidf = tfidf.fit_transform(X['Hashtags'])

# Convert the sparse matrices to dense NumPy arrays
caption_tfidf_array = caption_tfidf.toarray()
hashtags_tfidf_array = hashtags_tfidf.toarray()

# Create DataFrames for TF-IDF transformed data
caption_df = pd.DataFrame(caption_tfidf_array, columns=[f'caption_tfidf_{i}' for i in range(caption_tfidf_array.shape[1])]
                            )
hashtags_df = pd.DataFrame(hashtags_tfidf_array, columns=[f'hashtags_tfidf_{i}' for i in range(hashtags_tfidf_array.shape[1])])

# Combine the new DataFrames with the original DataFrame
X = pd.concat([caption_df, hashtags_df], axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a Machine Learning Model with optimized hyperparameters
model = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Perform k-fold cross-validation to evaluate the model
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mse = -cv_scores.mean()

# Make predictions
y_pred = model.predict(X_test_scaled)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Cross-Validation Mean Squared Error: {cv_mse}")
print(f"R-squared: {r2}")

# Visualize results (comparing predicted vs. actual views)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Views")
plt.ylabel("Predicted Views")
plt.title("Actual Views vs. Predicted Views")

# Line Graph for Actual vs. Predicted Views Over Time (assuming a time-based dataset)
# Replace 'Time' with your actual time column if applicable
if 'Time' in combined_df:
    df_sorted = combined_df.sort_values('Time')
    plt.figure(figsize=(10, 6))
    plt.plot(df_sorted['Time'], df_sorted['Views'], label='Actual Views')
    plt.plot(df_sorted['Time'], model.predict(X), label='Predicted Views')
    plt.xlabel("Time")
    plt.ylabel("Views")
    plt.title("Actual vs. Predicted Views Over Time")
    plt.legend()

# Bar Graph for Feature Importance
if hasattr(model, 'feature_importances_'):
    feature_importance = model.feature_importances_
    feature_names = X.columns
    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])  # Removed the .astype(int) and corrected line
    plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])  # Corrected line
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance')

# Provide a brief analysis
analysis_summary = f"Mean Squared Error: {mse}\nR-squared: {r2}\n"

# Save the analysis summary to a text file
with open('analysis_summary.txt', 'w') as file:
    file.write(analysis_summary)

# Show the graphs
plt.show()
