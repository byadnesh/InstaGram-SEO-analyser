from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# Load your Instagram dataset
data = pd.read_csv('manojsaru_instagram_posts_LIKES.csv')

# Split the data
X = data[['Views', 'Likes', 'Comments', 'Time']]
y = data['Views']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom Transformer for Time feature
class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure 'Time' column is present
        if 'Time' in X.columns:
            # You can add additional time-related features if needed
            X['Hour'] = pd.to_datetime(X['Time']).dt.hour
            X['Minute'] = pd.to_datetime(X['Time']).dt.minute
            return X[['Hour', 'Minute']]
        else:
            raise KeyError("Column 'Time' not found in the DataFrame.")

# Check columns before modifications
print("Before Modifications:")
print(X_train.columns)
print(X_test.columns)

# Modify the 'Time' column
time_extractor = TimeFeatureExtractor()
X_train_time = time_extractor.transform(X_train)
X_test_time = time_extractor.transform(X_test)

# Update 'Time' and add 'Hour' and 'Minute' columns
X_train[['Hour', 'Minute']] = X_train_time[['Hour', 'Minute']]
X_test[['Hour', 'Minute']] = X_test_time[['Hour', 'Minute']]

# Drop the 'Time' column if it exists
X_train = X_train.drop('Time', axis=1, errors='ignore')
X_test = X_test.drop('Time', axis=1, errors='ignore')

# Check columns after modifications
print("\nAfter Modifications:")
print(X_train.columns)
print(X_test.columns)

# Rest of the code remains the same...



# Rest of the code remains the same...

# Machine Learning Pipeline
engagement_pipeline = Pipeline([
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Views', 'Likes', 'Comments']),
            ('time', TimeFeatureExtractor(), ['Hour'])
        ]
    )),
    ('regressor', HistGradientBoostingRegressor())
])

# Impute NaN values in the target variable
imputer = SimpleImputer(strategy='mean')
y_train_imputed = pd.DataFrame(imputer.fit_transform(y_train.values.reshape(-1, 1)), columns=['Views'])

# Train the model
engagement_pipeline.fit(X_train, y_train_imputed)

# ... (rest of the code) ...

# Evaluate the model on the imputed test set
# Impute NaN values in the test set
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Impute NaN values in the target variable for predictions
y_test_imputed = pd.DataFrame(imputer.transform(y_test.values.reshape(-1, 1)), columns=['Views'])

# Use the trained model to make predictions on the imputed test set
predictions = engagement_pipeline.predict(X_test_imputed)

# Evaluate the model
mse = mean_squared_error(y_test_imputed, predictions)
print(f'Mean Squared Error: {mse}')

# Visualize predicted vs actual engagement
plt.scatter(y_test_imputed, predictions, alpha=0.5)
plt.xlabel('Actual Engagement')
plt.ylabel('Predicted Engagement')
plt.title('Engagement Predictor Performance')
plt.show()

# ... Rest of the code remains the same ...

# Caption Generator
data['Generated_Caption'] = data['Caption'].apply(lambda x: str(TextBlob(x).sentiment))

# Hashtag Recommender
tfidf_vectorizer = TfidfVectorizer(max_features=10)
hashtags_tfidf = tfidf_vectorizer.fit_transform(data['Hashtags'])
data['Recommended_Hashtags'] = hashtags_tfidf.toarray().tolist()

# Content Scheduler
# In a real scenario, analyze the best time for posting based on historical data

# Sentiment Analysis
data['Comment_Sentiment'] = data['Comments'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Create a PDF report
def create_pdf_report(data, filename='Instagram_Report.pdf'):
    # Create a PDF using ReportLab
    with canvas.Canvas(filename, pagesize=letter) as canvas_obj:
        canvas_obj.drawString(72, 800, 'Instagram Analytics Report')

        # Add more content to the PDF here...
        canvas_obj.drawString(72, 780, 'Caption Generator Results:')
        for index, row in data.iterrows():
            canvas_obj.drawString(72, 760 - index * 20, f"Post {index + 1}: {row['Generated_Caption']}")

        canvas_obj.drawString(72, 500, 'Hashtag Recommender Results:')
        for index, row in data.iterrows():
            canvas_obj.drawString(72, 480 - index * 20, f"Post {index + 1}: {row['Recommended_Hashtags']}")

        canvas_obj.drawString(72, 300, 'Sentiment Analysis Results:')
        for index, row in data.iterrows():
            canvas_obj.drawString(72, 280 - index * 20, f"Post {index + 1}: Sentiment - {row['Comment_Sentiment']}")

    print(f'Report saved as {filename}')

# Call the PDF creation function
create_pdf_report(data)
