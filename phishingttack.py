import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from urllib.parse import urlparse
import re

# Sample dataset (replace with your own data)
data = {
    'URL': [
        'https://google.com',
        'http://paysecure.com',
        'https://bank-login.com',
        'https://example.com',
        'https://phishing-site.com',
        'http://legit-site.com'
    ],
    'is_phishing': [0, 1, 1, 0, 1, 0]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Function to extract features from URL
def extract_features(url):
    features = {}
    parsed_url = urlparse(url)
    
    # Feature 1: URL length
    features['url_length'] = len(url)
    
    # Feature 2: Length of the domain
    features['domain_length'] = len(parsed_url.netloc)
    
    # Feature 3: Presence of "https"
    features['has_https'] = 1 if parsed_url.scheme == 'https' else 0
    
    # Feature 4: Presence of special characters like @, -, _
    features['has_special_char'] = 1 if re.search(r'[@\-_\.]', parsed_url.netloc) else 0
    
    return features

# Extract features for each URL
feature_list = [extract_features(url) for url in df['URL']]
features_df = pd.DataFrame(feature_list)

# Add the target variable (is_phishing) to the features dataframe
features_df['is_phishing'] = df['is_phishing']

# Split data into features and target
X = features_df.drop(columns='is_phishing')
y = features_df['is_phishing']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example of predicting a new URL
new_url = ['http://example-phishing-site.com']
new_features = [extract_features(url) for url in new_url]
new_features_df = pd.DataFrame(new_features)

# Predict if the new URL is phishing
new_prediction = model.predict(new_features_df)
print("\nPrediction for new URL:", "Phishing" if new_prediction[0] == 1 else "Legit")
