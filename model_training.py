import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('manual_pose_data.csv')  # Ensure this file contains both Working and Slacking
X = data.iloc[:, 1:]  # Features (pose landmarks)
y = data.iloc[:, 0]   # Labels (Working, Slacking)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model
import pickle
with open('pose_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)
