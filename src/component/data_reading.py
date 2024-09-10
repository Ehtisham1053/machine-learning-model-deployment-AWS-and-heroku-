# save_model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

# Ensure that the artifacts folder exists
artifacts_dir = 'artifacts'
if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)

# Load the dataset
df = pd.read_csv(r'dataset\magic04.data')
df.columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
df['class'].replace({'g': 0, 'h': 1}, inplace=True)

# Split the data into features and target
x = df.drop('class', axis=1)
y = df['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Data transformation (scaling and oversampling)
def preprocess_data(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Oversample the training data
    ros = RandomOverSampler()
    x_train_resampled, y_train_resampled = ros.fit_resample(x_train_scaled, y_train)

    # Save the preprocessor (scaler and resampler) in the artifacts folder
    with open(os.path.join(artifacts_dir, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump({'scaler': scaler, 'ros': ros}, f)

    return x_train_resampled, x_test_scaled, y_train_resampled

# Preprocess the data
x_train_resampled, x_test_scaled, y_train_resampled = preprocess_data(x_train, x_test)

# Build and train the RandomForest model
model = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200)
model.fit(x_train_resampled, y_train_resampled)

# Save the trained model in the artifacts folder
with open(os.path.join(artifacts_dir, 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)

# Evaluate the model on the test data
y_pred = model.predict(x_test_scaled)
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 2))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Print confusion matrix values
tn, fp, fn, tp = cm.ravel()
print(f'True Positive: {tp}')
print(f'True Negative: {tn}')
print(f'False Positive: {fp}')
print(f'False Negative: {fn}')
