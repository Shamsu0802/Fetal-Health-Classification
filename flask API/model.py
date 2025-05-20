# Importing the libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
dataset = pd.read_csv(r'C:\Users\DELL\Desktop\fetal health\fetal_health.csv')

# Splitting the dataset into features (X) and target (y)
X = dataset.drop('fetal_health', axis=1)
y = dataset['fetal_health']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the RandomForestClassifier on the Training set
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Saving the model to disk
pickle.dump(classifier, open('model.pkl', 'wb'))
pickle.dump(sc, open('scaler.pkl', 'wb'))  # Save the scaler too

# Loading the model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('scaler.pkl', 'rb'))

# Sample data (make sure it has 21 features)
sample_data = np.array([120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(1, -1)
sample_data = sc.transform(sample_data)
print(model.predict(sample_data))