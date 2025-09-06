import pandas as pd
import numpy as np


import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Load the dataset
df = pd.read_csv('data/loan_data.csv')


#handle missing values
df.fillna(df.mode().iloc[0], inplace=True)

#Label Encoding
from sklearn.preprocessing import LabelEncoder  
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


df1=df['loan_status'].value_counts()
print(df1)


# Features and Target variable
X = df.drop("loan_status", axis=1)
y = df["loan_status"]


#SMOTE for balancing the dataset
smote = SMOTE(random_state=42)
transform_feature,transform_label = smote.fit_resample(X, y)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(transform_feature,transform_label, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. Train Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 10. Evaluate the model
y_pred = model.predict(X_test)  

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", model.score(X_test, y_test))


# Save the model and scaler
with open('model/customer_mall_model.pkl', 'wb') as file:
    pickle.dump((scaler, model), file)

print("Model and scaler saved to 'model/customer_mall_model.pkl")



