# Enhanced model training script with cross-validation, model comparison, and feature importance
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Create directory for model artifacts if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load the dataset
print("Loading dataset...")
crop = pd.read_csv('Crop_recommendation.csv')

# Map the 'label' column to numerical values
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
    'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
    'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14,
    'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
    'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}

# Create a reverse mapping for later use
reverse_crop_dict = {v: k for k, v in crop_dict.items()}

# Map labels to numeric values
crop['label_num'] = crop['label'].map(crop_dict)

# Exploratory Data Analysis
print("Performing exploratory data analysis...")

# Count of each crop type
plt.figure(figsize=(12, 6))
sns.countplot(x='label', data=crop, order=crop['label'].value_counts().index)
plt.title('Count of Each Crop Type')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('models/crop_distribution.png')

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr_matrix = crop.drop('label', axis=1).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('models/correlation_heatmap.png')

# Split data into features (X) and target (y)
X = crop.drop(['label', 'label_num'], axis=1)
y = crop['label_num']

# Save feature names for later
feature_names = X.columns.tolist()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
print("Preprocessing data...")
# MinMax scaling
mx = MinMaxScaler()
X_train_mx = mx.fit_transform(X_train)
X_test_mx = mx.transform(X_test)

# Standard scaling (applied after MinMax)
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train_mx)
X_test_scaled = sc.transform(X_test_mx)

# Model training and evaluation
print("Training and evaluating models...")

# 1. Gaussian Naive Bayes (Original model)
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
y_pred_gnb = gnb.predict(X_test_scaled)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print(f'GaussianNB Accuracy: {accuracy_gnb:.4f}')

# 2. Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'RandomForest Accuracy: {accuracy_rf:.4f}')

# 3. Support Vector Machine
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'SVM Accuracy: {accuracy_svm:.4f}')

# Cross-validation
print("Performing cross-validation...")
cv_scores_gnb = cross_val_score(gnb, X_train_scaled, y_train, cv=5)
cv_scores_rf = cross_val_score(rf, X_train_scaled, y_train, cv=5)
cv_scores_svm = cross_val_score(svm, X_train_scaled, y_train, cv=5)

print(f'GaussianNB CV Accuracy: {cv_scores_gnb.mean():.4f} ± {cv_scores_gnb.std():.4f}')
print(f'RandomForest CV Accuracy: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}')
print(f'SVM CV Accuracy: {cv_scores_svm.mean():.4f} ± {cv_scores_svm.std():.4f}')

# Determine best model based on cross-validation
best_model_name = ""
best_model = None
best_cv_score = 0

if cv_scores_gnb.mean() > best_cv_score:
    best_cv_score = cv_scores_gnb.mean()
    best_model = gnb
    best_model_name = "GaussianNB"

if cv_scores_rf.mean() > best_cv_score:
    best_cv_score = cv_scores_rf.mean()
    best_model = rf
    best_model_name = "RandomForest"

if cv_scores_svm.mean() > best_cv_score:
    best_cv_score = cv_scores_svm.mean()
    best_model = svm
    best_model_name = "SVM"

print(f'Best model: {best_model_name} with CV accuracy: {best_cv_score:.4f}')

# If RandomForest is best, get feature importance
if best_model_name == "RandomForest":
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance for Crop Recommendation')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')

# Generate detailed classification report for best model
y_pred_best = best_model.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=[reverse_crop_dict[i] for i in sorted(reverse_crop_dict.keys())]))

# Confusion Matrix for best model
plt.figure(figsize=(16, 12))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[reverse_crop_dict[i] for i in sorted(reverse_crop_dict.keys())],
            yticklabels=[reverse_crop_dict[i] for i in sorted(reverse_crop_dict.keys())])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('models/confusion_matrix.png')

# Save all models and scalers
print("Saving models and scalers...")
pickle.dump(best_model, open('model.pkl', 'wb'))
pickle.dump(mx, open('minmaxscaler.pkl', 'wb'))
pickle.dump(sc, open('standscaler.pkl', 'wb'))

# Also save models with descriptive names
pickle.dump(gnb, open('models/gaussian_nb_model.pkl', 'wb'))
pickle.dump(rf, open('models/random_forest_model.pkl', 'wb'))
pickle.dump(svm, open('models/svm_model.pkl', 'wb'))

# Save metadata about the models
model_info = {
    'gnb_accuracy': accuracy_gnb,
    'rf_accuracy': accuracy_rf,
    'svm_accuracy': accuracy_svm,
    'gnb_cv_accuracy': cv_scores_gnb.mean(),
    'rf_cv_accuracy': cv_scores_rf.mean(),
    'svm_cv_accuracy': cv_scores_svm.mean(),
    'best_model': best_model_name,
    'feature_names': feature_names,
    'crop_mapping': crop_dict,
    'reverse_crop_mapping': reverse_crop_dict
}

with open('models/model_metadata.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print("Model training and evaluation complete!")