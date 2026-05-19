import os
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# 1. Download and Load the Dataset via Kagglehub
# ---------------------------------------------------------
# Download latest version of the dataset
path = kagglehub.dataset_download("saurabh00007/iriscsv")
print("Path to dataset files:", path)

# Find and load the CSV file from your Kaggle folder
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
csv_path = os.path.join(path, csv_file)
df = pd.read_csv(csv_path)

# Drop 'Id' column if it exists in your Kaggle file so it doesn't confuse the model
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Separate independent variables (X) and dependent target variable (y)
X = df.iloc[:, :-1]  # Features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
y = df.iloc[:, -1]   # Target: Species string column (e.g., 'Iris-setosa')

print("--- Data Loaded Successfully ---")
print(f"First 5 rows of your input data:\n{X.head()}\n")

# ---------------------------------------------------------
# 2. Split the Data
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 3. Initialize and Train the Model
# ---------------------------------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)  # Scikit-learn naturally handles the string targets from the CSV here!

# ---------------------------------------------------------
# 4. Make Predictions and Evaluate
# ---------------------------------------------------------
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("==================================================")
print("          MODEL PERFORMANCE EVALUATION            ")
print("==================================================")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report (showing Precision, Recall, & F1-Score):")
print(classification_report(y_test, predictions))
print("==================================================\n")

# ---------------------------------------------------------
# 5. Interactive Custom Prediction Section
# ---------------------------------------------------------
print("--- Predict a Custom Flower Species ---")
print("Enter the independent variables (measurements) below:")

try:
    # Use the precise column names from your Kaggle dataset
    feature_list = X.columns.tolist()
    
    user_inputs = []
    for feature in feature_list:
        val = float(input(f"Enter {feature}: "))
        user_inputs.append(val)
    
    # Format inputs into a DataFrame matching training structure exactly
    custom_inputs = pd.DataFrame([user_inputs], columns=feature_list)
    
    # Run prediction - this directly outputs the text species name
    predicted_species = model.predict(custom_inputs)[0]
    
    print("\n==================================================")
    print(f" PROPOSED SPECIES: {predicted_species.upper()}")
    print("==================================================")

except ValueError:
    print("\n[Error] Invalid input. Please make sure to type numbers only.")