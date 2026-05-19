import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# 1. Load the Local Dataset Manually
# ---------------------------------------------------------
# Automatically find the folder this script is saved in
script_directory = os.path.dirname(os.path.abspath(__file__))

# Look for the CSV file in that exact same folder. 
# IMPORTANT: It will look for "Iris.csv". Make sure the capitalization matches your file!
csv_path = os.path.join(script_directory, "Iris.csv")

print("Attempting to load data from:", csv_path)

try:
    # Load the dataset using pandas
    df = pd.read_csv(csv_path)
    
    # Drop 'Id' column if it exists in your CSV so it doesn't confuse the model
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    # Separate independent variables (X) and dependent target variable (y)
    X = df.iloc[:, :-1]  # Features (Measurements)
    y = df.iloc[:, -1]   # Target (Species string column)

    print("\n--- Data Loaded Successfully ---")
    print(f"First 5 rows of your input data:\n{X.head()}\n")

    # ---------------------------------------------------------
    # 2. Split the Data
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # 3. Initialize and Train the Model
    # ---------------------------------------------------------
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)  

    # ---------------------------------------------------------
    # 4. Make Predictions and Evaluate
    # ---------------------------------------------------------
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("==================================================")
    print("          MODEL PERFORMANCE EVALUATION            ")
    print("==================================================")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("==================================================\n")

    # ---------------------------------------------------------
    # 5. Interactive Custom Prediction Section
    # ---------------------------------------------------------
    print("--- Predict a Custom Flower Species ---")
    print("Enter the independent variables (measurements) below:")

    try:
        feature_list = X.columns.tolist()
        user_inputs = []
        for feature in feature_list:
            val = float(input(f"Enter {feature}: "))
            user_inputs.append(val)
        
        custom_inputs = pd.DataFrame([user_inputs], columns=feature_list)
        predicted_species = model.predict(custom_inputs)[0]
        
        print("\n==================================================")
        print(f" PROPOSED SPECIES: {predicted_species.upper()}")
        print("==================================================")

    except ValueError:
        print("\n[Error] Invalid input. Please make sure to type numbers only.")

except FileNotFoundError:
    print(f"\n[Error] Could not find the file at: {csv_path}")
    print("Please check that the CSV file name in the code matches your actual file perfectly!")