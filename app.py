"""import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# 1. Set up the UI Header
# ---------------------------------------------------------
st.set_page_config(page_title="Iris Classifier", page_icon="🌸")
st.title("🌸 Iris Flower Classifier")
st.write("Adjust the sliders on the left to instantly predict the species of an Iris flower!")

# ---------------------------------------------------------
# 2. Load Data, Train Model, & Evaluate (Cached)
# ---------------------------------------------------------
# We cache this so the model only trains once when the app boots up
@st.cache_data
def load_train_and_evaluate():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_directory, "Iris.csv")
    
    df = pd.read_csv(csv_path)
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    
    X = df.iloc[:, :-1]  
    y = df.iloc[:, -1]   
    
    # Split the data exactly like we did in the terminal version
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate performance metrics
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    return model, X.columns.tolist(), accuracy, report

try:
    # Unpack all our returned values, including the accuracy and report
    model, feature_list, accuracy, report = load_train_and_evaluate()
    
    # ---------------------------------------------------------
    # 3. Display Model Performance Metrics
    # ---------------------------------------------------------
    # This creates a collapsible dropdown box so it doesn't clutter the screen
    with st.expander("📊 View Model Performance Metrics"):
        st.write(f"**Overall Accuracy:** {accuracy * 100:.2f}%")
        st.text("Detailed Classification Report:")
        st.text(report) # st.text keeps the exact spacing of the terminal output!
    
    # ---------------------------------------------------------
    # 4. Sidebar for User Inputs 
    # ---------------------------------------------------------
    st.sidebar.header("Input Measurements")
    st.sidebar.write("Slide to adjust values (in cm):")
    
    sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.8)
    sepal_width  = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 4.3)
    petal_width  = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.3)
    
    # ---------------------------------------------------------
    # 5. Make Live Predictions
    # ---------------------------------------------------------
    custom_inputs = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                                 columns=feature_list)
    
    prediction = model.predict(custom_inputs)[0]
    
    st.subheader("Model Prediction:")
    st.success(f"Based on these measurements, the model predicts: **{prediction.upper()}**")
    
except FileNotFoundError:
    st.error("Could not find Iris.csv. Please ensure it is in the same folder as app.py.")
"""
"""import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# 1. Set up the UI Header
# ---------------------------------------------------------
st.set_page_config(page_title="Iris Classifier", page_icon="🌸")
st.title("🌸 Iris Flower Classifier")
st.write("Adjust the sliders on the left to input new measurements, then see the results below!")

# ---------------------------------------------------------
# 2. Load Data, Train Model, & Evaluate (Cached)
# ---------------------------------------------------------
@st.cache_data
def load_train_and_evaluate():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_directory, "Iris.csv")
    
    df = pd.read_csv(csv_path)
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    
    X = df.iloc[:, :-1]  
    y = df.iloc[:, -1]   
    
    # Split the data to calculate real performance metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate overall performance metrics
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    return model, X.columns.tolist(), model.classes_, accuracy, report

try:
    model, feature_list, class_names, accuracy, report = load_train_and_evaluate()
    
    # ---------------------------------------------------------
    # 3. Sidebar for User Inputs 
    # ---------------------------------------------------------
    st.sidebar.header("Input New Measurements")
    st.sidebar.write("Slide to adjust values (in cm):")
    
    sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.8)
    sepal_width  = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 4.3)
    petal_width  = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.3)
    
    # ---------------------------------------------------------
    # 4. Make Live Predictions & Get Probabilities
    # ---------------------------------------------------------
    custom_inputs = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                                 columns=feature_list)
    
    prediction = model.predict(custom_inputs)[0]
    probabilities = model.predict_proba(custom_inputs)[0]
    
    st.subheader("1. Model Prediction:")
    st.success(f"Based on your input measurements, the model predicts: **{prediction.upper()}**")
    
    st.divider() 
    
    # ---------------------------------------------------------
    # 5. Display Confidence Metrics for the Input
    # ---------------------------------------------------------
    st.subheader("2. Prediction Confidence:")
    st.write("Here is how confident the model is about this specific flower:")
    
    for i, class_name in enumerate(class_names):
        prob_percentage = probabilities[i] * 100
        st.write(f"**{class_name.upper()}**: {prob_percentage:.1f}%")
        st.progress(float(probabilities[i]))
        
    st.divider()
    
    # ---------------------------------------------------------
    # 6. Display Overall Model Performance Metrics
    # ---------------------------------------------------------
    st.subheader("3. Overall Model Performance:")
    st.write("Curious how accurate the underlying model is? Check out the evaluation metrics below:")
    
    with st.expander("📊 View Detailed Performance Report"):
        st.write(f"**Overall Accuracy:** {accuracy * 100:.2f}%")
        st.text("Detailed Classification Report:")
        st.text(report)
        
except FileNotFoundError:
    st.error("Could not find Iris.csv. Please ensure it is in the same folder as app.py.")
"""
import os
import pandas as pd
import customtkinter as ctk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# 1. Setup Theme & Window
# ---------------------------------------------------------
ctk.set_appearance_mode("System")  # Follows your Mac's Dark/Light mode
ctk.set_default_color_theme("blue")

class IrisApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🌸 Iris Flower Classifier")
        self.geometry("550x700")
        
        # ---------------------------------------------------------
        # 2. Train Model on Startup
        # ---------------------------------------------------------
        self.model, self.feature_names, self.class_names, self.accuracy, self.report = self.load_and_train()
        
        # ---------------------------------------------------------
        # 3. Build the User Interface
        # ---------------------------------------------------------
        # Title
        self.title_label = ctk.CTkLabel(self, text="Iris Flower Classifier", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=(20, 5))
        
        self.subtitle = ctk.CTkLabel(self, text="Adjust sliders to input new measurements in cm:", text_color="gray")
        self.subtitle.pack(pady=(0, 20))

        # Sliders Frame
        self.slider_frame = ctk.CTkFrame(self)
        self.slider_frame.pack(pady=10, padx=20, fill="x")

        # Create Sliders & Labels
        self.sepal_len_val = ctk.StringVar(value="Sepal Length: 5.8")
        self.sepal_len_label = ctk.CTkLabel(self.slider_frame, textvariable=self.sepal_len_val)
        self.sepal_len_label.pack(pady=(10, 0))
        self.sepal_len = ctk.CTkSlider(self.slider_frame, from_=4.0, to=8.0, command=self.update_live)
        self.sepal_len.set(5.8)
        self.sepal_len.pack(pady=(0, 10))

        self.sepal_wid_val = ctk.StringVar(value="Sepal Width: 3.0")
        self.sepal_wid_label = ctk.CTkLabel(self.slider_frame, textvariable=self.sepal_wid_val)
        self.sepal_wid_label.pack()
        self.sepal_wid = ctk.CTkSlider(self.slider_frame, from_=2.0, to=4.5, command=self.update_live)
        self.sepal_wid.set(3.0)
        self.sepal_wid.pack(pady=(0, 10))

        self.petal_len_val = ctk.StringVar(value="Petal Length: 4.3")
        self.petal_len_label = ctk.CTkLabel(self.slider_frame, textvariable=self.petal_len_val)
        self.petal_len_label.pack()
        self.petal_len = ctk.CTkSlider(self.slider_frame, from_=1.0, to=7.0, command=self.update_live)
        self.petal_len.set(4.3)
        self.petal_len.pack(pady=(0, 10))

        self.petal_wid_val = ctk.StringVar(value="Petal Width: 1.3")
        self.petal_wid_label = ctk.CTkLabel(self.slider_frame, textvariable=self.petal_wid_val)
        self.petal_wid_label.pack()
        self.petal_wid = ctk.CTkSlider(self.slider_frame, from_=0.1, to=2.5, command=self.update_live)
        self.petal_wid.set(1.3)
        self.petal_wid.pack(pady=(0, 10))

        # Results Frame
        self.result_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.result_frame.pack(pady=10, padx=20, fill="both", expand=True)

        self.prediction_label = ctk.CTkLabel(self.result_frame, text="Prediction: Waiting...", font=ctk.CTkFont(size=18, weight="bold"))
        self.prediction_label.pack(pady=10)

        self.confidence_label = ctk.CTkLabel(self.result_frame, text="")
        self.confidence_label.pack(pady=5)

        # Metrics Button
        self.metrics_btn = ctk.CTkButton(self, text="Show Model Performance Metrics", command=self.show_metrics)
        self.metrics_btn.pack(pady=20)
        
        # Run an initial prediction based on default slider values
        self.update_live(None)

    # ---------------------------------------------------------
    # 4. Machine Learning Logic
    # ---------------------------------------------------------
    def load_and_train(self):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_directory, "Iris.csv")
        
        try:
            df = pd.read_csv(csv_path)
            if 'Id' in df.columns:
                df = df.drop(columns=['Id'])
            
            X = df.iloc[:, :-1]  
            y = df.iloc[:, -1]   
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)
            
            return model, X.columns.tolist(), model.classes_, accuracy, report
        except FileNotFoundError:
            print("Error: Could not find Iris.csv in the same folder.")
            self.destroy() # Closes app if data is missing

    def update_live(self, _):
        # Update text labels above sliders
        sl, sw = round(self.sepal_len.get(), 1), round(self.sepal_wid.get(), 1)
        pl, pw = round(self.petal_len.get(), 1), round(self.petal_wid.get(), 1)
        
        self.sepal_len_val.set(f"Sepal Length: {sl} cm")
        self.sepal_wid_val.set(f"Sepal Width: {sw} cm")
        self.petal_len_val.set(f"Petal Length: {pl} cm")
        self.petal_wid_val.set(f"Petal Width: {pw} cm")

        # Format input and make prediction
        inputs = pd.DataFrame([[sl, sw, pl, pw]], columns=self.feature_names)
        
        prediction = self.model.predict(inputs)[0]
        probabilities = self.model.predict_proba(inputs)[0]

        # Update UI with prediction
        self.prediction_label.configure(text=f"Prediction: {prediction.upper()}", text_color="#1f538d")
        
        # Update UI with confidence breakdown
        conf_text = "Confidence Breakdown:\n"
        for i, class_name in enumerate(self.class_names):
            conf_text += f"{class_name.capitalize()}: {probabilities[i]*100:.1f}%\n"
        self.confidence_label.configure(text=conf_text)

    def show_metrics(self):
        # Creates a popup window with your overall metrics
        popup = ctk.CTkToplevel(self)
        popup.title("Model Performance")
        popup.geometry("400x350")
        popup.attributes("-topmost", True) # Keeps popup in front

        title = ctk.CTkLabel(popup, text="Overall Model Performance", font=ctk.CTkFont(size=16, weight="bold"))
        title.pack(pady=10)

        acc = ctk.CTkLabel(popup, text=f"Overall Accuracy: {self.accuracy * 100:.2f}%")
        acc.pack(pady=5)

        report_box = ctk.CTkTextbox(popup, width=350, height=200)
        report_box.pack(pady=10, padx=20)
        report_box.insert("0.0", f"Classification Report:\n\n{self.report}")
        report_box.configure(state="disabled") # Makes it read-only

if __name__ == "__main__":
    app = IrisApp()
    app.mainloop()
