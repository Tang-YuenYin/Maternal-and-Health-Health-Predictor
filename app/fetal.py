import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS"))
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = initialize_firebase()
# Dictionary to map predicted values to labels
label_map = {0: "normal", 1: "suspect", 2: "pathological"}

# Load the model
def load_model(model_path):
    model = pickle.load(open(model_path, "rb"))
    return model

loaded_model = load_model('fetalmodel.sav')
original_data = pd.read_csv('fetal_health.csv')  # Update with your actual dataset path

# Columns to exclude from description and data rows
columns_to_exclude = ['histogram_min', 'histogram_max', 'histogram_number_of_peaks',
                          'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
                          'histogram_median', 'histogram_variance', 'histogram_tendency', 'histogram_width']

# Filter out excluded columns
columns_for_description = [col for col in original_data.columns if col not in columns_to_exclude]

# Create sidebar
st.sidebar.subheader('Data Exploration')
section = st.sidebar.selectbox('Fetal Health', ('Data Rows', 'RiskLevel Counts', 'Data Description', 'FetalHealth Prediction'))

# Display selected section
if section == 'Data Rows':
    st.subheader("First 10 rows of the original dataset:")
    st.write(original_data[columns_for_description].head(10))  
elif section == 'RiskLevel Counts':
    st.subheader("Counts of RiskLevel for each Baseline Value:")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='baseline value', hue='fetal_health', data=original_data)  # Update with your visualization logic
    plt.title('Counts of RiskLevel for each Baseline Value')
    plt.xlabel('Baseline Value')
    plt.ylabel('Count')
    st.pyplot(plt)


    # Extract necessary columns for visualization
    fetal_health = original_data['fetal_health']
    baseline_heart_rate = original_data['baseline value']
    # Create a scatter plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=baseline_heart_rate, y=fetal_health,
                                 mode='markers',
                                 name='Fetal Health vs Baseline Heart Rate'))
    # Update layout of the plot
    fig.update_layout(
        title={
                'text': 'Relationship between Fetal Health and Baseline Value',
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
        xaxis_title='Baseline Value',
        yaxis_title='Fetal Health'
        )
    # Display the plot using Streamlit
    st.plotly_chart(fig)
elif section == 'Data Description':
    st.subheader("Description of Data:")
    st.write(original_data[columns_for_description].describe().T)
elif section == 'FetalHealth Prediction':
    st.subheader("Predict Fetal Health")
    # Get user input for prediction
    feature_names = ['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions',
                     'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
                     'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
                     'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability']

    # Get user input for prediction
    input_data = {}
    for feature in feature_names:
        input_data[feature] = st.number_input(f"Enter {feature}", value=0.000, format="%.3f")

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data], columns=feature_names)
        prediction = loaded_model.predict(input_df)
        prediction_label = label_map.get(prediction[0], "Unknown")
        # Display the predicted risk level
        st.subheader("Predicted Risk Level:")
        st.write(prediction_label)
        
        # Store the prediction result in session state
        st.session_state["prediction_result"] = {
            "input_data": input_data,
            "prediction_label": prediction_label,
            "prediction_value": int(prediction[0])
        }

# Check if there is a prediction result in session state and display the save button
if "prediction_result" in st.session_state:
    if st.button("Save to Mama's Journal"):
        result = st.session_state["prediction_result"]
        # Save prediction to Firebase
        doc_ref = db.collection("Maternal").add({
            "date": datetime.now().isoformat(),
            "input_data": result["input_data"],
            "prediction": result["prediction_label"],
            "prediction_value": result["prediction_value"]
        })
        st.success("Prediction saved to Firebase")


