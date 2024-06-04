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

def get_predictions(baseline, acceleration, fetal_movement, uterine_cont, light_dec, severe_dec, prolongued_dec, astv, mean_stv, percentage_altv, mean_ltv):
    # Load the saved model
    model = pickle.load(open('fetalmodel.sav', 'rb'))
    # Load the scaler used for preprocessing
    scaler = pickle.load(open('scaler.sav', 'rb'))

    # Transform the input data
    input_data = scaler.transform([[baseline, acceleration, fetal_movement, uterine_cont, light_dec, severe_dec, prolongued_dec, astv, mean_stv, percentage_altv, mean_ltv]])
    
    # Make predictions
    prediction = model.predict(input_data)

    if prediction == 0:
        return 'normal'
    elif prediction == 1:
        return 'suspect'
    elif prediction == 2:
        return 'pathological'
    else:
        return 'error'

def main():
    st.sidebar.subheader('Data Exploration')
    section = st.sidebar.selectbox('Fetal Health', ('Data Rows', 'RiskLevel Counts', 'Data Description', 'FetalHealth Prediction'))

    original_data = pd.read_csv('fetal_health.csv')  # Update with your actual dataset path

    columns_to_exclude = ['histogram_min', 'histogram_max', 'histogram_number_of_peaks',
                              'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
                              'histogram_median', 'histogram_variance', 'histogram_tendency', 'histogram_width']

    columns_for_description = [col for col in original_data.columns if col not in columns_to_exclude]

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

        fetal_health = original_data['fetal_health']
        baseline_heart_rate = original_data['baseline value']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=baseline_heart_rate, y=fetal_health,
                                     mode='markers',
                                     name='Fetal Health vs Baseline Heart Rate'))
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
        st.plotly_chart(fig)
    elif section == 'Data Description':
        st.subheader("Description of Data:")
        st.write(original_data[columns_for_description].describe().T)
    elif section == 'FetalHealth Prediction':
        st.subheader("Predict Fetal Health")
        baseline = st.number_input("Enter Baseline Value", value=0.0, format="%.3f")
        acceleration = st.number_input("Enter Accelerations", value=0.0, format="%.3f")
        fetal_movement = st.number_input("Enter Fetal Movement", value=0.0, format="%.3f")
        uterine_cont = st.number_input("Enter Uterine Contractions", value=0.0, format="%.3f")
        light_dec = st.number_input("Enter Light Decelerations", value=0.0, format="%.3f")
        severe_dec = st.number_input("Enter Severe Decelerations", value=0.0, format="%.3f")
        prolongued_dec = st.number_input("Enter Prolongued Decelerations", value=0.0, format="%.3f")
        astv = st.number_input("Enter Abnormal Short Term Variability", value=0.0, format="%.3f")
        mean_stv = st.number_input("Enter Mean Value of Short Term Variability", value=0.0, format="%.3f")
        percentage_altv = st.number_input("Enter Percentage of Time with Abnormal Long Term Variability", value=0.0, format="%.3f")
        mean_ltv = st.number_input("Enter Mean Value of Long Term Variability", value=0.0, format="%.3f")

        if st.button("Predict"):
            result = get_predictions(baseline, acceleration, fetal_movement, uterine_cont, light_dec, severe_dec, prolongued_dec, astv, mean_stv, percentage_altv, mean_ltv)
            st.write(f"Predicted Result: {result}")

            # Store the prediction result in session state
            st.session_state["prediction_result"] = {
                "input_data": {
                    'baseline value': baseline,
                    'accelerations': acceleration,
                    'fetal_movement': fetal_movement,
                    'uterine_contractions': uterine_cont,
                    'light_decelerations': light_dec,
                    'severe_decelerations': severe_dec,
                    'prolongued_decelerations': prolongued_dec,
                    'abnormal_short_term_variability': astv,
                    'mean_value_of_short_term_variability': mean_stv,
                    'percentage_of_time_with_abnormal_long_term_variability': percentage_altv,
                    'mean_value_of_long_term_variability': mean_ltv
                },
                "prediction_label": result
            }

        # Ensure session state is not empty before saving
        if "prediction_result" in st.session_state:
            if st.button("Save to Mama's Journal"):
                result = st.session_state["prediction_result"]
                # Save prediction to Firebase
                try:
                    db.collection("Maternal").add({
                        "date": datetime.now().isoformat(),
                        "input_data": result["input_data"],
                        "prediction": result["prediction_label"]
                    })
                    st.success("Prediction saved to Firebase")
                except Exception as e:
                    st.error(f"Error saving prediction: {e}")

if __name__ == "__main__":
    main()
