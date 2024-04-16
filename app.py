import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open("ml_model.sav", "rb"))

# Dictionary to map predicted values to labels
label_map = {0: "normal", 1: "suspect", 2: "pathological"}

# Function to make predictions
def predict_fetal_health(features):
    prediction = loaded_model.predict(features)
    return prediction

# Main function to run the Streamlit app
def main():
    st.title('Fetal Health Prediction')
    st.markdown(
        """
        <style>
            .header {
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
            }
            .sidebar {
                background-color: #34495e;
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
            }
            .main-content {
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
            }
            .button {
                background-color: #3498db;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            .button:hover {
                background-color: #2980b9;
            }
            .result-container {
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
        </style>
        """
    , unsafe_allow_html=True)

    # Add gif
    st.sidebar.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZWExMGRlMWZjYWI2YmE3ZWMxNTQ2MWEyYmEwYTg1NWMyOTA3MTM2ZiZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/pJbyeXszIZKNmdKbai/giphy.gif", use_column_width=True)

    # User input section
    st.sidebar.header('Input Features')
    st.sidebar.markdown("**Please provide the following input features:**")
    st.sidebar.markdown("*(You can adjust the sliders to input your values)*")

    # Add input fields for each feature without initializing the value
    feature_1 = st.sidebar.slider('Baseline value', min_value=0, max_value=200, step=1)
    feature_2 = st.sidebar.slider('Accelerations', min_value=0.0, max_value=5.0, step=0.001)
    feature_3 = st.sidebar.slider('Fetal movement', min_value=0.0, max_value=25.0, step=0.001)
    feature_4 = st.sidebar.slider('Uterine contractions', min_value=0.0, max_value=10.0, step=0.001)
    feature_5 = st.sidebar.slider('Light decelerations', min_value=0.0, max_value=5.0, step=0.001)
    feature_6 = st.sidebar.slider('Severe decelerations', min_value=0.0, max_value=5.0, step=0.001)
    feature_7 = st.sidebar.slider('Prolonged decelerations', min_value=0.0, max_value=5.0, step=0.001)
    feature_8 = st.sidebar.slider('Abnormal short term variability', min_value=0, max_value=100, step=1)
    feature_9 = st.sidebar.slider('Mean value of short term variability', min_value=0.0, max_value=10.0, step=0.1)
    feature_10 = st.sidebar.slider('Percentage of time with abnormal long term variability', min_value=0, max_value=100, step=1)
    feature_11 = st.sidebar.slider('Mean value of long term variability', min_value=0.0, max_value=100.0, step=1.0)

     # Button to trigger prediction
    if st.sidebar.button('Predict', key='predict_button'):
        # Collect input features into a list
        features = [[feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11]]  

        # Call prediction function
        prediction = predict_fetal_health(features)
        
        # Get the corresponding label for the prediction
        prediction_label = label_map.get(prediction[0], "Unknown")

        # Display prediction result
        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
        st.write('Predicted Fetal Health:')
        
        # Render HTML content based on prediction result
        if prediction_label == 'normal':
            st.markdown(
                """
                <h2 style="color: #2ecc71;"><i class="fas fa-swimmer"></i> Normal!</h2>
                <p>Congratulations! Your little swimmer is doing great! The assessment shows that your baby's health is right on track, just like an Olympic champion. Keep up the good work with your pregnancy journey!</p>
                """
            , unsafe_allow_html=True)
        elif prediction_label == 'suspect':
            st.markdown(
                """
                <h2 style="color: #f39c12;"><i class="fas fa-skull"></i> Suspect!</h2>
                <p>Uh-oh, we've spotted a few question marks during the assessment. It's like your baby is a little mischief-maker, keeping us on our toes! We're not sure yet what exactly is going on, but we'll keep a close eye on things and run some additional tests to unravel the mystery. Stay positive and let's investigate further!</p>
                """
            , unsafe_allow_html=True)
        elif prediction_label == 'pathological':
            st.markdown(
                """
                <h2 style="color: #e74c3c;"><i class="fas fa-skull"></i> Pathological!</h2>
                <p>Oh no! It seems like your little one is facing some hurdles in their health race. Don't worry, though! With the right care and attention, we can help them overcome these challenges and get back on the path to victory. Let's work together to ensure a healthy and happy outcome!</p>
                """
            , unsafe_allow_html=True)
        else:
            st.write("Error: Prediction result not recognized.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == '__main__':
    main()
