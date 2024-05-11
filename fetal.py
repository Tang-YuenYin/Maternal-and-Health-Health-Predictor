import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
@st.cache_resource
def load_model(model_path):
    model = joblib.load(model_path)
    return model

model = load_model('ml_model.sav')

# Load the original dataset (assuming it's a CSV file)
original_data = pd.read_csv('fetal_health.csv')  # Update with your actual dataset path

# Columns to exclude from description and data rows
columns_to_exclude = ['histogram_min', 'histogram_max', 'histogram_number_of_peaks',
                      'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
                      'histogram_median', 'histogram_variance', 'histogram_tendency', 'histogram_width']

# Filter out excluded columns
columns_for_description = [col for col in original_data.columns if col not in columns_to_exclude]

# Create sidebar
st.sidebar.subheader('Data Exploration')
section = st.sidebar.selectbox('Section', ('Data Rows', 'RiskLevel Counts', 'Data Description', 'FetalHealth Prediction'))

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
        input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Make a prediction using the loaded model
    prediction = model.predict(input_df)

    # Display the predicted fetal health
    st.subheader("Predicted Fetal Health:")
    st.write(prediction[0])
