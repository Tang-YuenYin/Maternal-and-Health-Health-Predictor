import streamlit as st
from firebase_admin import credentials, firestore, initialize_app
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Read the API key from the .env file
api_key = os.getenv("apiKey")
authDomain = os.getenv("authDomain")
projectId= os.getenv("projectId")
storageBucket=os.getenv("storageBucket")
messagingSenderId=os.getenv("messagingSenderId")
appId=os.getenv("appId")
measurementId=os.getenv("measurementId")

# Read the HTML file
with open("fetalbook.html", "r") as file:
    html_content = file.read()

# Replace the placeholder with the actual API key
html_content = html_content.replace("apiKey: '__API_KEY__'", f"apiKey: '{api_key}'")
html_content = html_content.replace("authDomain: '__AUTH_DOMAIN__'", f"authDomain: '{authDomain}'")
html_content = html_content.replace("projectId: '__PROJECT_ID__'", f"projectId: '{projectId}'")
html_content = html_content.replace("storageBucket: '__STORAGE_BUCKET__'", f"storageBucket: '{storageBucket}'")
html_content = html_content.replace("messagingSenderId: '__MESSAGING_SENDER_ID__'", f"messagingSenderId: '{messagingSenderId}'")
html_content = html_content.replace("appId: '__APP_ID__'", f"appId: '{appId}'")
html_content = html_content.replace("measurementId: '__MEASUREMENT_ID__'", f"measurementId: '{measurementId}'")

# Write the modified HTML content to a new file
with open("fetalbook.html", "w") as file:
    file.write(html_content)

# Read the HTML file
with open("maternalbook.html", "r") as file:
    html_content = file.read()

# Replace the placeholder with the actual API key
html_content = html_content.replace("apiKey: '__API_KEY__'", f"apiKey: '{api_key}'")
html_content = html_content.replace("authDomain: '__AUTH_DOMAIN__'", f"authDomain: '{authDomain}'")
html_content = html_content.replace("projectId: '__PROJECT_ID__'", f"projectId: '{projectId}'")
html_content = html_content.replace("storageBucket: '__STORAGE_BUCKET__'", f"storageBucket: '{storageBucket}'")
html_content = html_content.replace("messagingSenderId: '__MESSAGING_SENDER_ID__'", f"messagingSenderId: '{messagingSenderId}'")
html_content = html_content.replace("appId: '__APP_ID__'", f"appId: '{appId}'")
html_content = html_content.replace("measurementId: '__MEASUREMENT_ID__'", f"measurementId: '{measurementId}'")

# Write the modified HTML content to a new file
with open("maternalbook.html", "w") as file:
    file.write(html_content)
# # Initialize Firebase
# firebase_credentials_path = os.getenv("FIREBASE_CREDENTIALS")
# if firebase_credentials_path:
#     cred = credentials.Certificate(firebase_credentials_path)
#     initialize_app(cred)
#     db = firestore.client()
# else:
#     st.error("Firebase credentials not found. Please set the FIREBASE_CREDENTIALS environment variable.")

# # Define Streamlit API endpoints
# def get_Fetalrecords():
#     try:
#         records_ref = db.collection('Fetal')
#         docs = records_ref.stream()
#         records = [doc.to_dict() for doc in docs]
#         return records
#     except Exception as e:
#         st.error(f"Error fetching fetal records: {e}")
#         return []

# def get_Maternalrecords():
#     try:
#         records_ref = db.collection('Maternal')
#         docs = records_ref.stream()
#         records = [doc.to_dict() for doc in docs]
#         return records
#     except Exception as e:
#         st.error(f"Error fetching maternal records: {e}")
#         return []

# @st.cache_data
# def fetch_Fetalrecords():
#     return get_Fetalrecords()

# @st.cache_data
# def fetch_Maternalrecords():
#     return get_Maternalrecords()

# # Display records
# st.write("Fetal Records")
# fetal_records = fetch_Fetalrecords()
# st.write(fetal_records)

# st.write("Maternal Records")
# maternal_records = fetch_Maternalrecords()
# st.write(maternal_records)


