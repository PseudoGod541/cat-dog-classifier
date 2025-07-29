import streamlit as st
import requests
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Cat vs. Dog Classifier",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- FastAPI Backend URL ---
# This should point to the address where your FastAPI app is running.
# If using Docker Compose, this would be http://<service-name>:<port>/predict/
# For local testing, it's typically http://127.0.0.1:8000/predict/
API_URL = "http://api:8000/predict/"

# --- UI Design ---
st.title("🐱 Cat vs. Dog Image Classifier 🐶")
st.write(
    "Upload an image and the model will predict whether it contains a cat or a dog. "
    "The app sends the image to a backend API for processing."
)

# --- Image Uploader ---
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Click the button below to classify the image.")

    # --- Prediction Logic ---
    if st.button('Classify Image', key='predict_button'):
        # Show a spinner while waiting for the API response
        with st.spinner('Analyzing the image...'):
            try:
                # Prepare the file for the POST request
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

                # Send the POST request to the FastAPI backend
                response = requests.post(API_URL, files=files)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

                result = response.json()
                prediction = result.get('prediction')
                confidence = result.get('confidence')

                st.subheader("🧠 Prediction Result")

                if prediction == "Dog":
                    st.success(f"**Prediction: It's a Dog!** 🐕")
                elif prediction == "Cat":
                    st.success(f"**Prediction: It's a Cat!** �")
                else:
                    st.warning(f"**Prediction: {prediction}**")

                # Display confidence with a progress bar and metric
                st.metric(label="Confidence Score", value=f"{confidence*100:.2f}%")
                st.progress(confidence)

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the API. Please ensure the backend server is running. Error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
