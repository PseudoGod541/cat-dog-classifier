# Full-Stack Cat vs. Dog Image Classifier

This project is a complete, end-to-end deep learning application that classifies uploaded images as either a cat or a dog. The system is built with a Keras/TensorFlow model, served via a FastAPI backend, and presented through an interactive Streamlit web interface. The entire application is containerized with Docker.



---

## 📋 Features

-   **Deep Learning Model**: A Convolutional Neural Network (CNN) trained on the Microsoft Cats vs. Dogs dataset.
-   **FastAPI Backend**: A high-performance REST API that accepts image uploads and returns predictions.
-   **Streamlit Frontend**: A user-friendly web app to upload an image and view the classification result and confidence score.
-   **Dockerized**: The backend and frontend are containerized into a single, cohesive application using Docker and Docker Compose.

---

## 🛠️ Tech Stack

-   **Backend**: Python, FastAPI, Uvicorn
-   **Machine Learning**: TensorFlow, Keras, Pillow, NumPy
-   **Frontend**: Streamlit, Requests
-   **Deployment**: Docker, Docker Compose

---

## 🚀 How to Run

To run this application, you need to have Docker and Docker Compose installed.
```bash
1. Clone the Repository


git clone <your-repository-url>
cd <your-project-directory>

2. Place Model Files
Ensure your trained model and label files are placed inside a models/ directory:

best_model.h5 (or .keras)

labels.json

3. Run with Docker Compose
This command builds the Docker image and starts both the API and frontend services.

docker-compose up --build

4. Access the Application
Streamlit Frontend: Open your browser and go to http://localhost:8501

FastAPI Backend Docs: Open your browser and go to http://localhost:8000/docs

📁 Project Structure
.
├── models/               # Contains the trained .h5 model and labels
├── app.py                # FastAPI application
├── streamlit_app.py      # Streamlit frontend application
├── Dockerfile            # Instructions to build the Docker image
├── docker-compose.yml    # Defines and runs the multi-container setup
├── .dockerignore         # Specifies files to ignore during build
└── requirements.txt      # Python dependencies
