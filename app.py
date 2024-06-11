
import streamlit as st
from fastai.vision.all import load_learner, PILImage
import requests
from io import BytesIO

# Set the title of the Streamlit app
st.title('Bird or Forest Classifier')

# Cache the model loading function to avoid reloading the model on every interaction
@st.cache(allow_output_mutation=True)
def load_model():
    return load_learner('bird_forest_model.pkl')

# Load the model
model = load_model()

# Define the prediction function
def predict(image):
    img = PILImage.create(image)  # Convert the image to a format suitable for the model
    pred, _, probs = model.predict(img)  # Make prediction
    return pred, probs

# Button to show an example bird image
if st.button('Example Image - Bird'):
    url = 'https://example.com/example_bird.jpg'  # Replace with the URL to an example bird image
    response = requests.get(url)
    img = BytesIO(response.content)  # Convert the image content to a bytes object
    st.image(img, caption='Example Image - Bird', use_column_width=True)  # Display the image in the app
    label, probability = predict(img)  # Predict the label and probability
    st.write(f'Prediction: {label}, Probability: {probability.max().item():.6f}')  # Display the prediction result

# Button to show an example forest image
if st.button('Example Image - Forest'):
    url = 'https://example.com/example_forest.jpg'  # Replace with the URL to an example forest image
    response = requests.get(url)
    img = BytesIO(response.content)  # Convert the image content to a bytes object
    st.image(img, caption='Example Image - Forest', use_column_width=True)  # Display the image in the app
    label, probability = predict(img)  # Predict the label and probability
    st.write(f'Prediction: {label}, Probability: {probability.max().item():.6f}')  # Display the prediction result

# File uploader for users to upload their own images
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)  # Display the uploaded image
    label, probability = predict(uploaded_file)  # Predict the label and probability
    st.write(f'Prediction: {label}, Probability: {probability.max().item():.6f}')  # Display the prediction result

# Text input for users to enter an image URL
url = st.text_input('Or enter image URL')
if url:
    response = requests.get(url)
    img = BytesIO(response.content)  # Convert the image content to a bytes object
    st.image(img, caption='Image from URL', use_column_width=True)  # Display the image in the app
    label, probability = predict(img)  # Predict the label and probability
    st.write(f'Prediction: {label}, Probability: {probability.max().item():.6f}')  # Display the prediction result
    