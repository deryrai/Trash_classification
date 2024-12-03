import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained model
model = load_model('../model_rn.keras')

# Define class labels
class_labels = {0: 'cardboard', 1: 'glass', 2: 'metal' , 3 :'paper', 4: 'plastic' , 6 :'trash'}
# Define the image upload and prediction function
def predict_image(img):
    img = image.load_img(img, target_size=(224, 224))  # Adjust target_size as per your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the predicted class index
    return class_labels.get(predicted_class, 'Unknown')  # Map index to label

# Define the Streamlit app
def run():
    st.title('Klasifikasi Sampah')

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Predict and display the result
        result = predict_image(uploaded_file)
        st.write(f'Predicted class: {result}')

# Run the app
if __name__ == '__main__':
    run()


    # Predict and display the result
    result = predict_image(uploaded_file)
    st.write(f'Predicted class: {result}')
