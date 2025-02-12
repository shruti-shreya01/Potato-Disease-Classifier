


# import streamlit as st
# import pickle
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# from tensorflow.keras import layers 
# import numpy as np
# import os

# IMAGE_SIZE= 256
# # Function to load and preprocess the uploaded image
# def load_and_preprocess_image(image):
#     IMAGE_SIZE = 256 # Adjust according to your model's input size
#     img = load_img(image, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array = img_array / 255.0  # Normalize to [0, 1]
#     return img_array

# # Resize and rescale model input
# resize_and_rescale = tf.keras.Sequential([
#     layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
#     layers.Rescaling(1./255),
# ])



# # Path to the pickle file
# file_path = r"C:\Users\shrey\Downloads\potato_pickle.pkl"

# # Check if the file exists before loading
# if not os.path.exists(file_path):
#     st.error(f"File not found: {file_path}")
# else:
#     # Load the model from the pickle file
#     with open(file_path, "rb") as f:
#         data = pickle.load(f)

#     # Reconstruct the model from the architecture
#     model = tf.keras.models.model_from_json(data["architecture"])

#     # # Load the model weights
#     # model.set_weights(data["weights"])

#     # # Apply the resizing and rescaling
#     # model.layers[0] = resize_and_rescale

#     # # Compile the model (optional)
#     # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     # Streamlit app interface
#     st.title("Potato Leaf Disease Classification")
#     st.write("Upload an image of a potato leaf to classify the disease.")

#     def reset_session():
#         st.session_state["uploaded_file"] = None
#         st.session_state["output"] = None
#     # File uploader for image input
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # if uploaded_file is not None:
    #     # Load and preprocess the uploaded image
    #     img_array = load_and_preprocess_image(uploaded_file)

    #             # Predict the class of the leaf disease
    #     prediction = model.predict(img_array)
    #     predicted_class = np.argmax(prediction, axis=1)[0]
    #     confidence = np.max(prediction)  # Confidence score

    #     # Display the uploaded image
    #     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)



    #     # Map predicted class to the disease name (assuming you have a dictionary for class names)
    #     class_names = {0: "Early Blight", 1: "Late Blight", 2: "Healthy"}  # Example classes
    #     disease_name = class_names.get(predicted_class, "Unknown")

    #     # Display the prediction result
    #     st.write(f"Predicted Disease: **{disease_name}**")

    #             # Display the prediction result and confidence score

    #     st.write(f"Confidence Score: **{confidence:.2f}**")

#     if st.button("Rerun"):
#       st.experimental_rerun()



#     if st.button("Reset"):
#         reset_session()

#     st.sidebar.title("About")
#     st.sidebar.info("This app is designed to help farmers and agronomists identify diseases in potato leaves using AI technology.")

#     st.sidebar.subheader("About the Model")
#     st.sidebar.write("This model classifies potato leaf diseases with high accuracy. The classes are:")
#     st.sidebar.write("- Early Blight")
#     st.sidebar.write("- Late Blight")
#     st.sidebar.write("- Healthy")





import streamlit as st
import pickle
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image, ImageOps
import numpy as np

IMAGE_SIZE = 256

# Function to load and preprocess the uploaded image
def load_and_preprocess_image(image):
    img = load_img(image, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Path to the pickle file
file_path = "potato_pickle.pkl"

# Check if file exists and load the model
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Reconstruct the model from the architecture
    model = tf.keras.models.model_from_json(data["architecture"])
else:
    st.error(f"Model file not found at path: {file_path}")
    st.stop()  # Stop execution if file is not found

class_names = {0: "Early Blight", 1: "Late Blight", 2: "Healthy"}

# Initialize session state variables
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "confidence" not in st.session_state:
    st.session_state["confidence"] = None

# Streamlit app interface
st.title("Potato Leaf Disease Classification")
st.write("Upload an image of a potato leaf to classify the disease.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploaded_file")

if uploaded_file is not None:
    # Load and preprocess the uploaded image
    img_array = load_and_preprocess_image(uploaded_file)

    # Predict the class of the leaf disease
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)  # Confidence score

    # Store results in session state
    st.session_state["prediction"] = predicted_class
    st.session_state["confidence"] = confidence

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Map predicted class to the disease name
    disease_name = class_names.get(predicted_class, "Unknown")

    # Log raw prediction
    print("Raw Prediction:", prediction)  # Log raw prediction
    # Log predicted class and confidence
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")

    # Display the results in Streamlit
    st.write(f"Predicted Disease: **{disease_name}**")
    st.write(f"Confidence Score: **{confidence:.2f}**")




# Use a button to rerun the app conditionally
if st.button("Rerun"):
    # Check if necessary state is initialized before rerunning
    if st.session_state["prediction"] is not None and st.session_state["confidence"] is not None:
        st.rerun()
    else:
        st.warning("Please upload an image first.")

st.sidebar.title("About")
st.sidebar.info("This app is designed to help farmers and agronomists identify diseases in potato leaves using AI technology.")

st.sidebar.subheader("About the Model")
st.sidebar.write("This model classifies potato leaf diseases with high accuracy. The classes are:")
st.sidebar.write("- Early Blight")
st.sidebar.write("- Late Blight")
st.sidebar.write("- Healthy")










# import streamlit as st
# import pickle
# import tensorflow as tf
# import os
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# import numpy as np

# IMAGE_SIZE = 256

# # Function to load and preprocess the uploaded image
# def load_and_preprocess_image(image):
#     img = load_img(image, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array = img_array / 255.0  # Normalize to [0, 1]
#     return img_array

# # Path to the pickle file
# file_path = "potato_pickle_final (1).pkl"

# # Check if file exists and load the model
# if os.path.exists(file_path):
#     with open(file_path, "rb") as f:
#         data = pickle.load(f)

#     # Reconstruct the model from the architecture
#     model = tf.keras.models.model_from_json(data["architecture"])

#     # Load the weights from the file
#     model.load_weights(data["weights"])
# else:
#     st.error(f"Model file not found at path: {file_path}")
#     st.stop()  # Stop execution if file is not found
# class_names = {0: "Early Blight", 1: "Late Blight", 2: "Healthy"}


# # Initialize session state variables
# if "prediction" not in st.session_state:
#     st.session_state["prediction"] = None
# if "confidence" not in st.session_state:
#     st.session_state["confidence"] = None

# # Streamlit app interface
# st.title("Potato Leaf Disease Classification")
# st.write("Upload an image of a potato leaf to classify the disease.")

# # File uploader for image input
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploaded_file")

# if uploaded_file is not None:
#     # Load and preprocess the uploaded image
#     img_array = load_and_preprocess_image(uploaded_file)

#     # Predict the class of the leaf disease
#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     confidence = np.max(prediction)  # Confidence score

#     # Store results in session state
#     st.session_state["prediction"] = predicted_class
#     st.session_state["confidence"] = confidence

#     # Display the uploaded image
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

#     # Map predicted class to the disease name (assuming you have a dictionary for class names)
#     class_names = {0: "Early Blight", 1: "Late Blight", 2: "Healthy"}  # Example classes
#     disease_name = class_names.get(predicted_class, "Unknown")
#     # Inside the Streamlit app, after making a prediction
#     prediction = model.predict(img_array)
#     print("Raw Prediction:", prediction)  # Log raw prediction
    
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     confidence = np.max(prediction)  # Confidence score
    
#     # Log predicted class and confidence
#     print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")
    
#     # Display the results in Streamlit
#     st.write(f"Predicted Disease: **{disease_name}**")
#     st.write(f"Confidence Score: **{confidence:.2f}**")


# # Use a button to rerun the app conditionally
# if st.button("Rerun"):
#     # Check if necessary state is initialized before rerunning
#     if st.session_state["prediction"] is not None and st.session_state["confidence"] is not None:
#         st.rerun()
#     else:
#         st.warning("Please upload an image first.")

# st.sidebar.title("About")
# st.sidebar.info("This app is designed to help farmers and agronomists identify diseases in potato leaves using AI technology.")

# st.sidebar.subheader("About the Model")
# st.sidebar.write("This model classifies potato leaf diseases with high accuracy. The classes are:")
# st.sidebar.write("- Early Blight")
# st.sidebar.write("- Late Blight")
# st.sidebar.write("- Healthy")









