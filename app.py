

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image
import pickle
from keras.layers import GlobalAveragePooling2D
from keras.models import Model

image_size = (224, 224)

image_name = "user_eye.png"

PREDICTION_LABELS = ["Cataract", "Normal"]

PREDICTION_LABELS.sort()

#write functions and come back when we need to write them
@st.cache_resource
#HW: read up on on caching
def get_convext_model():
  base_model = tf.keras.applications.ConvNeXtXLarge(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  model_frozen = Model(inputs=base_model.input,outputs=x)
  return model_frozen

@st.cache_resource
def load_sklearn_models(model_path):
    with open(model_path, 'rb') as model_file:
        final_model = pickle.load(model_file)
    return final_model


def featurization(image_path, model):
  img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_batch = np.expand_dims(img_array, axis=0)
  img_preprocessed = preprocess_input(img_batch)
  predictions = model.predict(img_preprocessed)
  return predictions


convext_featurized_model = get_convext_model()
cataract_model = load_sklearn_models("ConvNexXtlarge_MLP_best_model")



st.title("Cataract Image Predictor")

st.image(
    "https://mediniz-images-2018-100.s3.ap-south-1.amazonaws.com/post-images/chokhm_1663869443.png",
    caption = "Cataract Eyes")


st.header("About the web app")
with st.expander("Web App 🌐"):
  st.subheader("Eye disease classification")
  st.write("The Web App helps predict, from the image, whether or not the user has cataract")

with st.expander("How to use"):
  st.write("1. Scroll down to the 2 tabs that allow you to upload an image or take a picture if your eye")
  st.write("2. Click on the method that you want to use")
  st.write("3. For uploading an image, click on the 'Browse files' button that is present and choose the fundus image you want to upload")
  st.write("4. For taking a picture, position your eye so that the eye is within the camera frame and click 'Take photo'")
  st.write("5. Wait for the model to detect the cataracts, and scroll down for the diagnosis")

tab1, tab2 = st.tabs(["Image Upload 👁️", "Camera Upload 📷"])
with tab1:
  image = st.file_uploader(label="Upload an image",accept_multiple_files=False, help="Upload an image to classify them")
  if image:
    image_type = image.type.split("/")[-1]
    if image_type not in ['jpg','jpeg','png','jfif']:
      st.error("Invalid file type : {}".format(image.type), icon="🚨)")
    else:
      st.image(image, caption="User uploaded image")
      if image:
        user_image = Image.open(image)
        user_image.save(image_name)
        with st.spinner("Processing..."):
          image_features = featurization(image_name, convext_featurized_model)
          model_predict = cataract_model.predict(image_features)
          model_predict_proba = cataract_model.predict_proba(image_features)
          probability = model_predict_proba[0][model_predict[0]]
        st.write("Model Prediction:", model_predict)
        st.write("Model Prediction Probabilities:", model_predict_proba)
        col1, col2 = st.columns(2)
        with col1:
          st.header("Disease Type")
          st.subheader("{}".format(PREDICTION_LABELS[model_predict[0]]))
        with col2:
          st.header("Prediction Probability")
          st.subheader("{}".format(probability))

with tab2:
  cam_image = st.camera_input("Take a photo of the eye")
  if cam_image:
    st.image(image, caption="Captured image")
    user_image = Image.open(cam_image)
    user_image.save(image_name)
    with st.spinner("Processing..."):
          image_features = featurization(image_name, convext_featurized_model)
          model_predict = cataract_model.predict(image_features)
          model_predict_proba = cataract_model.predict_proba(image_features)
          probability = model_predict_proba[0][model_predict[0]]
    #st.write("Model Prediction:", model_predict)
    #st.write("Model Prediction Probabilities:", model_predict_proba)
    col1, col2 = st.columns(2)
    with col1:
          st.header("Disease Type")
          st.subheader("{}".format(PREDICTION_LABELS[model_predict[0]]))
    with col2:
          st.header("Prediction Probability")
          st.subheader("{}".format(probability))

#Does not work first time when I run it
#Github file does not run properly; cannot read app.py
