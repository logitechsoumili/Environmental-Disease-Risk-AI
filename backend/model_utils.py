import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load model once when server starts
tf.config.optimizer.set_jit(True)
model_eff = load_model("../model/efficientnet_model.keras", compile = False)

# Class labels
class_names = [
    "air_pollution",
    "garbage_dirty",
    "hygienic_environment",
    "stagnant_water"
]

# Image preprocessing
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array

# Prediction function
def predict_environment(img_path):
    img_array = preprocess_image(img_path)
    predictions = model_eff.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(predictions[0]) * 100)

    return predicted_class, confidence