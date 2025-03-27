import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Load and preprocess the image
img_path = 'green-cheek-conure.jpeg'  # Add a bird image with this name to the same folder
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Predict
preds = model.predict(img_array)
decoded = decode_predictions(preds, top=3)[0]

# Output predictions
print("Top predictions:")
for i, (imagenetID, label, prob) in enumerate(decoded):
    print(f"{i + 1}. {label} ({prob:.4f})")
